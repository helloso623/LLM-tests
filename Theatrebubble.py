# mic_circle_derivative.py
import argparse, queue, sys, math, time
import numpy as np
import sounddevice as sd
import pygame, pywt

# ---- args ----
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["volume","freq"], default="volume")
parser.add_argument("--samplerate", type=int, default=12000)
parser.add_argument("--blocksize", type=int, default=1024)
parser.add_argument("--channels", type=int, default=1)
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--fft_thresh", type=float, default=2.2)   # denoise gate
parser.add_argument("--spring_hz", type=float, default=3.5)    # response speed
parser.add_argument("--damping",   type=float, default=0.95)   # 0.85..1.0
parser.add_argument("--k_err",     type=float, default=0.7)    # extra vel boost ∝ error
args = parser.parse_args()

# ---- constants added (no new CLI params) ----
VOL_FLOOR   = 0.06   # ignore volume below this normalized level
IDLE_AFTER  = 1.2    # seconds below floor before we freeze at rest

# ---- audio ----
q = queue.Queue()
def audio_callback(indata, frames, time_info, status):
    if status: pass
    q.put(indata.copy())

try:
    stream = sd.InputStream(
        samplerate=args.samplerate,
        blocksize=args.blocksize,
        channels=args.channels,
        dtype="float32",
        device=args.device,
        callback=audio_callback,
    )
    stream.start()
except Exception as e:
    print(f"Audio stream error: {e}", file=sys.stderr); sys.exit(1)

# ---- helpers ----
def stft_gate_block(x: np.ndarray, thresh_factor: float) -> np.ndarray:
    if x.size == 0: return x
    x = x.reshape(-1).astype(np.float32)
    w = np.hanning(x.size).astype(np.float32)
    X = np.fft.rfft(w * x)
    mag = np.abs(X)
    m = float(mag.mean()) + 1e-12
    X *= (mag >= thresh_factor * m)
    xr = np.fft.irfft(X, n=x.size)
    w_rms = np.sqrt((w*w).mean()) + 1e-12
    return (xr / w_rms).astype(np.float32)

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0

def loudness_norm(x: np.ndarray) -> float:
    e = rms(x) + 1e-9
    return float(np.log1p(150*e)/np.log1p(150))  # 0..~1

class WaveletPitch:
    # Morlet CWT dominant frequency with EMA smoothing (α=1/6)
    def __init__(self, sr, fmin=85.0, fmax=950.0, n_scales=72, alpha=1/6):
        self.sr = sr
        self.fmin, self.fmax = fmin, fmax
        self.freqs = np.geomspace(fmin, fmax, num=n_scales).astype(np.float32)
        fc = pywt.central_frequency("morl")
        self.scales = (fc * sr / self.freqs).astype(np.float32)
        self.prev = None
        self.alpha = float(alpha)
    def __call__(self, x: np.ndarray) -> float:
        if x.size == 0: return 0.0
        coefs, _ = pywt.cwt(x.reshape(-1), self.scales, "morl", sampling_period=1.0/self.sr)
        power = (np.abs(coefs)**2).mean(axis=1)
        f = float(self.freqs[int(np.argmax(power))])
        if not np.isfinite(f): f = 0.0
        if self.prev is None: self.prev = f
        self.prev = (1 - self.alpha)*self.prev + self.alpha*f
        return self.prev

pitch = WaveletPitch(args.samplerate)

# ---- UI ----
W, H = 600, 400
pygame.init()
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("Mic Circle — derivative spring (with floor + idle)")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 24)

base_radius = 40
max_radius  = min(W, H)//2 - 20
vol_gain    = 320.0  # px per normalized loudness

# state
r = float(base_radius)  # radius
v = 0.0                 # radius velocity [px/s]
last_audio = np.zeros(args.blocksize, dtype=np.float32)
quiet_since = time.time()  # time since last above-floor activity

running = True
while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT: running = False
        if e.type == pygame.KEYDOWN and e.key in (pygame.K_ESCAPE, pygame.K_q): running = False

    try:
        while True:
            last_audio = q.get_nowait().reshape(-1)
    except queue.Empty:
        pass

    den = stft_gate_block(last_audio, args.fft_thresh)

    # target radius
    lv = loudness_norm(den)
    # volume floor: ignore sub-threshold noise
    active = lv >= VOL_FLOOR
    if active:
        quiet_since = time.time()

    idle = (time.time() - quiet_since) >= IDLE_AFTER

    if idle:
        # freeze at rest until activity resumes
        r = float(base_radius)
        v = 0.0
        r_t = base_radius
    else:
        if args.mode == "freq":
            f = pitch(den)
            fmin, fmax = 300.0, 500.0
            if f <= 0:
                frac = 0.0
            else:
                f_clamped = min(max(f, fmin), fmax)
                frac = math.log(f_clamped / fmin) / math.log(fmax / fmin)
            pitch_factor = 0.01
            vol_factor   = 0.1
            # if below floor, treat volume as zero
            lv_eff = lv if active else 0.0
            combined = (pitch_factor * frac + vol_factor * lv_eff) / (pitch_factor + vol_factor + 2)
        else:
            lv_eff = lv if active else 0.0
            combined = lv_eff

        r_t = base_radius + combined * (max_radius - base_radius) * (5 if args.mode=="freq" else 1)
        r_t = float(np.clip(r_t, base_radius, max_radius))

        # derivative spring with error-proportional velocity boost
        dt = max(1e-6, args.blocksize / args.samplerate)
        w  = 1.5*math.pi*args.spring_hz
        z  = args.damping*10
        err = r_t - r
        a = -1*z*w*v + (w*w)*err
        v += (0.00085*a + args.k_err * err*0.05) * dt
        r += v * np.sqrt(abs(v+0.001)) * dt

    # clamp and light friction
    r = float(np.clip(r, base_radius, max_radius))/1.05
    v *= 0.99

    # ---- draw ----
    screen.fill((5,5,7))
    cx, cy = W//2, H//2
    color_core = (160, 255, 255)
    color_ring = (80, 180, 190)
    glow_intensity = 60 if active and not idle else 10
    for i in range(4):
        alpha = 1 - i/4
        col = [int(color_ring[j]*alpha) for j in range(3)]
        pygame.draw.circle(screen, col, (cx,cy), int(r + i*2 + glow_intensity*0.03), width=1)
    pygame.draw.circle(screen, color_core, (cx, cy), int(r), width=2)

    hud = f"{'IDLE' if idle else 'LIVE'}  vol:{lv:.2f}"
    text = font.render(hud, True, (120,170,190))
    screen.blit(text, (20, 20))

    pygame.display.flip()
    clock.tick(60)

# ---- cleanup ----
stream.stop(); stream.close(); pygame.quit(); sys.exit(0)
