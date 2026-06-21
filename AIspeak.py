import asyncio, threading, time, json, queue, os, sys, math
from pathlib import Path
import numpy as np
from bleak import BleakScanner

# ---------- CONFIG ----------
TARGET_MAC = "B0:C7:DE:C2:7F:1E"
FCD2_UUID = "0000fcd2-0000-1000-8000-00805f9b34fb"

EV_SINGLE = 1
EV_DOUBLE = 2
EV_TRIPLE = 3
EV_LONG   = 4
EV_LONG_DOUBLE = 5
EV_LONG_TRIPLE = 6


LLM_BACKEND = "ollama"
OLLAMA_MODEL = "qwen3:4b-instruct"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"

DATA_DIR = Path.home() / "shelly_ai_data"
CHAT_LOG = DATA_DIR / "chat.jsonl"

# UI (bigger by default)
WIN_W, WIN_H = 820, 820
FPS = 60

TTS_RATE = 175
TTS_VOLUME = 1.0

OCR_LANGS = ["en"]

SYSTEM_PROMPT = """You are a desktop AI assistant.
Be direct.
If user says stop/exit/cancel: confirm and stop the current action.
If unclear: ask ONE short question.
When user asks to analyze a screenshot: infer based on OCR text and describe what it likely shows, plus uncertainties.
"""

# ================= AUTO DEPENDENCY BOOTSTRAP =================
import subprocess, importlib

def ensure(pkg, import_name=None):
    try:
        return importlib.import_module(import_name or pkg)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
        return importlib.import_module(import_name or pkg)

pygame = ensure("pygame")
sd = ensure("sounddevice")
pyttsx3 = ensure("pyttsx3")
requests = ensure("requests")
PIL = ensure("Pillow", "PIL")
from PIL import ImageGrab  # ✅

# SpeechRecognition is the "works now" STT for Py3.12
sr = ensure("SpeechRecognition", "speech_recognition")

# easyocr can be heavy; keep script alive if it fails
_easyocr_ok = True
try:
    easyocr = ensure("easyocr")
except Exception:
    _easyocr_ok = False

# =============================================================

def now_ts() -> float:
    return time.time()

def log_turn(role: str, text: str):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with CHAT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": now_ts(), "role": role, "text": text}, ensure_ascii=False) + "\n")

def parse_bthome(raw: bytes):
    if not raw or len(raw) < 2:
        return None, None
    i = 1
    packet_id = None
    button_ev = None
    while i < len(raw):
        obj = raw[i]; i += 1
        if i >= len(raw): break
        val = raw[i]; i += 1
        if obj == 0x00:
            packet_id = val
        elif obj == 0x3A:
            button_ev = val
    return packet_id, button_ev


# ---------- TTS (RELIABLE) ----------
class TTS:
    def __init__(self, rate=TTS_RATE, volume=TTS_VOLUME):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        self.engine.setProperty("volume", volume)
        self._lock = threading.Lock()
        self._speaking = False

    def speak_async(self, text: str):
        def run():
            with self._lock:
                self._speaking = True
                try:
                    self.engine.say(text)
                    self.engine.runAndWait()
                except Exception as e:
                    print("TTS error:", e)
                finally:
                    self._speaking = False
        threading.Thread(target=run, daemon=True).start()

    def stop(self):
        with self._lock:
            try:
                self.engine.stop()
            except:
                pass
            self._speaking = False

    def is_speaking(self) -> bool:
        return self._speaking


# ---------- STT + mic level (Google STT via SpeechRecognition) ----------
class SpeechIn:
    def __init__(self):
        self.rec = sr.Recognizer()
        self.mic = sr.Microphone()

    def listen_once(self, seconds=5.5):
        with self.mic as source:
            self.rec.adjust_for_ambient_noise(source, duration=0.3)
            audio = self.rec.listen(source, phrase_time_limit=seconds)
        try:
            return self.rec.recognize_google(audio).strip()
        except Exception:
            return ""

    def mic_level_stream(self):
        q = queue.Queue()
        def cb(indata, frames, time_info, status):
            x = indata.astype(np.float32).reshape(-1)
            rms = float(np.sqrt(np.mean(x * x)) + 1e-9)
            q.put(rms)
        stream = sd.InputStream(
            samplerate=12000,
            blocksize=1024,
            channels=1,
            dtype="float32",
            callback=cb,
        )
        stream.start()
        return stream, q


# ---------- OCR screenshot ----------
class ScreenOCR:
    def __init__(self, langs):
        self.ok = _easyocr_ok
        self.langs = langs
        self.reader = None
        if self.ok:
            try:
                self.reader = easyocr.Reader(langs)
            except Exception:
                self.ok = False
                self.reader = None

    def screenshot_ocr(self) -> str:
        img = ImageGrab.grab().convert("RGB")
        if not self.ok or self.reader is None:
            return ""  # no OCR available
        arr = np.array(img)
        results = self.reader.readtext(arr)
        return " ".join([r[1] for r in results]).strip()


# ---------- LLM (Ollama) ----------
class LLM:
    def __init__(self):
        self.history = []

    def _trim(self, msgs, max_pairs=6):
        sysm = msgs[0:1]
        rest = msgs[1:]
        if len(rest) > 2 * max_pairs:
            rest = rest[-2 * max_pairs:]
        return sysm + rest

    def reply(self, user_text: str) -> str:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history + [{"role": "user", "content": user_text}]
        msgs = self._trim(msgs, 6)

        if LLM_BACKEND != "ollama":
            return "Only ollama backend is enabled in this build."

        try:
            payload = {"model": OLLAMA_MODEL, "messages": msgs, "stream": False}
            r = requests.post(OLLAMA_URL, json=payload, timeout=120)
            # ✅ don't hard crash if Ollama down
            if r.status_code != 200:
                return f"Ollama error {r.status_code}: {r.text[:200]}"
            data = r.json()
            out = (data.get("message", {}) or {}).get("content", "").strip()
            if not out:
                out = "No response from Ollama."
        except Exception as e:
            out = f"Ollama not reachable: {e}"

        self.history += [{"role": "user", "content": user_text}, {"role": "assistant", "content": out}]
        self.history = self._trim([{"role": "system", "content": SYSTEM_PROMPT}] + self.history, 6)[1:]
        return out


# ---------- Orb UI (Resizable + Fullscreen + Better BG + Speaker colors) ----------
class OrbUI:
    def __init__(self):
        pygame.init()
        self.flags = pygame.RESIZABLE
        self.screen = pygame.display.set_mode((WIN_W, WIN_H), self.flags)
        pygame.display.set_caption("AI Orb (F11 fullscreen)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.running = True

        self.fullscreen = False
        self.w, self.h = WIN_W, WIN_H

        self.base = 70.0
        self.r = self.base
        self.v = 0.0

        self.mode = "idle"
        self.last_text = ""
        self._t0 = time.time()

        self.spring_hz = 3.2
        self.damping = 0.90

        self.set_mode("idle")

    def set_mode(self, m: str):
        self.mode = m
        if m == "listen":
            self.core_color = (0, 255, 120)     # GREEN user
            self.glow_color = (0, 180, 90)
        elif m == "speak":
            self.core_color = (80, 160, 255)    # BLUE ai
            self.glow_color = (40, 120, 255)
        else:
            self.core_color = (140, 200, 210)
            self.glow_color = (80, 120, 140)

    def set_text(self, s: str):
        self.last_text = (s or "")[:240]

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            self.w, self.h = self.screen.get_size()
        else:
            self.screen = pygame.display.set_mode((self.w, self.h), self.flags)

    def tick(self, mic_level: float, speaking: bool):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.running = False
            elif e.type == pygame.VIDEORESIZE and not self.fullscreen:
                self.w, self.h = e.w, e.h
                self.screen = pygame.display.set_mode((self.w, self.h), self.flags)
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_F11:
                    self.toggle_fullscreen()

        t = time.time() - self._t0

        # ---- background gradient + subtle waves ----
        for y in range(self.h):
            c = int(10 + 10 * math.sin((y / max(1, self.h)) * math.pi))
            pygame.draw.line(self.screen, (6, 8, 14 + c), (0, y), (self.w, y))
        for i in range(6):
            y = int((math.sin(t * 0.4 + i) + 1) * 0.5 * self.h)
            pygame.draw.line(self.screen, (10, 20, 40), (0, y), (self.w, y), 1)

        # scale orb base to window
        min_dim = min(self.w, self.h)
        self.base = max(40.0, min_dim * 0.10)
        self.r = max(self.r, self.base)  # ✅ prevents resize glitches

        # target radius
        if self.mode == "listen":
            lv = min(1.0, mic_level * 10.0)
            rt = self.base + lv * (min_dim * 0.22)
        elif self.mode == "speak" or speaking:
            rt = self.base + (min_dim * 0.10) + (min_dim * 0.02) * math.sin(2 * math.pi * 3.2 * t)
        else:
            rt = self.base + (min_dim * 0.01) * math.sin(2 * math.pi * 0.8 * t)

        # spring
        dt = 1.0 / FPS
        w = 2 * math.pi * self.spring_hz
        z = self.damping
        err = rt - self.r
        a = (w * w) * err - 2 * z * w * self.v
        self.v += a * dt
        self.r += self.v * dt

        cx, cy = self.w // 2, self.h // 2

        # glow rings
        for i in range(12):
            alpha = 1 - i / 12
            col = (
                int(self.glow_color[0] * alpha),
                int(self.glow_color[1] * alpha),
                int(self.glow_color[2] * alpha),
            )
            pygame.draw.circle(self.screen, col, (cx, cy), int(self.r + i * 3), width=2)

        # core
        pygame.draw.circle(self.screen, self.core_color, (cx, cy), int(self.r), width=3)

        # HUD
        hud = "F11 fullscreen | long=talk mode | single=screenshot | double=stop"
        txt = self.font.render(hud, True, (110, 165, 175))
        self.screen.blit(txt, (12, 12))

        # subtitle (color-coded)
        if self.last_text:
            lines = self._wrap(self.last_text, 60)[:5]
            y = self.h - 20 * (len(lines) + 1)
            for ln in lines:
                color = (120, 255, 160) if self.mode == "listen" else (140, 180, 255)
                t2 = self.font.render(ln, True, color)
                self.screen.blit(t2, ((self.w - t2.get_width()) // 2, y))
                y += 20

        pygame.display.flip()
        self.clock.tick(FPS)

    def _wrap(self, s: str, n: int):
        words = s.split()
        out, cur = [], ""
        for w in words:
            if len(cur) + len(w) + 1 <= n:
                cur = (cur + " " + w).strip()
            else:
                out.append(cur)
                cur = w
        if cur:
            out.append(cur)
        return out

    def close(self):
        pygame.quit()


# ---------- Shelly BLE listener ----------
class ShellyListener:
    def __init__(self, mac: str, uuid: str, out_q: "queue.Queue[int]"):
        self.mac = mac.upper()
        self.uuid = uuid
        self.out_q = out_q
        self.last_packet_id = None
        self.last_time = 0.0
        self.dedup_idle = 0.25

    async def run(self):
        def handle(dev, adv):
            if (dev.address or "").upper() != self.mac:
                return
            raw = (adv.service_data or {}).get(self.uuid)
            if not raw:
                return
            pkt, ev = parse_bthome(raw)
            if ev is None:
                return
            now = time.monotonic()
            if pkt is not None:
                if pkt == self.last_packet_id:
                    return
                self.last_packet_id = pkt
            else:
                if (now - self.last_time) < self.dedup_idle:
                    return
            self.last_time = now
            self.out_q.put(ev)

        async with BleakScanner(detection_callback=handle):
            await asyncio.Event().wait()


# ---------- App ----------
class App:
    def __init__(self):
        self.ui = OrbUI()
        self.tts = TTS()
        self.stt = SpeechIn()
        self.ocr = ScreenOCR(OCR_LANGS)
        self.llm = LLM()

        self.btn_q = queue.Queue()
        self.convo_mode = False
        self.busy = False

        self.mic_stream, self.mic_q = self.stt.mic_level_stream()
        self.mic_level = 0.0

        threading.Thread(target=self._run_ble, daemon=True).start()

    def _run_ble(self):
        asyncio.run(ShellyListener(TARGET_MAC, FCD2_UUID, self.btn_q).run())

    def toggle_convo(self):
        self.convo_mode = not self.convo_mode
        if not self.convo_mode:
            self.ui.set_mode("idle")
            self.tts.stop()
            self.ui.set_text("Conversation mode OFF")
        else:
            self.ui.set_mode("listen")
            self.ui.set_text("Conversation mode ON (talk now)")

    def _do_screenshot_analysis(self):
        try:
            self.ui.set_mode("listen")
            self.ui.set_text("Screenshot… OCR…")

            ocr_text = self.ocr.screenshot_ocr()
            if not ocr_text:
                prompt = "I took a screenshot. OCR found no readable text. Ask me what window/app it was and what I want to do."
            else:
                prompt = f"Analyze this screenshot based on OCR text. Explain what it likely shows and what I should do next.\n\nOCR TEXT:\n{ocr_text}"

            log_turn("user", "[screenshot]")
            reply = self.llm.reply(prompt)
            log_turn("assistant", reply)

            self.ui.set_text(reply)
            self.ui.set_mode("speak")
            self.tts.speak_async(reply)
        except Exception as e:
            self.ui.set_text(f"Screenshot/OCR error: {e}")
            self.ui.set_mode("idle")
        finally:
            self.busy = False

    def handle_single_press(self):
        if self.busy:
            return
        self.busy = True
        threading.Thread(target=self._do_screenshot_analysis, daemon=True).start()

    def handle_double_press(self):
        self.tts.stop()
        self.ui.set_text("Stopped speaking")
        self.ui.set_mode("listen" if self.convo_mode else "idle")

    def convo_loop(self):
        def run():
            while self.convo_mode and self.ui.running:
                if self.busy:
                    time.sleep(0.05)
                    continue
                self.busy = True
                try:
                    self.ui.set_mode("listen")
                    self.ui.set_text("Listening…")
                    text = self.stt.listen_once(5.5)
                    if not self.convo_mode:
                        break
                    if not text:
                        self.ui.set_text("…")
                        self.busy = False
                        continue
                    if text.strip().lower() in ("stop", "exit", "cancel", "quit"):
                        self.toggle_convo()
                        break

                    log_turn("user", text)
                    self.ui.set_text(f"You: {text}")

                    reply = self.llm.reply(text)
                    log_turn("assistant", reply)

                    self.ui.set_text(reply)
                    self.ui.set_mode("speak")
                    self.tts.speak_async(reply)

                    while self.tts.is_speaking() and self.convo_mode and self.ui.running:
                        time.sleep(0.03)

                except Exception as e:
                    self.ui.set_text(f"Convo error: {e}")
                    self.ui.set_mode("idle")
                    time.sleep(0.2)
                finally:
                    self.busy = False

            if self.ui.running:
                self.ui.set_mode("idle")
        threading.Thread(target=run, daemon=True).start()

    def update_mic_level(self):
        try:
            while True:
                v = self.mic_q.get_nowait()
                self.mic_level = 0.85 * self.mic_level + 0.15 * min(1.0, v * 12.0)
        except queue.Empty:
            pass

    def process_button_events(self):
        try:
            while True:
                ev = self.btn_q.get_nowait()
                if ev == EV_LONG:
                    self.toggle_convo()
                    if self.convo_mode:
                        self.convo_loop()
                elif ev == EV_SINGLE:
                    self.handle_single_press()
                elif ev == EV_DOUBLE:
                    self.handle_double_press()
        except queue.Empty:
            pass

    def run(self):
        self.ui.set_mode("idle")
        self.ui.set_text("Ready. Long press = talk mode. Single press = screenshot. F11 fullscreen.")
        while self.ui.running:
            self.update_mic_level()
            self.process_button_events()
            self.ui.tick(self.mic_level, self.tts.is_speaking())

        try:
            self.mic_stream.stop()
            self.mic_stream.close()
        except:
            pass
        self.ui.close()


if __name__ == "__main__":
    App().run()
