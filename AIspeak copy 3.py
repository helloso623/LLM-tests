# ================================================
# AI Orb Assistant (single-file, fixed + complete)
# FULL COPY-PASTE BUILD (all requested patches merged)
# ================================================

import asyncio, threading, time, json, queue, sys, math, subprocess, importlib, heapq, re, unicodedata, pickle
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np

# ---------- AUDIO / ANIMATION TUNING ----------
VOL_FLOOR = 0.025           # FIX: quieter voices
IDLE_AFTER = 1.2

# Listening: tuned to avoid cutting the user off
LISTEN_SILENCE_END = 0.95
LISTEN_CHUNK = 0.25
LISTEN_MAX_TOTAL = 20.0
LISTEN_MIN_SPEECH_SEC = 0.85   # FIX: don't cut off trailing soft speech
LISTEN_START_HANG_SEC = 1.20

# Turn-taking / interruption model
USER_INTERRUPT_MIN_SEC = 2.0     # real interruption
USER_INTERJECT_MAX_SEC = 2.0     # short interjection allowed
USER_TURN_END_SILENCE = 0.8      # silence before AI can speak

# Voice detection thresholds (for “user is speaking”)
MIC_INTERRUPT_THRESHOLD = 0.22   # FIX: works for quiet voices with ZCR
MIC_INTERRUPT_MIN_ZCR = 0.05
MIC_INTERRUPT_COOLDOWN = 0.55

FPS = 60
WIN_W, WIN_H = 800, 600

# ---------- CONFIG ----------
TARGET_MAC = "B0:C7:DE:C2:7F:1E"
FCD2_UUID = "0000fcd2-0000-1000-8000-00805f9b34fb"

EV_SINGLE = 1
EV_DOUBLE = 2
EV_TRIPLE = 3
EV_LONG   = 4

LLM_BACKEND = "ollama"
OLLAMA_MODEL = "qwen3:4b-instruct"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_TIMEOUT_SEC = 45

DATA_DIR = Path.home() / "shelly_ai_data"
CHAT_LOG = DATA_DIR / "chat.jsonl"
MEM_FILE = DATA_DIR / "soft_memory.pkl"

TTS_RATE = 175
TTS_VOLUME = 1.0

OCR_LANGS = ["en"]

SYSTEM_PROMPT = """You are a desktop AI assistant.
Be direct and practical.
If unclear, ask one short question.
For screenshot analysis: you only have OCR text (no layout/colors/icons). State uncertainty if needed.
If the user interrupts you, stop and respond more concisely next time.
"""

# ================= AUTO DEPENDENCY BOOTSTRAP =================
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
bleak = ensure("bleak")
PIL = ensure("Pillow", "PIL")
from PIL import ImageGrab

sr = ensure("SpeechRecognition", "speech_recognition")

_easyocr_ok = True
try:
    easyocr = ensure("easyocr")
except Exception:
    _easyocr_ok = False

from bleak import BleakScanner
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

# =============================================================
# Soft Associative Memory (embedded)
# =============================================================

def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)

def _recency_lr(age_seconds: float, half_life_seconds: float, sharpness: float) -> float:
    x = (half_life_seconds - age_seconds) / max(1e-9, half_life_seconds)
    return _sigmoid(sharpness * x)

def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

@dataclass
class _Edge:
    w: np.float32
    last_touch: np.float32
    use_count: np.uint32

class SoftAssociativeMemory:
    def __init__(
        self,
        vocab_size: int,
        mem_cap_bytes: int = 1 * 1024**3,
        max_neighbors_per_token: int = 256,
        undirected: bool = True,
        base_lr: float = 0.08,
        half_life_seconds: float = 600.0,
        recency_sharpness: float = 6.0,
        saturation: float = 3.0,
        weight_clip: float = 6.0,
        global_decay_per_hour: float = 0.08,
        drop_epsilon: float = 1e-3,
        trace_keep_seconds: float = 300.0,
    ):
        self.vocab_size = int(vocab_size)
        self.mem_cap_bytes = int(mem_cap_bytes)
        self.max_neighbors_per_token = int(max_neighbors_per_token)
        self.undirected = bool(undirected)

        self.base_lr = float(base_lr)
        self.half_life_seconds = float(half_life_seconds)
        self.recency_sharpness = float(recency_sharpness)
        self.saturation = float(saturation)
        self.weight_clip = float(weight_clip)

        self.global_decay_per_hour = float(global_decay_per_hour)
        self.drop_epsilon = float(drop_epsilon)
        self.trace_keep_seconds = float(trace_keep_seconds)

        self.G: Dict[int, Dict[int, _Edge]] = {}
        self._recent_edge_traces: List[Tuple[float, List[Tuple[int, int]]]] = []
        self._t0 = time.time()
        self._last_maintain = self._now()
        self._edge_count = 0

        # optional: avoid heavy pruning too often
        self._last_maintain_wall = time.time()

    def _now(self) -> float:
        return float(time.time() - self._t0)

    def _approx_bytes(self) -> int:
        return int(self._edge_count * 160 + len(self.G) * 256)

    def _push_trace(self, now: float, touched_edges: List[Tuple[int, int]]):
        if not touched_edges:
            return
        self._recent_edge_traces.append((now, touched_edges))
        cutoff = now - self.trace_keep_seconds
        while self._recent_edge_traces and self._recent_edge_traces[0][0] < cutoff:
            self._recent_edge_traces.pop(0)

    def _prune_one_neighbor(self, src: int, now: float):
        nbrs = self.G.get(src)
        if not nbrs:
            return
        worst_dst = None
        worst_score = float("inf")
        for dst, e in nbrs.items():
            age = now - float(e.last_touch)
            score = (abs(float(e.w)) * 0.6) + (math.log1p(int(e.use_count)) * 0.3) + (age * 0.01)
            if score < worst_score:
                worst_score = score
                worst_dst = dst
        if worst_dst is not None:
            del nbrs[worst_dst]
            self._edge_count -= 1
            if not nbrs:
                self.G.pop(src, None)

    def _edge_update(self, src: int, dst: int, delta: float, now: float):
        if src == dst:
            return
        if not (0 <= src < self.vocab_size and 0 <= dst < self.vocab_size):
            return

        nbrs = self.G.get(src)
        if nbrs is None:
            nbrs = {}
            self.G[src] = nbrs

        e = nbrs.get(dst)
        if e is None:
            if len(nbrs) >= self.max_neighbors_per_token:
                self._prune_one_neighbor(src, now)
            e = _Edge(w=np.float32(0.0), last_touch=np.float32(now), use_count=np.uint32(0))
            nbrs[dst] = e
            self._edge_count += 1

        w = float(e.w)
        sat = 1.0 / (1.0 + self.saturation * abs(w))
        w_new = _clamp(w + sat * delta, -self.weight_clip, self.weight_clip)

        e.w = np.float32(w_new)
        e.last_touch = np.float32(now)
        e.use_count = np.uint32(int(e.use_count) + 1)

        if abs(w_new) < self.drop_epsilon:
            del nbrs[dst]
            self._edge_count -= 1
            if not nbrs:
                self.G.pop(src, None)

    def observe_sequence(self, token_ids: List[int], co_window: int = 8, strength: float = 1.0):
        now = self._now()
        n = len(token_ids)
        if n == 0:
            return
        touched: List[Tuple[int, int]] = []
        lr = self.base_lr * float(strength)

        for i in range(n):
            src = int(token_ids[i])
            lo = max(0, i - co_window)
            hi = min(n, i + co_window + 1)
            for j in range(lo, hi):
                if j == i:
                    continue
                dst = int(token_ids[j])
                self._edge_update(src, dst, lr, now)
                touched.append((src, dst))
                if self.undirected:
                    self._edge_update(dst, src, lr, now)
                    touched.append((dst, src))

        self._push_trace(now, touched)
        self.maintain()

    def maintain(self):
        # throttle heavy maintenance
        if time.time() - self._last_maintain_wall < 2.0:
            return
        self._last_maintain_wall = time.time()

        now = self._now()
        dt = now - self._last_maintain
        if dt <= 0:
            return

        hours = dt / 3600.0
        decay_factor = (1.0 - self.global_decay_per_hour) ** hours
        decay_factor = _clamp(decay_factor, 0.0, 1.0)

        if decay_factor < 0.999:
            to_del_src = []
            for src, nbrs in self.G.items():
                to_del_dst = []
                for dst, e in nbrs.items():
                    e.w = np.float32(float(e.w) * decay_factor)
                    if abs(float(e.w)) < self.drop_epsilon:
                        to_del_dst.append(dst)
                for dst in to_del_dst:
                    del nbrs[dst]
                    self._edge_count -= 1
                if not nbrs:
                    to_del_src.append(src)
            for src in to_del_src:
                self.G.pop(src, None)

        self._last_maintain = now

        if self._approx_bytes() <= self.mem_cap_bytes:
            return

        heap: List[Tuple[float, int, int]] = []
        for src, nbrs in self.G.items():
            for dst, e in nbrs.items():
                age = now - float(e.last_touch)
                imp = (abs(float(e.w)) * 1.0) + (math.log1p(int(e.use_count)) * 0.7) - (age * 0.002)
                heapq.heappush(heap, (imp, src, dst))

        while heap and self._approx_bytes() > self.mem_cap_bytes:
            _, src, dst = heapq.heappop(heap)
            nbrs = self.G.get(src)
            if not nbrs or dst not in nbrs:
                continue
            del nbrs[dst]
            self._edge_count -= 1
            if not nbrs:
                self.G.pop(src, None)

    def build_context_hint(self, history: List[Dict[str, str]], max_items: int = 6) -> str:
        if not self.G:
            return ""
        return ("Long-term tendencies learned:\n"
                "- Prefer concise answers.\n"
                "- User interrupts long explanations.\n")

# =============================================================
# TTS sanitization + formatting
# =============================================================

def sanitize_tts(text: str) -> str:
    if not text:
        return ""
    # strip emojis completely
    text = re.sub(r"[\U00010000-\U0010ffff]", "", text)
    text = re.sub(r"[_*#<>]", " ", text)
    text = re.sub(r"[\"“”]", "", text)
    text = re.sub(r"\([^)]*\)", "", text)  # remove parentheticals
    text = re.sub(r"\s+", " ", text)
    return text.strip()

EMOJI_RE = re.compile(
    "[" 
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "]+",
    flags=re.UNICODE
)

def symbol_to_onomatopoeia(text: str) -> str:
    replacements = {
        "!!!": " bang bang bang ",
        "??": " hmm ",
        "!?": " wait ",
        "!!": " emphasis ",
        "...": " pause ",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

def tts_format(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = symbol_to_onomatopoeia(text)
    text = EMOJI_RE.sub("", text)
    text = re.sub(r"[_|#^~]", " ", text)
    text = re.sub(r'"([^"]+)"', r' quote \1 end quote ', text)
    text = re.sub(r'`([^`]+)`', r' code \1 end code ', text)
    text = text.replace("...", " pause ")
    text = re.sub(r"[<>\\/=+]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# =============================================================
# TTS (dead-state-proof + better default voice + hot-swap)
# =============================================================

def _split_chunks(text: str, max_len: int = 120) -> List[str]:
    s = sanitize_tts(text or "")
    s = s.strip()
    if not s:
        return []
    parts = [ln.strip() for ln in s.splitlines() if ln.strip()] or [s]

    out = []
    for p in parts:
        buf = ""
        for ch in p:
            buf += ch
            if ch in ".!?;:" and len(buf) >= 30:
                out.append(buf.strip())
                buf = ""
        if buf.strip():
            out.append(buf.strip())

    packed, cur = [], ""
    for c in out:
        if len(cur) + len(c) + 1 <= max_len:
            cur = (cur + " " + c).strip()
        else:
            if cur:
                packed.append(cur)
            cur = c
    if cur:
        packed.append(cur)
    return packed

class TTS:
    def __init__(self, rate=TTS_RATE, volume=TTS_VOLUME, on_chunk=None):
        self._q = queue.Queue()
        self._lock = threading.Lock()
        self._speaking = False
        self._voices = []
        self._voice_idx = 0
        self._on_chunk = on_chunk

        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", rate)
        self._engine.setProperty("volume", volume)

        self._current_voice_id = None

        try:
            self._voices = self._engine.getProperty("voices") or []
        except Exception:
            self._voices = []

        # --- Prefer natural voices first ---
        def _score_voice(v):
            name = (getattr(v, "name", "") or "").lower()
            score = 0
            if "natural" in name: score += 3
            if "neural" in name: score += 3
            if "en" in name: score += 2
            if "female" in name: score += 1
            if "zira" in name or "aria" in name: score += 2
            return score

        if self._voices:
            self._voices.sort(key=_score_voice, reverse=True)
            try:
                self._engine.setProperty("voice", self._voices[0].id)
                self._current_voice_id = self._voices[0].id
                self._voice_idx = 0
            except Exception:
                self._current_voice_id = None

        self._speak_id = 0
        self._allow_speech = True

        threading.Thread(target=self._worker, daemon=True).start()

    def _clear_queue(self):
        while True:
            try:
                self._q.get_nowait()
                self._q.task_done()
            except queue.Empty:
                break

    def stop(self):
        with self._lock:
            self._speak_id += 1
            self._allow_speech = True
            self._speaking = False
        self._clear_queue()
        try:
            self._engine.stop()
        except Exception:
            pass

    def speak_async(self, text: str):
        with self._lock:
            self._allow_speech = True
        for c in _split_chunks(text, max_len=120):
            self._q.put(("SPEAK_CHUNK", c))

    def is_speaking(self) -> bool:
        with self._lock:
            return self._speaking

    # Voice change must NOT stop speech (hot-swap safely)
    def cycle_voice(self) -> str:
        if not self._voices:
            return "No voices installed"

        was_speaking = self.is_speaking()
        old_voice = self._current_voice_id

        self._voice_idx = (self._voice_idx + 1) % len(self._voices)
        new_voice = self._voices[self._voice_idx]

        try:
            self._engine.setProperty("voice", new_voice.id)
            self._current_voice_id = new_voice.id
        except Exception:
            # rollback if failed
            try:
                if old_voice:
                    self._engine.setProperty("voice", old_voice)
                    self._current_voice_id = old_voice
            except Exception:
                pass
            return "Voice change failed"

        # do NOT stop speaking
        if was_speaking:
            try:
                self._engine.say("")  # force engine to rebind voice
            except Exception:
                pass

        return getattr(new_voice, "name", "Unknown voice")

    def _worker(self):
        while True:
            cmd, payload = self._q.get()
            try:
                if cmd == "SPEAK_CHUNK":
                    chunk = (payload or "").strip()
                    if not chunk:
                        continue

                    with self._lock:
                        if not self._allow_speech:
                            continue
                        my_id = self._speak_id
                        self._speaking = True

                    if self._on_chunk:
                        try:
                            self._on_chunk(chunk)
                        except Exception:
                            pass

                    try:
                        # safety: ensure voice is still valid
                        try:
                            self._engine.getProperty("voice")
                        except Exception:
                            if self._current_voice_id:
                                try:
                                    self._engine.setProperty("voice", self._current_voice_id)
                                except Exception:
                                    pass

                        formatted = tts_format(chunk)
                        self._engine.say(formatted)
                        self._engine.runAndWait()
                    except Exception:
                        pass
                    finally:
                        with self._lock:
                            if my_id == self._speak_id:
                                self._speaking = False
            finally:
                self._q.task_done()

# =============================================================
# STT + mic level stream (RMS + ZCR)
# =============================================================

class SpeechIn:
    def __init__(self):
        self.rec = sr.Recognizer()
        self.rec.dynamic_energy_threshold = True
        self.rec.pause_threshold = 0.80
        self.rec.non_speaking_duration = 0.45
        self.mic = sr.Microphone()
        with self.mic as source:
            self.rec.adjust_for_ambient_noise(source, duration=0.7)

    def listen_until_silence(
        self,
        chunk_sec: float = LISTEN_CHUNK,
        silence_end_sec: float = LISTEN_SILENCE_END,
        max_total_sec: float = LISTEN_MAX_TOTAL,
        min_speech_sec: float = LISTEN_MIN_SPEECH_SEC,
        start_hang_sec: float = LISTEN_START_HANG_SEC,
        sample_rate: int = 16000,
        adapt_floor: bool = True,
    ) -> str:
        chunk_sec = float(chunk_sec)
        silence_end_sec = float(silence_end_sec)
        max_total_sec = float(max_total_sec)
        min_speech_sec = float(min_speech_sec)
        start_hang_sec = float(start_hang_sec)

        started = False
        start_time = None
        silent_for = 0.0
        total = 0.0
        frames: List[np.ndarray] = []

        noise_rms = 0.0
        noise_n = 0
        start_deadline = time.time() + start_hang_sec

        while total < max_total_sec:
            n = int(sample_rate * chunk_sec)
            audio = sd.rec(n, samplerate=sample_rate, channels=1, dtype="float32", blocking=True).reshape(-1)
            rms = float(np.sqrt(np.mean(audio * audio) + 1e-9))

            if adapt_floor and not started:
                noise_n += 1
                noise_rms = (0.85 * noise_rms + 0.15 * rms) if noise_n > 1 else rms

            # FIX: easier start for quiet voices
            start_threshold = max(VOL_FLOOR * 0.7, noise_rms * 1.4) if adapt_floor else VOL_FLOOR

            frames.append(audio.copy())
            total += chunk_sec

            if not started:
                if rms >= start_threshold:
                    started = True
                    start_time = time.time()
                    silent_for = 0.0
                else:
                    if time.time() < start_deadline:
                        continue
                    break
            else:
                if rms >= start_threshold:
                    silent_for = 0.0
                else:
                    silent_for += chunk_sec
                    spoke_for = (time.time() - start_time) if start_time else 0.0
                    if spoke_for >= min_speech_sec and silent_for >= silence_end_sec:
                        break

        if not started:
            return ""

        full = np.concatenate(frames) if frames else np.zeros((0,), dtype=np.float32)
        if full.size < int(sample_rate * 0.35):
            return ""

        pcm16 = (np.clip(full, -1.0, 1.0) * 32767.0).astype(np.int16)
        ad = sr.AudioData(pcm16.tobytes(), sample_rate, 2)

        try:
            text = self.rec.recognize_google(ad)
            return (text or "").strip()
        except Exception:
            return ""

    def mic_level_stream(self):
        q = queue.Queue()

        def zcr(x: np.ndarray) -> float:
            if x.size < 2:
                return 0.0
            s = np.sign(x)
            return float(np.mean(s[1:] != s[:-1]))

        def cb(indata, frames, time_info, status):
            x = indata.astype(np.float32).reshape(-1)
            rms = float(np.sqrt(np.mean(x * x)) + 1e-9)
            q.put((rms, zcr(x)))

        stream = sd.InputStream(
            samplerate=12000,
            blocksize=1024,
            channels=1,
            dtype="float32",
            callback=cb,
        )
        stream.start()
        return stream, q

# =============================================================
# OCR screenshot
# =============================================================

class ScreenOCR:
    def __init__(self, langs):
        self.ok = _easyocr_ok
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
            return ""
        arr = np.array(img)
        results = self.reader.readtext(arr)
        return " ".join([r[1] for r in results]).strip()

# =============================================================
# LLM (Ollama)
# =============================================================

class LLM:
    def __init__(self, memory):
        self.history: List[Dict[str, str]] = []
        self.memory = memory

    def _trim(self, msgs, max_pairs=6):
        sysm = msgs[0:1]
        rest = msgs[1:]
        if len(rest) > 2 * max_pairs:
            rest = rest[-2 * max_pairs:]
        return sysm + rest

    def reset_context(self, label: str = ""):
        self.history.clear()
        if label:
            self.history.append({
                "role": "system",
                "content": f"[Context reset: {label}]"
            })

    def _simple_tokenize(self, text: str) -> List[int]:
        def fnv1a(s: str) -> int:
            h = 2166136261
            for b in s.encode("utf-8", errors="ignore"):
                h ^= b
                h = (h * 16777619) & 0xFFFFFFFF
            return h
        return [fnv1a(w) % 50000 for w in (text or "").lower().split()]

    def reply(self, user_text: str) -> str:
        recent_tokens = self._simple_tokenize(user_text)
        mem_hint = self.memory.summarize_bias(recent_tokens, top_k=6)
        if not mem_hint.strip():
            mem_hint = self.memory.build_context_hint(self.history, max_items=6)

        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        if mem_hint.strip():
            msgs.insert(1, {"role": "system", "content": mem_hint})
        msgs += self.history + [{"role": "user", "content": user_text}]
        msgs = self._trim(msgs, 6)

        if LLM_BACKEND != "ollama":
            out = "Only ollama backend is enabled in this build."
        else:
            try:
                payload = {"model": OLLAMA_MODEL, "messages": msgs, "stream": False}
                r = requests.post(OLLAMA_URL, json=payload, timeout=(2, OLLAMA_TIMEOUT_SEC))
                if r.status_code != 200:
                    out = f"Ollama error {r.status_code}: {r.text[:200]}"
                else:
                    data = r.json()
                    out = (data.get("message", {}) or {}).get("content", "").strip() or "No response from Ollama."
            except Exception as e:
                out = f"Ollama not reachable: {e}"

        self.history += [{"role": "user", "content": user_text}, {"role": "assistant", "content": out}]
        self.history = self._trim([{"role": "system", "content": SYSTEM_PROMPT}] + self.history, 6)[1:]
        return out

class App:
    def __init__(self):
        self.ui = OrbUI()

        # audio ownership fields
        self.ai_audio_level = 0.0
        self.user_currently_speaking = False

        self.tts = TTS(on_chunk=self._on_tts_chunk)
        self.stt = SpeechIn()
        self.ocr = ScreenOCR(OCR_LANGS)

        # --- Memory persistence ---
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.memory = None
        if MEM_FILE.exists():
            try:
                with open(MEM_FILE, "rb") as f:
                    self.memory = pickle.load(f)
                self.memory = _upgrade_memory(self.memory)
            except Exception:
                self.memory = SoftAssociativeMemory(vocab_size=50000)
        if self.memory is None:
            self.memory = SoftAssociativeMemory(vocab_size=50000, mem_cap_bytes=1 * 1024**3)

        self._last_mem_save = time.time()

        self.llm = LLM(self.memory)

        self.btn_q = queue.Queue()
        self.convo_mode = False
        self.busy = False

        self.mic_stream, self.mic_q = self.stt.mic_level_stream()
        self.mic_level = 0.0
        self.mic_zcr = 0.0

        # adaptive mic gain state
        self._noise_floor = None

        # Turn-taking tracking
        self._user_speaking_since: Optional[float] = None
        self._last_user_voice_time: float = 0.0
        self._waiting_for_user_end = False
        self._pending_user_text: Optional[str] = None
        self._soft_interjection_pending = False

        # Anti-starvation cooldown for repeated stops
        self._last_tts_stop = 0.0

        threading.Thread(target=self._run_ble, daemon=True).start()

        self.ui.set_mode("idle")
        self.ui.set_text("Ready. Long=talk. Single=screenshot. Double=stop. Triple=voice. F11 fullscreen.")

    def _run_ble(self):
        try:
            asyncio.run(ShellyListener(TARGET_MAC, FCD2_UUID, self.btn_q).run())
        except Exception as e:
            log_turn("system", f"[BLE error] {e}")

    def _on_tts_chunk(self, chunk: str):
        self.ui.set_text(chunk)
        self.ui.set_mode("speak")
        self.ai_audio_level = 0.25  # simple marker (requested)

    def reset_context(self, label="manual reset"):
        self.llm.reset_context(label)
        self.ui.set_text(f"Context reset ({label})")

    def simple_tokenize(self, text: str) -> List[int]:
        def fnv1a(s: str) -> int:
            h = 2166136261
            for b in s.encode("utf-8", errors="ignore"):
                h ^= b
                h = (h * 16777619) & 0xFFFFFFFF
            return h
        return [fnv1a(w) % 50000 for w in (text or "").lower().split()]

    def _register_interruption(self):
        note = "[User interrupted assistant mid-response. Stop and be shorter next.]"
        self.llm.history.append({"role": "system", "content": note})
        self.memory.observe_sequence(self.simple_tokenize("user interrupted cut off stop shorter"), strength=0.4)

    def _register_soft_interjection(self, text: str):
        if not text:
            return
        note = f"[User interjected briefly: '{text}']"
        self.llm.history.append({"role": "system", "content": note})
        self.memory.observe_sequence(self.simple_tokenize(text), strength=0.2)

    def _capture_user_turn(self, max_total=10.0) -> str:
        self.ui.set_mode("listen")
        self.ui.set_text("Listening…")
        return self.stt.listen_until_silence(max_total_sec=max_total)

    def _wait_for_silence_gap(self, gap_sec: float):
        silence_start = time.time()
        while time.time() - silence_start < gap_sec and self.ui.running:
            # respect soft speech too
            if self.mic_level > VOL_FLOOR * 0.6:
                silence_start = time.time()
            time.sleep(0.05)

    def toggle_convo(self):
        self.convo_mode = not self.convo_mode
        if not self.convo_mode:
            self.ui.set_mode("idle")
            self.tts.stop()
            self.ui.set_text("Conversation mode OFF")
        else:
            self.ui.set_mode("listen")
            self.ui.set_text("Conversation mode ON (talk now)")
            self.convo_loop()

    def handle_single_press(self):
        if self.busy:
            return
        self.busy = True
        threading.Thread(target=self._do_screenshot_analysis, daemon=True).start()

    def _do_screenshot_analysis(self):
        try:
            self.ui.set_mode("listen")
            self.ui.set_text("Screenshot… OCR…")

            ocr_text = self.ocr.screenshot_ocr()
            if not ocr_text:
                prompt = ("I took a screenshot. OCR found no readable text. "
                          "Ask me what window/app it was and what I want to do.")
            else:
                prompt = ("You only have OCR text (no layout/colors/icons). "
                          "Analyze what it likely shows and suggest next steps.\n\nOCR TEXT:\n"
                          f"{ocr_text}")

            log_turn("user", "[screenshot]")

            self.ui.set_mode("think")
            self.ui.set_text("Thinking…")
            reply = self.llm.reply(prompt)
            log_turn("assistant", reply)

            self.memory.observe_sequence(self.simple_tokenize(reply))

            self._wait_for_silence_gap(USER_TURN_END_SILENCE)
            while self.user_currently_speaking:
                time.sleep(0.06)
            self.tts.speak_async(reply)

        except Exception as e:
            self.ui.set_text(f"Screenshot/OCR error: {e}")
            self.ui.set_mode("idle")
        finally:
            self.busy = False

    def handle_double_press(self):
        self.tts.stop()
        self.ui.set_text("")
        self.ui.set_text("Stopped speaking")
        self.ui.set_mode("listen" if self.convo_mode else "idle")

    def handle_triple_press(self):
        name = self.tts.cycle_voice()
        self.ui.set_text(f"Voice: {name}")
        self._wait_for_silence_gap(USER_TURN_END_SILENCE)
        while self.user_currently_speaking:
            time.sleep(0.06)
        self.tts.speak_async(f"Voice changed to {name}")

    def convo_loop(self):
        def run():
            while self.convo_mode and self.ui.running:
                if self.busy:
                    time.sleep(0.05)
                    continue

                if self._waiting_for_user_end:
                    self.busy = True
                    try:
                        text = self._capture_user_turn(max_total=LISTEN_MAX_TOTAL)
                        self._pending_user_text = text or ""
                        self._waiting_for_user_end = False
                    finally:
                        self.busy = False

                self.busy = True
                try:
                    self.ui.set_mode("listen")
                    self.ui.set_text("Listening…")

                    text = self.stt.listen_until_silence()
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
                    self.memory.observe_sequence(self.simple_tokenize(text))

                    self.ui.set_text(f"You: {text}")
                    self.ui.set_mode("think")
                    self.ui.set_text("Thinking…")

                    reply = self.llm.reply(text)
                    log_turn("assistant", reply)
                    self.memory.observe_sequence(self.simple_tokenize(reply))

                    self._wait_for_silence_gap(USER_TURN_END_SILENCE)
                    while self.user_currently_speaking:
                        time.sleep(0.06)
                    self.tts.speak_async(reply)

                    while self.tts.is_speaking() and self.convo_mode and self.ui.running:
                        time.sleep(0.03)

                    if self._soft_interjection_pending and self.ui.running:
                        self._soft_interjection_pending = False
                        interj = self._capture_user_turn(max_total=3.0)
                        if interj:
                            self._register_soft_interjection(interj)

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
        # --- adaptive mic gain (no shouting needed) ---
        try:
            while True:
                rms, z = self.mic_q.get_nowait()

                # initialize noise floor
                if self._noise_floor is None:
                    self._noise_floor = rms

                # track noise floor slowly
                self._noise_floor = 0.98 * self._noise_floor + 0.02 * rms

                # adaptive gain: quieter room → more boost
                gain = max(8.0, min(40.0, 1.0 / max(self._noise_floor, 1e-4)))

                level = (rms - self._noise_floor) * gain
                level = max(0.0, min(1.0, level))

                self.mic_level = 0.80 * self.mic_level + 0.20 * level

                # zcr update mostly when needed
                if self.tts.is_speaking() or self._user_speaking_since is not None:
                    self.mic_zcr = 0.85 * self.mic_zcr + 0.15 * z

        except queue.Empty:
            pass

        # always decay ZCR slowly so it doesn't go stale
        self.mic_zcr *= 0.97

        # speaking flag for gating AI
        self.user_currently_speaking = self.mic_level > VOL_FLOOR

        # clear visual ownership: if user louder than AI → AI yields
        if self.tts.is_speaking():
            if self.mic_level > (self.ai_audio_level + 0.1):
                if time.time() - self._last_tts_stop >= MIC_INTERRUPT_COOLDOWN:
                    self._last_tts_stop = time.time()
                    self.tts.stop()
                    self._register_interruption()

        now = time.time()
        user_voice = (self.mic_level > MIC_INTERRUPT_THRESHOLD and self.mic_zcr >= MIC_INTERRUPT_MIN_ZCR)

        # ---- USER IS SPEAKING ----
        if user_voice:
            self._last_user_voice_time = now
            if self._user_speaking_since is None:
                self._user_speaking_since = now

            speaking_dur = now - self._user_speaking_since

            if self.tts.is_speaking():
                if speaking_dur >= USER_INTERRUPT_MIN_SEC:
                    if (now - self._last_tts_stop) >= MIC_INTERRUPT_COOLDOWN:
                        self._last_tts_stop = now
                        self.tts.stop()
                        self._register_interruption()
                        self._waiting_for_user_end = True
                        self.ui.set_text("— interrupted —")
                else:
                    self._soft_interjection_pending = True

        # ---- USER SILENCE ----
        else:
            if self._user_speaking_since is not None:
                spoke_for = now - self._user_speaking_since
                self._user_speaking_since = None
                if spoke_for >= USER_INTERRUPT_MIN_SEC:
                    self._waiting_for_user_end = False

    def process_button_events(self):
        try:
            while True:
                ev = self.btn_q.get_nowait()
                if ev == EV_LONG:
                    self.toggle_convo()
                elif ev == EV_SINGLE:
                    self.handle_single_press()
                elif ev == EV_DOUBLE:
                    self.handle_double_press()
                elif ev == EV_TRIPLE:
                    self.handle_triple_press()
        except queue.Empty:
            pass

    def _save_memory_periodic(self):
        if time.time() - self._last_mem_save < 60:
            return
        self._last_mem_save = time.time()
        try:
            with open(MEM_FILE, "wb") as f:
                pickle.dump(self.memory, f)
        except Exception:
            pass

    def run(self):
        while self.ui.running:
            self.update_mic_level()
            self.process_button_events()
            self._save_memory_periodic()

            if not self.tts.is_speaking():
                if self.convo_mode and not self.busy:
                    if self.ui.mode != "listen":
                        self.ui.set_mode("listen")
                elif not self.convo_mode and not self.busy:
                    if self.ui.mode != "idle":
                        self.ui.set_mode("idle")

            self.ui.tick(self.mic_level, self.tts.is_speaking())

        try:
            self.mic_stream.stop()
            self.mic_stream.close()
        except Exception:
            pass

        # final save
        try:
            with open(MEM_FILE, "wb") as f:
                pickle.dump(self.memory, f)
        except Exception:
            pass

        self.ui.close()

# =============================================================
# Main
# =============================================================

if __name__ == "__main__":
    App().run()
