"""
main.py — memorix
  python main.py
"""

import os, sys, time, random, threading, json, math

if getattr(sys, "frozen", False):
    _HERE = os.path.dirname(sys.executable)
else:
    _HERE = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, _HERE)

for s in (sys.stdout, sys.stderr):
    if s and hasattr(s, "reconfigure"):
        try: s.reconfigure(encoding="utf-8", errors="replace")
        except: pass

import tkinter as tk

BG    = "#0f0f0f"
ACC   = "#00e5ff"
WARN  = "#ff9500"
DIM   = "#334455"
BTN   = "#1a2a3a"
BTN_H = "#203040"


def _make_params(h, m, s, ranges):
    from core.frequencies import TimeParams, _lerp
    sf        = s / 59.0 if s else 0.0
    min_hand  = m / 59.0 if m else 0.0
    hour_hand = (h % 12) / 12.0
    return TimeParams(
        second_hz = _lerp(ranges["sec_min"], ranges["sec_max"], sf),
        amplitude = ranges["amp_min"] * (ranges["amp_max"] / ranges["amp_min"]) ** min_hand,
        pan_angle = hour_hand * 2.0 * math.pi,
        hour=h, minute=m, second=s,
        label=f"{h:02d}:{m:02d}:{s:02d}",
    )


class App(tk.Tk):
    def __init__(self, engine):
        super().__init__()
        self.engine       = engine
        self._testing     = False
        self._test_answer = None
        self._fullscreen  = False

        self.title("memorix")
        self.configure(bg=BG)
        self.geometry("640x360")
        self.minsize(420, 280)
        self._build_ui()

        self.bind("<F11>", self._toggle_fullscreen)
        self.bind("<Escape>", lambda e: self._exit_fullscreen())
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._tick()

    # ── layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        bar = tk.Frame(self, bg=BG)
        bar.pack(fill="x")

        tk.Label(bar, text="memorix", bg=BG, fg=DIM,
                 font=("Courier", 10)).pack(side="left", padx=14, pady=6)

        tk.Button(bar, text="✕", bg=BG, fg="#554444", bd=0,
                  font=("Courier", 15, "bold"),
                  activebackground="#200000", activeforeground="#ff5555",
                  cursor="hand2", relief="flat",
                  command=self._on_close).pack(side="right", padx=10, pady=2)

        tk.Button(bar, text="⛶", bg=BG, fg=DIM, bd=0, font=("Courier", 12),
                  activebackground=BTN_H, activeforeground=ACC,
                  cursor="hand2", relief="flat",
                  command=self._toggle_fullscreen).pack(side="right", padx=4, pady=2)

        tk.Frame(self, bg="#1a2a3a", height=1).pack(fill="x")

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)

        self._time_var = tk.StringVar(value="--:--:--")
        tk.Label(body, textvariable=self._time_var, bg=BG, fg=ACC,
                 font=("Courier", 54, "bold")).pack(pady=(22, 2))

        self._day_var = tk.StringVar(value="")
        tk.Label(body, textvariable=self._day_var, bg=BG, fg=DIM,
                 font=("Courier", 11)).pack()

        self._status_var = tk.StringVar(value="press TEST to start a round")
        tk.Label(body, textvariable=self._status_var, bg=BG, fg=WARN,
                 font=("Courier", 11), wraplength=560,
                 justify="center").pack(pady=10)

        self._test_btn = tk.Button(
            body, text="TEST", bg=BTN, fg=ACC,
            font=("Courier", 16, "bold"), bd=0, padx=36, pady=12,
            activebackground=BTN_H, activeforeground=ACC,
            cursor="hand2", relief="flat",
            command=self._start_test,
        )
        self._test_btn.pack(pady=4)

        # guess row — hidden until the listening phase ends
        self._guess_frame = tk.Frame(body, bg=BG)
        self._gh = tk.StringVar(value="00")
        self._gm = tk.StringVar(value="00")
        self._gs = tk.StringVar(value="00")
        self._entries = []
        for var, lbl in [(self._gh, "HH"), (self._gm, "MM"), (self._gs, "SS")]:
            col = tk.Frame(self._guess_frame, bg=BG)
            col.pack(side="left", padx=8)
            tk.Label(col, text=lbl, bg=BG, fg=DIM, font=("Courier", 8)).pack()
            e = tk.Entry(col, textvariable=var, bg=BTN, fg=ACC,
                         font=("Courier", 24, "bold"), width=3,
                         insertbackground=ACC, bd=0, justify="center",
                         highlightthickness=1, highlightbackground=DIM,
                         highlightcolor=ACC)
            e.pack()
            self._entries.append(e)
        tk.Button(self._guess_frame, text="SUBMIT", bg=BTN, fg=ACC,
                  font=("Courier", 13, "bold"), bd=0, padx=18, pady=10,
                  activebackground=BTN_H, activeforeground=ACC,
                  cursor="hand2", relief="flat",
                  command=self._submit_guess).pack(side="left", padx=16)

    # ── live clock ────────────────────────────────────────────────────────────

    def _tick(self):
        from datetime import datetime
        now = datetime.now()
        self._time_var.set(now.strftime("%H:%M:%S"))
        self._day_var.set(now.strftime("%A").upper())
        self.after(500, self._tick)

    # ── window controls ───────────────────────────────────────────────────────

    def _toggle_fullscreen(self, _=None):
        self._fullscreen = not self._fullscreen
        self.attributes("-fullscreen", self._fullscreen)

    def _exit_fullscreen(self):
        if self._fullscreen:
            self._fullscreen = False
            self.attributes("-fullscreen", False)

    def _on_close(self):
        self.engine.stop()
        self.destroy()

    # ── test flow ─────────────────────────────────────────────────────────────

    def _start_test(self):
        if self._testing:
            return
        self._testing = True
        self._test_btn.config(state="disabled", fg=DIM)
        self._guess_frame.pack_forget()
        threading.Thread(target=self._run_test, daemon=True, name="memorix-test").start()

    def _run_test(self):
        h = random.randint(0, 23)
        m = random.randint(0, 59)
        s = random.randint(0, 59)
        self._test_answer = (h, m, s)

        self.engine.freeze()
        rng = self.engine._ranges or {}
        ch, cm, cs = h, m, s
        p0 = _make_params(ch, cm, cs, rng)
        self.engine._synth._cur        = p0
        self.engine._synth._tgt        = p0
        self.engine._synth._pan_smooth = p0.pan_angle

        for i in range(10, 0, -1):
            self.after(0, lambda i=i: self._status_var.set(f"listening…  {i}s"))
            self.engine._synth.update(_make_params(ch, cm, cs, rng))
            time.sleep(1)
            cs += 1
            if cs >= 60: cs = 0; cm += 1
            if cm >= 60: cm = 0; ch = (ch + 1) % 24

        self.engine.unfreeze()
        self.after(0, self._show_guess)

    def _show_guess(self):
        self._status_var.set("what time did you hear?")
        for e in self._entries:
            e.delete(0, tk.END)
            e.insert(0, "00")
        self._guess_frame.pack(pady=10)
        if self._entries:
            self._entries[0].focus_set()
            self._entries[0].select_range(0, tk.END)

    def _submit_guess(self):
        if not self._test_answer:
            return
        try:
            gh = int(self._gh.get()) % 24
            gm = int(self._gm.get()) % 60
            gs = int(self._gs.get()) % 60
        except ValueError:
            self._status_var.set("enter numbers in all three fields")
            return

        h, m, s   = self._test_answer
        raw_dh    = abs((h % 12) - (gh % 12))
        dh        = min(raw_dh, 12 - raw_dh)
        dm        = abs(m - gm)
        ds        = abs(s - gs)
        eh        = dh / 6.0
        em        = dm / 59.0
        es        = ds / 59.0
        score     = round((1.0 - (eh**2 + em**2 + es**2) / 3.0) * 100, 1)

        self._status_var.set(
            f"answer  {h:02d}:{m:02d}:{s:02d}   guess  {gh:02d}:{gm:02d}:{gs:02d}\n"
            f"off by  {dh}h {dm}m {ds}s   score  {score}/100"
        )
        self._guess_frame.pack_forget()
        self._testing     = False
        self._test_answer = None
        self._test_btn.config(state="normal", fg=ACC)


# ── entry ──────────────────────────────────────────────────────────────────────

def main():
    from output.sound import SoundEngine
    from core.frequencies import _DEFAULT_RANGES

    _save = os.path.join(_HERE, "memorix_session.json")
    try:
        with open(_save) as f:
            _saved = json.load(f).get("ranges", {})
        if all(k in _saved for k in ("sec_min", "sec_max", "amp_min", "amp_max")):
            _ranges = _saved
        else:
            _ranges = dict(_DEFAULT_RANGES)
    except Exception:
        _ranges = dict(_DEFAULT_RANGES)

    engine = SoundEngine()
    engine.set_ranges(_ranges)
    engine.start()
    engine.play_intro()

    app = App(engine)
    app.mainloop()


if __name__ == "__main__":
    main()
