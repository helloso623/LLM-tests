# memorix

Semantic node-graph memory system with an always-on frequency sound display.
No API keys needed.  No ML models needed for the memory engine.

---

## What it does

- Builds a live knowledge graph from free public APIs (time, weather, crypto, news)
- Lets you query the graph with natural language and get ranked results
- Plays **a continuous audio buzz** that encodes real-world data as sound frequencies
  - Default: current time as 3 simultaneous tones (hour / minute / second)
  - On query: routes to the right data source and blends in up to 5 tones
  - Auto-reverts to live time after 30 s of silence

---

## Frequency system

Up to 5 additive sine/triangle oscillators play simultaneously.  Each slot
covers a distinct frequency band:

| Slot | Band (Hz)   | Default meaning     | Waveform  |
|------|-------------|---------------------|-----------|
| 0    | 80–160      | Category anchor     | Sine      |
| 1    | 160–320     | Slow dimension      | Sine      |
| 2    | 320–640     | Medium dimension    | Triangle  |
| 3    | 640–1280    | Fast dimension      | Triangle  |
| 4    | 1280–2560   | Confidence accent   | Sine      |

### Time encoding

| Detail | Slots used                          |
|--------|-------------------------------------|
| 3      | 1=hour, 2=minute, 3=second          |
| 4      | 0=day-of-month, 1=hour, 2=min, 3=sec |
| 5      | 0=weekday, 1=day, 2=hour, 3=min, 4=sec |

### Crypto (log-10 scale)

Price is mapped on a log₁₀ scale so $10 / $1 000 / $100 000 are equally
spaced across slot 1.  24h price change goes to slot 2 (center=neutral).

### Confidence / truthness

The LLM (or keyword router) emits a confidence score 0–1:
- Sets slot 4 amplitude proportionally
- Limits the number of active slots (low confidence → fewer active slots)
- Higher confidence = richer, fuller sound

---

## Project structure

```
memorix/
├── main.py           CLI entry point
├── router.py         LLM (Qwen/Ollama) + keyword query router
├── core/
│   ├── __init__.py   Memory high-level class
│   ├── apis.py       Free data fetchers (wttr.in, HackerNews, CoinGecko)
│   ├── embedder.py   Hash-projection embedder (no ML libs needed)
│   ├── frequencies.py  data → FreqChord mapper
│   ├── listener.py     always-on mic + wake-phrase detection
│   ├── ingest.py     dict → semantic graph nodes
│   ├── nodes.py      SemanticGraph, Node, Edge, salience decay
│   ├── recall.py     cosine similarity search
│   ├── store.py      JSON persistence + AutoSaver
│   └── synth.py      Stateful additive synthesizer (numpy)
├── output/
│   ├── __init__.py
│   └── sound.py      SoundEngine — continuous sounddevice stream
└── data/
    └── graph.json    Persistent graph (auto-created)
```

---

## Install

```bash
pip install numpy sounddevice
pip install SpeechRecognition pyaudio   # voice input
# optional — for better embeddings:
pip install sentence-transformers
# optional — for LLM routing:
# install Ollama then: ollama pull qwen2.5:7b
```

> **Windows pyaudio note:** if `pip install pyaudio` fails, use the unofficial wheel:
> ```
> pip install pipwin && pipwin install pyaudio
> ```

---

## Run

### Always-on interactive mode (recommended)

```bash
python main.py listen
```

The engine starts immediately playing time tones.

**Voice** — say the wake phrase then your query:
```
"From A to C"  →  pause  →  "bitcoin price"
"From A to C what's the weather in Tokyo"   ← inline also works
```

**Keyboard** — type and press Enter:
```
> bitcoin price
> weather in Tokyo
> what's the news
> detail 5       ← adds weekday + day-of-month to time display
> quit
```

### One-shot query

```bash
python main.py sound "ethereum price"
python main.py sound "weather in London"
```

Plays the chord, auto-reverts to time after 30 s.

### Memory / graph commands

```bash
python main.py ingest                  # fetch all APIs → graph
python main.py recall "bitcoin price"  # semantic search
python main.py timeline --hours 6      # recent nodes
python main.py stats                   # node/edge counts
python main.py prune                   # remove stale nodes
python main.py loop 60                 # ingest every 60 s
```

---

## LLM routing (Qwen via Ollama)

When Ollama is running the router sends your query to Qwen 2.5 and asks it
to return JSON like:

```json
{ "category": "crypto", "symbol": "bitcoin", "confidence": 0.92 }
{ "category": "weather", "location": "Tokyo", "confidence": 0.88 }
{ "category": "time", "detail": 5, "confidence": 0.95 }
```

The model decides whether to use 3, 4, or 5 frequency slots for time queries
(e.g. "what day is it?" → detail 4, "what week of the year?" → detail 5).

If Ollama is not running the router falls back to keyword matching — no model
required.

---

## Audio notes

- Output is mono, 44 100 Hz, float32
- Frame size 1 024 samples ≈ 23 ms latency
- All 5 oscillators share a single soft-clip ceiling so they can never clip
- Phase accumulators persist across frames — no clicks on chord changes
- Parameter changes (freq, amplitude) ramp linearly across each frame
