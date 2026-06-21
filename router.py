"""
router.py — Routes a natural-language query to a FreqChord.

Priority:
  1. Ollama (Qwen 2.5 or any local model) if reachable
  2. Keyword fallback (always works offline)

The LLM response is a JSON object that tells us:
  - category   : "time" | "crypto" | "weather" | "news" | "silence"
  - symbol     : (crypto only) e.g. "bitcoin"
  - location   : (weather only) e.g. "Paris"
  - detail     : (time only) 3 / 4 / 5 — how many slots to use
  - confidence : 0–1  float
"""

from __future__ import annotations

import json
import re
from typing import Tuple
from urllib.request import urlopen, Request
from urllib.error   import URLError

from core.frequencies import (
    FreqChord, map_time, map_crypto, map_weather, map_news,
    apply_confidence, silence,
)
from core.apis import (
    fetch_time, fetch_weather, fetch_crypto, fetch_headlines,
)

# ── Ollama config ─────────────────────────────────────────────────────────────

_OLLAMA_URL   = "http://localhost:11434/api/generate"
_MODEL        = "qwen2.5:7b"          # any qwen / mistral / llama3 works
_TIMEOUT      = 10

_SYSTEM = """\
You are a routing agent for a sound synthesizer.
Given a user query, decide what real-world data best answers it and output ONLY
valid JSON with these fields:
  category   : one of "time", "crypto", "weather", "news", "silence"
  symbol     : (crypto only) "bitcoin" | "ethereum" | "solana"
  location   : (weather only) city name string
  detail     : (time only) integer 3, 4, or 5
               3 = hour/min/sec  4 = +day  5 = +weekday
               Use 5 for questions about day/week/month.
               Use 4 for questions about what day it is.
               Use 3 for questions about the current time.
  confidence : float 0.0–1.0  (how confident you are this is the right answer)

Output only the JSON object, nothing else.
"""


# ── public entry point ────────────────────────────────────────────────────────

def route(query: str) -> Tuple[FreqChord, str]:
    """
    Returns (FreqChord, description_string).
    Always succeeds — falls back to keyword routing if LLM is unavailable.
    """
    params = _llm_route(query) or _keyword_route(query)
    chord, label = _build_chord(params)
    confidence   = float(params.get("confidence", 0.8))
    chord        = apply_confidence(chord, confidence)
    return chord, label


# ── LLM routing ───────────────────────────────────────────────────────────────

def _llm_route(query: str) -> dict | None:
    payload = json.dumps({
        "model":  _MODEL,
        "system": _SYSTEM,
        "prompt": query,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 120},
    }).encode()
    try:
        req = Request(_OLLAMA_URL, data=payload,
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=_TIMEOUT) as r:
            body = json.loads(r.read())
        text = body.get("response", "")
        return _parse_json(text)
    except (URLError, OSError):
        return None   # Ollama not running — fall through to keyword router
    except Exception:
        return None


def _parse_json(text: str) -> dict | None:
    try:
        # strip markdown fences if present
        text = re.sub(r"```[a-z]*\n?", "", text).strip()
        return json.loads(text)
    except Exception:
        # try extracting first {...} block
        m = re.search(r"\{[^}]+\}", text, re.S)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
        return None


# ── keyword fallback ──────────────────────────────────────────────────────────

_KW_CRYPTO  = {"bitcoin","btc","ethereum","eth","solana","sol","crypto","coin","price"}
_KW_WEATHER = {"weather","temperature","rain","wind","humidity","forecast","hot","cold"}
_KW_NEWS    = {"news","headline","hacker","hn","article","story","tech"}
_KW_TIME    = {"time","clock","hour","minute","second","day","week","month","year"}
_KW_WEEK    = {"week","weekday","day of","monday","tuesday","wednesday","thursday","friday"}
_KW_MONTH   = {"month","year","season","quarter"}

def _keyword_route(query: str) -> dict:
    q = query.lower()
    words = set(re.findall(r"[a-z]+", q))

    if words & _KW_CRYPTO:
        sym = "bitcoin"
        for s in ("ethereum","eth"):
            if s in words: sym = "ethereum"
        for s in ("solana","sol"):
            if s in words: sym = "solana"
        return {"category": "crypto", "symbol": sym, "confidence": 0.75}

    if words & _KW_WEATHER:
        loc = "Paris"
        # crude: capitalised word after "in" or "for"
        m = re.search(r"\b(?:in|for)\s+([A-Z][a-zA-Z]+)", query)
        if m:
            loc = m.group(1)
        return {"category": "weather", "location": loc, "confidence": 0.75}

    if words & _KW_NEWS:
        return {"category": "news", "confidence": 0.7}

    if words & _KW_TIME:
        detail = 3
        if words & _KW_MONTH:  detail = 5
        elif words & _KW_WEEK: detail = 5
        elif "day" in words:   detail = 4
        return {"category": "time", "detail": detail, "confidence": 0.85}

    # default: time
    return {"category": "time", "detail": 3, "confidence": 0.6}


# ── chord builder ─────────────────────────────────────────────────────────────

def _build_chord(params: dict) -> Tuple[FreqChord, str]:
    cat = params.get("category", "time")

    if cat == "crypto":
        sym  = params.get("symbol", "bitcoin")
        data = fetch_crypto(sym)
        if data:
            chord = map_crypto(data, symbol=sym)
            p     = data.get(sym, {}).get("usd", 0)
            return chord, f"{sym}  ${p:,.0f}"
        return silence(), f"{sym} unavailable"

    if cat == "weather":
        loc  = params.get("location", "Paris")
        data = fetch_weather(loc)
        if data:
            chord = map_weather(data)
            return chord, f"weather {loc}  {data.get('temp_c',0)}°C"
        return silence(), f"weather {loc} unavailable"

    if cat == "news":
        data = fetch_headlines(5)
        if data:
            chord = map_news(data)
            return chord, "tech headlines"
        return silence(), "news unavailable"

    if cat == "silence":
        return silence(), "silence"

    # time (default)
    detail = int(params.get("detail", 3))
    chord  = map_time(fetch_time(), detail=detail)
    labels = {3: "time hh:mm:ss", 4: "time +day", 5: "time +weekday"}
    return chord, labels.get(detail, "time")
