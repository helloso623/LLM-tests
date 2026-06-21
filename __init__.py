"""
memorix — semantic node-graph memory system.

Quick start (from inside memorix/ or with memorix/ on sys.path):
    from core import Memory
    mem = Memory()
    mem.ingest_apis()
    print(mem.summary("bitcoin"))
    results = mem.recall("weather temperature")

Or from outside the directory:
    import sys; sys.path.insert(0, "/path/to/memorix")
    from core import Memory
"""
