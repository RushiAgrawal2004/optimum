import hashlib
import json
from typing import Any, Callable
from config import CACHE_DIR

def key(*parts) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.md5(raw.encode()).hexdigest()[:16]

def _path(k: str):
    return CACHE_DIR / f"{k}.json"

def get(k: str) -> Any | None:
    p = _path(k)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

def put(k: str, value: Any) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _path(k).write_text(json.dumps(value))

def cached(k: str, compute: Callable) -> Any:
    hit = get(k)
    if hit is not None:
        return hit
    value = compute()
    put(k, value)
    return value
