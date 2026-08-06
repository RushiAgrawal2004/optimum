import os
from pathlib import Path


def _env_path(var: str, default: Path) -> Path:
    v = os.environ.get(var)
    return Path(v) if v else default


# Every path below can be overridden with an environment variable, so a
# pip-installed copy on someone else's machine isn't stuck with one
# developer's folder layout. Leaving every variable unset reproduces the
# original hardcoded layout exactly (C:\nativetune\...) — that's what this
# repo's own checkout has always pointed at, so an existing install with no
# env vars set keeps working unchanged. scripts/install.ps1 sets
# NATIVETUNE_HOME for fresh installs instead of relying on this fallback.
ROOT_DIR = _env_path("NATIVETUNE_HOME", Path(r"C:\nativetune"))

LLAMA_DIR  = _env_path("NATIVETUNE_LLAMA_DIR", ROOT_DIR / "llama")
BENCH      = LLAMA_DIR / "llama-bench.exe"
PERPLEXITY = LLAMA_DIR / "llama-perplexity.exe"
CLI        = LLAMA_DIR / "llama-cli.exe"

MODEL_DIR  = _env_path("NATIVETUNE_MODEL_DIR", ROOT_DIR / "models")
CACHE_DIR  = _env_path("NATIVETUNE_CACHE_DIR", ROOT_DIR / "cache")
DB_PATH    = _env_path("NATIVETUNE_DB_PATH", ROOT_DIR / "results.db")
CALIB_FILE = _env_path("NATIVETUNE_CALIB_FILE", ROOT_DIR / "data" / "calib-50kb.txt")

CACHE_DIR.mkdir(parents=True, exist_ok=True)

VRAM_SAFETY_MB = 250

FAST_BENCH  = {"p": 128, "n": 32, "r": 1}
PROBE_BENCH = {"p": 32,  "n": 16, "r": 1}

TENSOR_GROUPS = ["attn_q", "attn_k", "attn_v", "attn_output",
                 "ffn_gate", "ffn_up", "ffn_down"]