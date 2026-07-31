from pathlib import Path

LLAMA_DIR  = Path(r"C:\nativetune\llama")
BENCH      = LLAMA_DIR / "llama-bench.exe"
PERPLEXITY = LLAMA_DIR / "llama-perplexity.exe"
CLI        = LLAMA_DIR / "llama-cli.exe"

MODEL_DIR = Path(r"C:\nativetune\models")
CACHE_DIR = Path(r"C:\nativetune\cache")
DB_PATH   = Path(r"C:\nativetune\results.db")

CACHE_DIR.mkdir(parents=True, exist_ok=True)

VRAM_SAFETY_MB = 250

FAST_BENCH  = {"p": 128, "n": 32, "r": 1}
PROBE_BENCH = {"p": 32,  "n": 16, "r": 1}

TENSOR_GROUPS = ["attn_q", "attn_k", "attn_v", "attn_output",
                 "ffn_gate", "ffn_up", "ffn_down"]