import hashlib
from pathlib import Path
import runner
from config import PERPLEXITY, CACHE_DIR

def file_hash(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        h.update(f.read(1024 * 1024))       # first MB is enough
    h.update(str(p.stat().st_size).encode())
    return h.hexdigest()[:12]

def ensure_reference(ref_model: Path, calib: Path, ngl: int = 20) -> Path:
    key = f"{file_hash(ref_model)}_{file_hash(calib)}"
    out = CACHE_DIR / f"ref_{key}.kld"
    if out.exists():
        print(f"reference cached: {out.name}")
        return out

    print("building reference (one time, a few minutes)...")
    res = runner.run(PERPLEXITY, [
        "-m", ref_model, "-f", calib,
        "--kl-divergence-base", out,
        "-ngl", ngl, "-c", 512,
    ], timeout=3600)

    if not res.ok or not out.exists():
        raise RuntimeError(f"reference failed: {res.reason}\n{res.stderr[-800:]}")
    return out

def prepare_calibration(source: Path, out: Path, kb: int = 50) -> Path:
    text = source.read_text(encoding="utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text[:kb * 1000], encoding="utf-8")
    return out
