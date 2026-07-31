import json
from pathlib import Path
from dataclasses import dataclass
import runner, hardware
from config import BENCH, FAST_BENCH

@dataclass
class SpeedResult:
    gen_tps: float
    prompt_tps: float
    vram_used_mb: float
    spilled: bool

def parse_bench_json(raw: str) -> dict | None:
    """Pure function. Testable with no GPU."""
    try:
        rows = json.loads(raw)
    except Exception:
        return None
    out = {"gen_tps": None, "prompt_tps": None}
    for r in rows:
        if r.get("n_gen", 0) > 0:
            out["gen_tps"] = r.get("avg_ts")
        elif r.get("n_prompt", 0) > 0:
            out["prompt_tps"] = r.get("avg_ts")
    return out if out["gen_tps"] else None

def measure(model: Path, ngl: int, ot: str = None, threads: int = 6,
            ctk: str = "f16", profile: dict = None) -> SpeedResult | None:
    profile = profile or FAST_BENCH
    args = ["-m", model, "-ngl", ngl, "-t", threads,
            "-ctk", ctk, "-ctv", ctk,
            "-p", profile["p"], "-n", profile["n"], "-r", profile["r"],
            "-o", "json"]
    if ot:
        args += ["-ot", ot]

    before = hardware.free_vram_mb()
    res = runner.run(BENCH, args, timeout=600)
    if not res.ok:
        return None
    parsed = parse_bench_json(res.stdout)
    if not parsed:
        return None
    # llama-bench has already exited and freed its VRAM by the time we read
    # "after", so vram_used_mb reads ~0 here. To get a real peak, VRAM must be
    # polled from a background thread while the subprocess is still running.
    after = hardware.free_vram_mb()

    return SpeedResult(
        gen_tps=parsed["gen_tps"],
        prompt_tps=parsed["prompt_tps"] or 0,
        vram_used_mb=before - after,
        spilled=(after < 150),
    )

def measure_default(model: Path, profile: dict = None) -> SpeedResult | None:
    """Runs llama-bench with none of -ngl/-t/-ctk/-ctv/-ot set — every optimization
    flag left at whatever llama.cpp itself compiles in, not a NativeTune choice."""
    profile = profile or FAST_BENCH
    args = ["-m", model, "-p", profile["p"], "-n", profile["n"], "-r", profile["r"],
            "-o", "json"]

    before = hardware.free_vram_mb()
    res = runner.run(BENCH, args, timeout=600)
    if not res.ok:
        return None
    parsed = parse_bench_json(res.stdout)
    if not parsed:
        return None
    after = hardware.free_vram_mb()

    return SpeedResult(
        gen_tps=parsed["gen_tps"],
        prompt_tps=parsed["prompt_tps"] or 0,
        vram_used_mb=before - after,
        spilled=(after < 150),
    )

def warmup(model: Path, rounds: int = 3):
    from config import PROBE_BENCH
    for _ in range(rounds):
        measure(model, ngl=99, profile=PROBE_BENCH)

if __name__ == "__main__":
    from config import MODEL_DIR
    m = MODEL_DIR / "qwen1.5b-q4.gguf"
    warmup(m)
    for ngl in [0, 10, 20, 28]:
        r = measure(m, ngl)
        print(f"ngl={ngl:3d}  {r.gen_tps if r else 'FAILED'}")