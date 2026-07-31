import random
from dataclasses import dataclass
from pathlib import Path
import speed, quality

@dataclass
class Evaluation:
    candidate: object
    gen_tps: float
    quality: float
    kl_mean: float
    vram_mb: float
    spilled: bool

def evaluate(model: Path, cand, ref_kld: Path, calib: Path) -> Evaluation | None:
    s = speed.measure(model, cand.ngl, cand.ot_pattern, cand.threads, cand.ctk)
    if s is None:
        return None
    q = quality.measure_kl(model, ref_kld, calib, cand.ngl, cand.ot_pattern, cand.ctk)
    if q is None:
        return None
    return Evaluation(cand, s.gen_tps, quality.quality_score(q),
                      q.kl_mean, s.vram_mb, s.spilled)

def evaluate_all(model, candidates, ref_kld, calib, shuffle=True) -> list:
    order = list(candidates)
    if shuffle:
        random.shuffle(order)          # the GPU warms up. always shuffle.
    results = []
    for i, c in enumerate(order, 1):
        print(f"[{i}/{len(order)}] {c.label or 'candidate'} ... ", end="", flush=True)
        ev = evaluate(model, c, ref_kld, calib)
        if ev:
            print(f"{ev.gen_tps:.1f} tok/s, quality {ev.quality:.4f}")
            results.append(ev)
        else:
            print("failed")
    return results