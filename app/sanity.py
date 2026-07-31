"""Run before any real study. Checks whether measurements are trustworthy,
not whether the code is correct — pytest covers that."""
from pathlib import Path
import speed, quality


def check_repeatable(model: Path, ngl: int, rounds: int = 3, max_spread: float = 0.05) -> bool:
    runs = [r.gen_tps for r in (speed.measure(model, ngl) for _ in range(rounds)) if r]
    if len(runs) < rounds:
        print("  FAIL: one or more runs errored out")
        return False
    spread = (max(runs) - min(runs)) / min(runs)
    print(f"  speeds: {runs}  spread: {spread*100:.1f}%")
    ok = spread < max_spread
    print("  PASS" if ok else "  FAIL: too noisy, fix warmup before continuing")
    return ok


def check_direction(model: Path, layers=(0, 8, 16, 24)) -> bool:
    speeds = []
    for n in layers:
        r = speed.measure(model, ngl=n)
        speeds.append(r.gen_tps if r else None)
    print(f"  ngl {list(layers)} -> {speeds}")
    if None in speeds:
        print("  FAIL: a run errored out")
        return False
    ok = speeds[0] < speeds[-1]
    print("  PASS" if ok else "  FAIL: more GPU layers made it slower, CUDA may not be in use")
    return ok


def check_quality_ordering(model_low: Path, model_high: Path, ref_kld: Path, calib: Path, ngl: int) -> bool:
    q_low = quality.measure_kl(model_low, ref_kld, calib, ngl)
    q_high = quality.measure_kl(model_high, ref_kld, calib, ngl)
    if not q_low or not q_high:
        print("  FAIL: a measurement errored out")
        return False
    s_low, s_high = quality.quality_score(q_low), quality.quality_score(q_high)
    print(f"  low-precision: {s_low:.4f}   high-precision: {s_high:.4f}")
    ok = s_high > s_low
    print("  PASS" if ok else "  FAIL: heavier squeezing scored better, scorer is broken")
    return ok


def check_scorer_detects_broken(ruined_model: Path, ref_kld: Path, calib: Path, ngl: int, max_score: float = 0.15) -> bool:
    q = quality.measure_kl(ruined_model, ref_kld, calib, ngl)
    if not q:
        print("  FAIL: measurement errored out")
        return False
    s = quality.quality_score(q)
    print(f"  ruined model scored: {s:.4f}")
    ok = s < max_score
    print("  PASS" if ok else "  FAIL: a ruined model scored well, scorer is not measuring anything")
    return ok


def check_ot_pattern(model: Path, all_cpu_pattern: str) -> bool:
    baseline = speed.measure(model, ngl=99)
    dropped = speed.measure(model, ngl=99, ot=all_cpu_pattern)
    if not baseline or not dropped:
        print("  FAIL: a run errored out")
        return False
    print(f"  ngl=99: {baseline.gen_tps:.1f} tok/s   ngl=99+all-CPU: {dropped.gen_tps:.1f} tok/s")
    ok = dropped.gen_tps < baseline.gen_tps * 0.8
    print("  PASS" if ok else "  FAIL: pattern is not matching, sensitivity numbers are fiction")
    return ok


if __name__ == "__main__":
    import runner
    from config import MODEL_DIR

    m = MODEL_DIR / "qwen1.5b-q4.gguf"
    speed.warmup(m)

    print("[1] repeat check")
    check_repeatable(m, ngl=20)

    print("\n[2] direction check")
    check_direction(m)

    print("\n[3] -ot pattern check")
    check_ot_pattern(m, runner.build_ot_pattern(
        ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"]))

    print("\nrun quality/order/ruined-model checks manually once you have a "
          "reference .kld and a deliberately ruined model file.")
