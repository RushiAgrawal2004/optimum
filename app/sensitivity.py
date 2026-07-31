from dataclasses import dataclass
from pathlib import Path
import speed, runner
from config import PROBE_BENCH, TENSOR_GROUPS

@dataclass
class GroupValue:
    name: str
    size_mb: float
    gain_tps: float
    value_per_mb: float

def measure_groups(model: Path, info) -> dict:
    all_groups = list(info.groups.keys())

    # floor: everything in RAM
    floor_ot = runner.build_ot_pattern(all_groups)
    floor = speed.measure(model, ngl=99, ot=floor_ot, profile=PROBE_BENCH)
    if floor is None:
        raise RuntimeError("floor measurement failed")
    print(f"floor (all CPU): {floor.gen_tps:.1f} tok/s\n")

    values = {}
    for g in all_groups:
        others = [x for x in all_groups if x != g]
        ot = runner.build_ot_pattern(others)      # only g on GPU
        r = speed.measure(model, ngl=99, ot=ot, profile=PROBE_BENCH)
        if r is None:
            continue
        gain = r.gen_tps - floor.gen_tps
        size = info.groups[g].size_mb
        values[g] = GroupValue(g, size, gain, gain / size if size else 0)
        print(f"  {g:14s} {size:7.0f} MB  +{gain:5.2f} tok/s  "
              f"value {gain/size*1000:6.3f} per GB")

    return values

if __name__ == "__main__":
    import model_info
    from config import MODEL_DIR
    m = MODEL_DIR / "qwen1.5b-q4.gguf"
    info = model_info.inspect(m)
    speed.warmup(m)
    vals = measure_groups(m, info)
    print("\nBest value first:")
    for v in sorted(vals.values(), key=lambda x: -x.value_per_mb):
        print(f"  {v.name}")