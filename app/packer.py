from dataclasses import dataclass, field

@dataclass
class Candidate:
    gpu_groups: list
    cpu_groups: list
    ot_pattern: str
    ngl: int = 99
    threads: int = 6
    ctk: str = "f16"
    predicted_tps: float = 0.0
    mb_used: float = 0.0
    label: str = ""

def pack(values: dict, budget_mb: float) -> tuple:
    ranked = sorted(values.values(), key=lambda v: -v.value_per_mb)
    gpu, cpu, used, predicted = [], [], 0.0, 0.0
    for v in ranked:
        if used + v.size_mb <= budget_mb:
            gpu.append(v.name)
            used += v.size_mb
            predicted += v.gain_tps
        else:
            cpu.append(v.name)
    return gpu, cpu, used, predicted

def build_candidates(values: dict, budget_mb: float, n: int = 8) -> list:
    import runner
    cands, seen = [], set()

    def add(gpu, cpu, used, pred, label):
        key = tuple(sorted(cpu))
        if key in seen:
            return
        seen.add(key)
        cands.append(Candidate(
            gpu_groups=gpu, cpu_groups=cpu,
            ot_pattern=runner.build_ot_pattern(cpu),
            predicted_tps=pred, mb_used=used, label=label))

    gpu, cpu, used, pred = pack(values, budget_mb)
    add(gpu, cpu, used, pred, "greedy")

    # one step safer
    if len(gpu) > 1:
        smallest = min(gpu, key=lambda g: values[g].value_per_mb)
        g2 = [x for x in gpu if x != smallest]
        add(g2, cpu + [smallest],
            used - values[smallest].size_mb,
            pred - values[smallest].gain_tps, "safer")

    # one step greedier
    if cpu:
        best_cpu = max(cpu, key=lambda g: values[g].value_per_mb)
        add(gpu + [best_cpu], [x for x in cpu if x != best_cpu],
            used + values[best_cpu].size_mb,
            pred + values[best_cpu].gain_tps, "greedier")

    # thread and cache variations on the greedy answer
    base = cands[0]
    for t in [4, 8]:
        c = Candidate(base.gpu_groups, base.cpu_groups, base.ot_pattern,
                      threads=t, predicted_tps=base.predicted_tps,
                      mb_used=base.mb_used, label=f"threads={t}")
        cands.append(c)
    c = Candidate(base.gpu_groups, base.cpu_groups, base.ot_pattern,
                  ctk="q8_0", predicted_tps=base.predicted_tps,
                  mb_used=base.mb_used, label="ctk=q8_0")
    cands.append(c)

    return cands[:n]