def is_dominated(a, b) -> bool:
    """b beats a on both goals."""
    return b.gen_tps >= a.gen_tps and b.quality >= a.quality and \
           (b.gen_tps > a.gen_tps or b.quality > a.quality)

def frontier(evals: list) -> list:
    return [a for a in evals if not any(is_dominated(a, b) for b in evals if b is not a)]

def pick(front: list, min_quality: float):
    good = [e for e in front if e.quality >= min_quality]
    if not good:
        return None
    return max(good, key=lambda e: e.gen_tps)