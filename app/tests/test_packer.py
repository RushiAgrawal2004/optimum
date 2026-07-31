from packer import pack, build_candidates
from sensitivity import GroupValue

def test_takes_best_value_first():
    vals = {
        "small_good": GroupValue("small_good", 100, 3.0, 0.030),
        "big_bad":    GroupValue("big_bad",   1000, 4.0, 0.004),
    }
    gpu, cpu, used, _ = pack(vals, budget_mb=500)
    assert gpu == ["small_good"]
    assert cpu == ["big_bad"]

def test_nothing_fits():
    vals = {"huge": GroupValue("huge", 5000, 10.0, 0.002)}
    gpu, cpu, _, _ = pack(vals, budget_mb=100)
    assert gpu == [] and cpu == ["huge"]

def test_everything_fits():
    vals = {
        "a": GroupValue("a", 100, 1.0, 0.01),
        "b": GroupValue("b", 100, 2.0, 0.02),
    }
    gpu, cpu, used, predicted = pack(vals, budget_mb=1000)
    assert set(gpu) == {"a", "b"}
    assert cpu == []
    assert used == 200
    assert predicted == 3.0

def test_build_candidates_returns_unique_labelled_options():
    vals = {
        "attn_v":  GroupValue("attn_v", 140, 5.0, 0.0357),
        "ffn_up":  GroupValue("ffn_up", 1150, 2.0, 0.0017),
    }
    cands = build_candidates(vals, budget_mb=200, n=8)
    assert len(cands) > 0
    assert cands[0].label == "greedy"
    labels = [c.label for c in cands]
    assert len(labels) == len(set(labels))
