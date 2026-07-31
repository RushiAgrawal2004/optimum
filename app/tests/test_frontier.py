from types import SimpleNamespace as S
from frontier import frontier, pick

def make(tps, q): return S(gen_tps=tps, quality=q)

def test_removes_losers():
    a, b, c = make(60, 0.90), make(40, 0.85), make(45, 0.98)
    assert b not in frontier([a, b, c])
    assert a in frontier([a, b, c])

def test_respects_floor():
    front = [make(60, 0.90), make(45, 0.98)]
    assert pick(front, 0.95).gen_tps == 45
    assert pick(front, 0.99) is None

def test_equal_candidates_both_survive():
    a, b = make(50, 0.9), make(50, 0.9)
    front = frontier([a, b])
    assert a in front and b in front

def test_pick_returns_fastest_above_floor():
    front = [make(30, 0.99), make(50, 0.96), make(45, 0.97)]
    assert pick(front, 0.95).gen_tps == 50
