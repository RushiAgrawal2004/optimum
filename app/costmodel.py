"""Cost model: predicts speed and quality for a setting combination that was
never actually run, by looking at how similar settings performed elsewhere
in results.db.

This is distance-weighted k-nearest-neighbors over whatever has actually
been measured — not a trained network. results.db has a few dozen rows,
not thousands; fitting anything fancier than KNN to that little data would
be overfitting to noise, not learning (see nativetune-scope.md section 5,
"Reinforcement learning controller" — same reasoning applies here one size
down). What this buys you: a full 'tune' run costs real GPU minutes per
candidate; this answers in under a millisecond, at the cost of being a
guess whenever nothing in results.db is actually nearby. The confidence
field says which case you're in.

See nativetune-explained.md, Layer 3, for the ambition this is scoped down
from.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# KV-cache type, encoded as a small ordinal so distance between them means
# something (f16 is "no squeeze", q8_0/q4_0 squeeze progressively harder).
CTK_INDEX = {"f16": 0, "q8_0": 1, "q4_0": 2}

# "Settings left at llama.cpp's own compiled-in default" (ngl/threads/ctk
# all None) is not a point on the numeric scale of any of these features —
# it is a different, unmeasured category. Marking it with a value clearly
# outside the real range keeps it from being treated as "ngl=0" or
# "threads=0", which would be a false neighbor.
DEFAULT_MARKER = -1.0

FIELDS = ("size_mb", "n_layers", "ngl_frac", "threads", "ctk_idx")


@dataclass
class TrainingRow:
    size_mb: float
    n_layers: int
    ngl_frac: float   # ngl / n_layers, or DEFAULT_MARKER
    threads: float     # or DEFAULT_MARKER
    ctk_idx: float      # or DEFAULT_MARKER
    gen_tps: float
    quality: float
    label: str = ""


@dataclass
class Prediction:
    gen_tps: float
    quality: float
    confidence: str      # "high" | "medium" | "low"
    nearest_dist: float
    n_neighbors: int
    n_training_rows: int


def _encode_ngl(ngl: int | None, n_layers: int) -> float:
    if ngl is None or not n_layers:
        return DEFAULT_MARKER
    return ngl / n_layers


def _encode_ctk(ctk: str | None) -> float:
    if not ctk or ctk == "default":
        return DEFAULT_MARKER
    return CTK_INDEX.get(ctk, DEFAULT_MARKER)


def build_training_set(runs: list[dict], models: dict[str, dict]) -> list[TrainingRow]:
    """Turn raw db rows into feature vectors. Skips runs with no speed/quality
    measurement yet, and runs whose model was never inspected (no row in the
    models table to pull size/layers from)."""
    rows = []
    for r in runs:
        if r.get("gen_tps") is None or r.get("quality") is None:
            continue
        m = models.get(r["model_hash"])
        if not m:
            continue
        rows.append(TrainingRow(
            size_mb=m["file_size_mb"],
            n_layers=m["n_layers"],
            ngl_frac=_encode_ngl(r["ngl"], m["n_layers"]),
            threads=DEFAULT_MARKER if r["threads"] is None else float(r["threads"]),
            ctk_idx=_encode_ctk(r["ctk"]),
            gen_tps=r["gen_tps"],
            quality=r["quality"],
            label=r.get("label", ""),
        ))
    return rows


def _mean_std(rows: list[TrainingRow], field: str) -> tuple[float, float]:
    vals = [getattr(r, field) for r in rows]
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return mean, math.sqrt(var) or 1.0  # a constant feature can't divide by 0


def _vector(d: dict, stats: dict) -> list[float]:
    return [(d[f] - stats[f][0]) / stats[f][1] for f in FIELDS]


def predict(rows: list[TrainingRow], size_mb: float, n_layers: int,
            ngl: int | None, threads: int | None, ctk: str | None,
            k: int = 3) -> Prediction | None:
    """Distance-weighted k-NN prediction of (gen_tps, quality) for a setting
    combination, using every recorded run (any model, any prior study) as
    the training set. Returns None if nothing has ever been recorded."""
    if not rows:
        return None

    stats = {f: _mean_std(rows, f) for f in FIELDS}
    target = _vector({
        "size_mb": size_mb, "n_layers": n_layers,
        "ngl_frac": _encode_ngl(ngl, n_layers), "threads": DEFAULT_MARKER if threads is None else float(threads),
        "ctk_idx": _encode_ctk(ctk),
    }, stats)

    scored = []
    for row in rows:
        rvec = _vector({f: getattr(row, f) for f in FIELDS}, stats)
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(target, rvec)))
        scored.append((dist, row))
    scored.sort(key=lambda x: x[0])
    neighbors = scored[:min(k, len(scored))]

    nearest = neighbors[0][0]
    if nearest == 0:
        # An exact match exists — use it alone rather than blending in
        # farther neighbors that would just add noise.
        weights = [1.0 if d == 0 else 0.0 for d, _ in neighbors]
    else:
        weights = [1.0 / d for d, _ in neighbors]
    wsum = sum(weights)

    gen_tps = sum(w * r.gen_tps for w, (_, r) in zip(weights, neighbors)) / wsum
    quality = sum(w * r.quality for w, (_, r) in zip(weights, neighbors)) / wsum

    if nearest < 0.5:
        confidence = "high"
    elif nearest < 2.0:
        confidence = "medium"
    else:
        confidence = "low"

    return Prediction(gen_tps=gen_tps, quality=quality, confidence=confidence,
                       nearest_dist=nearest, n_neighbors=len(neighbors),
                       n_training_rows=len(rows))
