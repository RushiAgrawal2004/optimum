from costmodel import TrainingRow, build_training_set, predict


def row(**kw):
    base = dict(size_mb=1000, n_layers=28, ngl_frac=99/28, threads=6.0,
                ctk_idx=0, gen_tps=100.0, quality=0.9)
    base.update(kw)
    return TrainingRow(**base)


def test_no_training_data_returns_none():
    assert predict([], 1000, 28, ngl=99, threads=6, ctk="f16") is None


def test_exact_match_returns_recorded_value_with_high_confidence():
    rows = [row(gen_tps=100.0), row(threads=4.0, gen_tps=90.0)]
    pred = predict(rows, 1000, 28, ngl=99, threads=6, ctk="f16", k=2)
    assert abs(pred.gen_tps - 100.0) < 0.01
    assert pred.nearest_dist == 0
    assert pred.confidence == "high"


def test_interpolates_between_two_neighbors():
    rows = [
        row(threads=4.0, gen_tps=80.0),
        row(threads=8.0, gen_tps=100.0),
    ]
    pred = predict(rows, 1000, 28, ngl=99, threads=6, ctk="f16", k=2)
    assert 80.0 < pred.gen_tps < 100.0


def test_closer_neighbor_weighted_more():
    rows = [
        row(threads=6.0, gen_tps=100.0),   # exact match on threads
        row(threads=2.0, gen_tps=0.0),     # far away
    ]
    pred = predict(rows, 1000, 28, ngl=99, threads=6, ctk="f16", k=2)
    assert pred.gen_tps > 50.0  # pulled toward the close neighbor, not the midpoint


def test_far_off_setting_gets_low_confidence():
    rows = [row()]
    # a model and setting combo nothing in the tiny dataset resembles
    pred = predict(rows, 8000, 60, ngl=10, threads=2, ctk="q4_0", k=3)
    assert pred.confidence == "low"


def test_default_settings_dont_get_mistaken_for_ngl_zero():
    default_row = row(ngl_frac=-1.0, threads=-1.0, ctk_idx=-1.0, gen_tps=50.0)
    tuned_row = row(ngl_frac=1.0, threads=6.0, ctk_idx=0, gen_tps=100.0)
    pred = predict([default_row, tuned_row], 1000, 28, ngl=28, threads=6, ctk="f16", k=1)
    assert abs(pred.gen_tps - 100.0) < 0.01


def test_build_training_set_skips_unmeasured_runs():
    runs = [
        {"model_hash": "abc", "gen_tps": None, "quality": None,
         "ngl": 99, "threads": 6, "ctk": "f16", "label": "unfinished"},
        {"model_hash": "abc", "gen_tps": 50.0, "quality": 0.9,
         "ngl": 99, "threads": 6, "ctk": "f16", "label": "done"},
    ]
    models = {"abc": {"file_size_mb": 1000, "n_layers": 28}}
    rows = build_training_set(runs, models)
    assert len(rows) == 1
    assert rows[0].label == "done"


def test_build_training_set_skips_rows_with_no_model_info():
    runs = [{"model_hash": "missing", "gen_tps": 50.0, "quality": 0.9,
             "ngl": 99, "threads": 6, "ctk": "f16", "label": "y"}]
    assert build_training_set(runs, {}) == []


def test_build_training_set_encodes_default_row_distinctly():
    runs = [{"model_hash": "abc", "gen_tps": 50.0, "quality": 0.9,
             "ngl": None, "threads": None, "ctk": "default", "label": "llama.cpp default"}]
    models = {"abc": {"file_size_mb": 1000, "n_layers": 28}}
    rows = build_training_set(runs, models)
    assert rows[0].ngl_frac == -1.0
    assert rows[0].threads == -1.0
    assert rows[0].ctk_idx == -1.0
