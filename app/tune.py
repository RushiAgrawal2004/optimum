from pathlib import Path
import db, hardware, model_info, sensitivity, packer, evaluate, frontier, reference, report, speed

def tune(model: Path, ref_model: Path, calib: Path, min_quality: float = 0.97):
    hw = hardware.probe()
    print(f"{hw.name}  {hw.free_mb:.0f} MB free")

    info = model_info.inspect(model)
    print(f"{info.arch}  {info.n_layers} layers  {info.file_size_mb:.0f} MB\n")

    model_hash = reference.file_hash(model)
    calib_hash = reference.file_hash(calib)

    con = db.connect()
    db.save_model(con, model_hash, info)

    ref = reference.ensure_reference(ref_model, calib)

    speed.warmup(model)

    cached = db.load_sensitivity(con, model_hash, hw.name)
    if cached:
        print("sensitivity: loaded from db\n")
        values = {g: sensitivity.GroupValue(r["group_name"], r["size_mb"],
                                            r["gain_tps"], r["value_per_mb"])
                  for g, r in cached.items()}
    else:
        print("measuring what each part is worth...")
        values = sensitivity.measure_groups(model, info)
        db.save_sensitivity(con, model_hash, hw.name, values)

    budget = hardware.usable_vram_mb(hw)
    cands = packer.build_candidates(values, budget, n=8)
    print(f"\ntesting {len(cands)} candidates...\n")

    evals = evaluate.evaluate_all(model, cands, ref, calib)
    for cand, ev in zip((e.candidate for e in evals), evals):
        db.save_evaluation(con, model_hash, hw, cand, ev, calib_hash)

    front = frontier.frontier(evals)
    best = frontier.pick(front, min_quality)

    print("\n" + report.render_text(model, hw, front, best))

    export_path = Path("cache") / "last_result.json"
    report.export_json(model, hw, front, best, export_path)
    print(f"\nfull results written to {export_path}")

    return best
