from pathlib import Path
import hardware, model_info, sensitivity, packer, evaluate, frontier, reference, speed

def tune(model: Path, ref_model: Path, calib: Path, min_quality: float = 0.97):
    hw = hardware.probe()
    print(f"{hw.name}  {hw.free_mb:.0f} MB free")

    info = model_info.inspect(model)
    print(f"{info.arch}  {info.n_layers} layers  {info.file_size_mb:.0f} MB\n")

    ref = reference.ensure_reference(ref_model, calib)

    speed.warmup(model)

    print("measuring what each part is worth...")
    values = sensitivity.measure_groups(model, info)

    budget = hardware.usable_vram_mb(hw)
    cands = packer.build_candidates(values, budget, n=8)
    print(f"\ntesting {len(cands)} candidates...\n")

    evals = evaluate.evaluate_all(model, cands, ref, calib)
    front = frontier.frontier(evals)
    best = frontier.pick(front, min_quality)

    print("\n--- good options ---")
    for e in sorted(front, key=lambda x: -x.gen_tps):
        mark = " <-- pick" if e is best else ""
        print(f"  {e.gen_tps:6.1f} tok/s   quality {e.quality:.4f}   "
              f"{e.candidate.label}{mark}")

    if best:
        ot = f' -ot "{best.candidate.ot_pattern}"' if best.candidate.ot_pattern else ""
        print(f"\nllama-server.exe -m {model.name} -ngl {best.candidate.ngl}"
              f"{ot} -t {best.candidate.threads} -ctk {best.candidate.ctk}")
    else:
        print(f"\nNothing reached quality {min_quality}. Try a bigger model file.")

    return best