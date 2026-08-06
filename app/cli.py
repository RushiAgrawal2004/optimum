import argparse
import sys
from pathlib import Path

import hardware
import model_info
import sensitivity
import speed
import tune as tune_module
from config import CALIB_FILE, MODEL_DIR


def _resolve(name: str) -> Path:
    p = Path(name)
    if not p.exists():
        p = MODEL_DIR / name
    if not p.exists():
        sys.exit(f"model not found: {name}")
    return p


def cmd_probe(_args):
    hw = hardware.probe()
    print(f"gpu       {hw.name}")
    print(f"vram      {hw.free_mb:.0f} MB free / {hw.total_mb:.0f} MB total")
    print(f"usable    {hardware.usable_vram_mb(hw):.0f} MB (safety margin applied)")
    print(f"ram       {hw.ram_free_mb:.0f} MB free / {hw.ram_total_mb:.0f} MB total")


def cmd_inspect(args):
    info = model_info.inspect(_resolve(args.model))
    print(f"{info.arch}  {info.n_layers} layers  {info.file_size_mb:.0f} MB\n")
    for g in sorted(info.groups.values(), key=lambda x: -x.size_mb):
        print(f"  {g.name:14s} {g.size_mb:8.1f} MB  ({g.count} tensors)")


def cmd_sensitivity(args):
    model = _resolve(args.model)
    info = model_info.inspect(model)
    speed.warmup(model)
    values = sensitivity.measure_groups(model, info)
    print("\nBest value first:")
    for v in sorted(values.values(), key=lambda x: -x.value_per_mb):
        print(f"  {v.name:14s} {v.size_mb:8.0f} MB  +{v.gain_tps:5.2f} tok/s")


def cmd_tune(args):
    model = _resolve(args.model)
    ref_model = _resolve(args.ref_model) if args.ref_model else model
    calib = Path(args.calib)
    if not calib.exists():
        sys.exit(f"calibration file not found: {calib}")
    tune_module.tune(model, ref_model, calib, min_quality=args.min_quality)


def cmd_report(args):
    import db
    con = db.connect()
    rows = db.load_all(con, "runs")
    if not rows:
        print("no runs recorded yet")
        return
    for r in rows[-args.limit:]:
        ngl = r["ngl"] if r["ngl"] is not None else "default"
        print(f"{r['ts']}  {r['label']:<18}  {r['gen_tps'] or 0:6.1f} tok/s  "
              f"quality {r['quality'] or 0:.4f}  ngl={ngl}")


def cmd_start(args):
    import webapp
    webapp.serve(port=args.port, open_browser=not args.no_browser)


def cmd_default(args):
    import db
    import quality
    import reference
    from types import SimpleNamespace

    model = _resolve(args.model)
    ref_model = _resolve(args.ref_model) if args.ref_model else model
    calib = Path(args.calib)
    if not calib.exists():
        sys.exit(f"calibration file not found: {calib}")

    hw = hardware.probe()
    info = model_info.inspect(model)
    model_hash = reference.file_hash(model)
    calib_hash = reference.file_hash(calib)

    con = db.connect()
    db.save_model(con, model_hash, info)

    print("running llama.cpp with no settings overridden - its own compiled-in "
          "defaults for -ngl, -t, -ctk/-ctv, -ot\n")

    ref = reference.ensure_reference(ref_model, calib)
    s = speed.measure_default(model)
    if s is None:
        sys.exit("default speed measurement failed")
    q = quality.measure_kl_default(model, ref, calib)

    print(f"gen:     {s.gen_tps:.1f} tok/s")
    print(f"prompt:  {s.prompt_tps:.1f} tok/s")
    if q:
        print(f"quality: {quality.quality_score(q):.4f}   kl_mean: {q.kl_mean:.6f}")
    else:
        print("quality: measurement failed")

    cand = SimpleNamespace(label="llama.cpp default", ngl=None, threads=None,
                           ctk="default", ot_pattern=None)
    ev = SimpleNamespace(gen_tps=s.gen_tps,
                        quality=quality.quality_score(q) if q else None,
                        kl_mean=q.kl_mean if q else None,
                        vram_mb=s.vram_used_mb, spilled=s.spilled)
    db.save_evaluation(con, model_hash, hw, cand, ev, calib_hash)

    print(f"\nrun it exactly as llama.cpp ships, no flags at all:\n"
          f"  llama-server.exe -m {model.name}")
    print("\n(this baseline is now saved and will show up in 'optimum start' and "
          "'optimum report' next to your tuned candidates)")


def cmd_predict(args):
    import costmodel
    import db

    model = _resolve(args.model)
    info = model_info.inspect(model)

    con = db.connect()
    models = {m["model_hash"]: m for m in db.load_all(con, "models")}
    runs = db.load_all(con, "runs")
    rows = costmodel.build_training_set(runs, models)

    pred = costmodel.predict(rows, info.file_size_mb, info.n_layers,
                             args.ngl, args.threads, args.ctk)
    if pred is None:
        sys.exit("no recorded runs yet - run 'tune' or 'default' on at least one "
                 "model first, the cost model has nothing to learn from")

    print(f"predicted   {pred.gen_tps:6.1f} tok/s   quality {pred.quality:.4f}   "
          f"confidence={pred.confidence}  "
          f"(nearest neighbor distance {pred.nearest_dist:.2f}, "
          f"{pred.n_neighbors} of {pred.n_training_rows} recorded runs used)")

    import reference
    model_hash = reference.file_hash(model)
    actual = next((r for r in runs if r["model_hash"] == model_hash
                   and r["ngl"] == args.ngl and r["threads"] == args.threads
                   and r["ctk"] == args.ctk and r["gen_tps"] is not None), None)
    if actual:
        print(f"actual      {actual['gen_tps']:6.1f} tok/s   quality {actual['quality']:.4f}   "
              f"(this exact setting was already measured for real)")
        print(f"error       {pred.gen_tps - actual['gen_tps']:+6.1f} tok/s   "
              f"{pred.quality - actual['quality']:+.4f} quality")
    else:
        print("(no real measurement of this exact setting on this model to check against)")


def cmd_serve(args):
    import subprocess
    import time
    import webbrowser
    from types import SimpleNamespace

    import db
    import frontier
    import reference
    import requests
    from config import LLAMA_DIR

    model = _resolve(args.model)
    model_hash = reference.file_hash(model)

    con = db.connect()
    runs = [r for r in db.load_all(con, "runs")
            if r["model_hash"] == model_hash
            and r["gen_tps"] is not None and r["quality"] is not None]
    if not runs:
        sys.exit(f"no tuned settings found for {model.name} yet.\n"
                 f"run one of these first:\n"
                 f"  python cli.py tune {model.name} --ref-model <higher-precision-model>\n"
                 f"  python cli.py default {model.name} --ref-model <higher-precision-model>")

    wrapped = [SimpleNamespace(gen_tps=r["gen_tps"], quality=r["quality"], _row=r) for r in runs]
    front = frontier.frontier(wrapped)
    good = [w for w in front if w.quality >= args.min_quality]
    pick = max(good, key=lambda w: w.gen_tps) if good else max(front, key=lambda w: w.gen_tps)
    best = pick._row

    server_args = [str(LLAMA_DIR / "llama-server.exe"), "-m", str(model),
                  "-c", str(args.ctx), "--port", str(args.port)]
    if best["ngl"] is not None:
        server_args += ["-ngl", str(best["ngl"])]
    if best["threads"] is not None:
        server_args += ["-t", str(best["threads"])]
    if best["ctk"] and best["ctk"] != "default":
        server_args += ["-ctk", best["ctk"], "-ctv", best["ctk"]]
    if best["ot_pattern"]:
        server_args += ["-ot", best["ot_pattern"]]

    print(f"using settings: {best['label']}  ({best['gen_tps']:.1f} tok/s, "
          f"quality {best['quality']:.4f})")
    print(" ".join(server_args))

    proc = subprocess.Popen(server_args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"\nwaiting for llama-server to come up at {url} ...")

    ready = False
    for _ in range(90):
        if proc.poll() is not None:
            sys.exit(f"llama-server exited early (code {proc.returncode}) - "
                     f"check the settings above, or that the port isn't already in use")
        try:
            if requests.get(url + "health", timeout=1).status_code == 200:
                ready = True
                break
        except requests.RequestException:
            pass
        time.sleep(1)

    if not ready:
        proc.kill()
        sys.exit("llama-server did not become healthy in time")

    print(f"ready - opening {url} (llama.cpp's own web UI)")
    webbrowser.open(url)
    print("press Ctrl+C to stop the server")
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\nstopping llama-server...")
        proc.terminate()
        proc.wait()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="optimum",
        description="Auto-tune llama.cpp for your GPU and model.")
    # Subcommands are declared in the order you'd actually use them, so
    # `optimum --help` reads as a walkthrough rather than an alphabet soup.
    # Each keeps its pre-0.2 name as an alias so existing scripts don't break.
    sub = p.add_subparsers(dest="cmd", required=True, metavar="<command>")

    gpu = sub.add_parser("gpu", aliases=["probe"],
                         help="show your GPU and memory")
    gpu.set_defaults(func=cmd_probe)

    ins = sub.add_parser("inspect",
                         help="show a model's size, layers and parts")
    ins.add_argument("model")
    ins.set_defaults(func=cmd_inspect)

    ana = sub.add_parser("analyze", aliases=["sensitivity"],
                         help="measure which parts of a model are worth GPU space")
    ana.add_argument("model")
    ana.set_defaults(func=cmd_sensitivity)

    tune = sub.add_parser("tune",
                          help="find the best settings for a model")
    tune.add_argument("model")
    tune.add_argument("--ref-model", help="higher-precision model to measure quality against (defaults to the model itself)")
    tune.add_argument("--calib", default=str(CALIB_FILE))
    tune.add_argument("--min-quality", type=float, default=0.97)
    tune.set_defaults(func=cmd_tune)

    base = sub.add_parser("baseline", aliases=["default"],
                          help="measure llama.cpp untuned, to compare against")
    base.add_argument("model")
    base.add_argument("--ref-model", help="higher-precision model to measure quality against (defaults to the model itself)")
    base.add_argument("--calib", default=str(CALIB_FILE))
    base.set_defaults(func=cmd_default)

    pred = sub.add_parser("predict",
                          help="estimate speed and quality without running anything")
    pred.add_argument("model")
    pred.add_argument("--ngl", type=int, default=99)
    pred.add_argument("--threads", type=int, default=6)
    pred.add_argument("--ctk", default="f16")
    pred.set_defaults(func=cmd_predict)

    hist = sub.add_parser("history", aliases=["report"],
                          help="list past runs")
    hist.add_argument("--limit", type=int, default=20)
    hist.set_defaults(func=cmd_report)

    dash = sub.add_parser("dashboard", aliases=["start"],
                          help="open the dashboard with charts and results")
    dash.add_argument("--port", type=int, default=8765)
    dash.add_argument("--no-browser", action="store_true", help="don't auto-open a browser tab")
    dash.set_defaults(func=cmd_start)

    run = sub.add_parser("run", aliases=["serve"],
                         help="launch the model with its best settings and open the web UI")
    run.add_argument("model")
    run.add_argument("--ctx", type=int, default=4096)
    run.add_argument("--port", type=int, default=8080)
    run.add_argument("--min-quality", type=float, default=0.9)
    run.set_defaults(func=cmd_serve)

    return p


def main():
    """Console-script entry point (registered in pyproject.toml as `optimum`)."""
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
