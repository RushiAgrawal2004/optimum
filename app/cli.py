import argparse
import sys
from pathlib import Path

import hardware
import model_info
import sensitivity
import speed
import tune as tune_module
from config import MODEL_DIR


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
        print(f"{r['ts']}  {r['label']:<12}  {r['gen_tps'] or 0:6.1f} tok/s  "
              f"quality {r['quality'] or 0:.4f}  ngl={r['ngl']}")


def cmd_start(args):
    import webapp
    webapp.serve(port=args.port, open_browser=not args.no_browser)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="optimum")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("probe", help="show GPU/RAM").set_defaults(func=cmd_probe)

    i = sub.add_parser("inspect", help="show a model's parts and sizes")
    i.add_argument("model")
    i.set_defaults(func=cmd_inspect)

    s = sub.add_parser("sensitivity", help="value of each tensor group")
    s.add_argument("model")
    s.set_defaults(func=cmd_sensitivity)

    t = sub.add_parser("tune", help="find the best settings for a model")
    t.add_argument("model")
    t.add_argument("--ref-model", help="unsqueezed/high-precision model for the quality baseline (defaults to --model)")
    t.add_argument("--calib", default=r"C:\nativetune\data\calib-50kb.txt")
    t.add_argument("--min-quality", type=float, default=0.97)
    t.set_defaults(func=cmd_tune)

    r = sub.add_parser("report", help="show recent recorded runs")
    r.add_argument("--limit", type=int, default=20)
    r.set_defaults(func=cmd_report)

    st = sub.add_parser("start", help="launch the local Optimum dashboard (graphs + metrics)")
    st.add_argument("--port", type=int, default=8765)
    st.add_argument("--no-browser", action="store_true", help="don't auto-open a browser tab")
    st.set_defaults(func=cmd_start)

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
