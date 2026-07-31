import json
from pathlib import Path

def render_text(model_name: str, hw, front: list, best) -> str:
    lines = [f"model: {model_name}   gpu: {hw.name}   {hw.free_mb:.0f} MB free", ""]
    lines.append("--- good options ---")
    for e in sorted(front, key=lambda x: -x.gen_tps):
        mark = " <-- pick" if e is best else ""
        lines.append(f"  {e.gen_tps:6.1f} tok/s   quality {e.quality:.4f}   "
                     f"{e.candidate.label}{mark}")
    if best:
        lines.append("")
        lines.append(launch_command(best.candidate, model_name))
    else:
        lines.append("\nNothing reached the quality floor.")
    return "\n".join(lines)

def launch_command(cand, model_name) -> str:
    name = Path(model_name).name if isinstance(model_name, (str, Path)) else model_name
    ot = f' -ot "{cand.ot_pattern}"' if cand.ot_pattern else ""
    return (f"llama-server.exe -m {name} -ngl {cand.ngl}"
            f"{ot} -t {cand.threads} -ctk {cand.ctk}")

def export_json(model_name: str, hw, front: list, best, path: Path) -> None:
    def ev_to_dict(e):
        return {
            "gen_tps": e.gen_tps,
            "quality": e.quality,
            "kl_mean": e.kl_mean,
            "vram_mb": e.vram_mb,
            "spilled": e.spilled,
            "candidate": {
                "label": e.candidate.label,
                "ngl": e.candidate.ngl,
                "threads": e.candidate.threads,
                "ctk": e.candidate.ctk,
                "ot_pattern": e.candidate.ot_pattern,
                "gpu_groups": e.candidate.gpu_groups,
                "cpu_groups": e.candidate.cpu_groups,
            },
        }

    data = {
        "model": str(model_name),
        "gpu": hw.name,
        "front": [ev_to_dict(e) for e in front],
        "best": ev_to_dict(best) if best else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
