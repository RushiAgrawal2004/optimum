import subprocess, time
from dataclasses import dataclass
from pathlib import Path

@dataclass
class RunResult:
    ok: bool
    stdout: str
    stderr: str
    seconds: float
    reason: str = ""

def run(exe: Path, args: list, timeout: int = 300) -> RunResult:
    cmd = [str(exe)] + [str(a) for a in args]
    start = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return RunResult(False, "", "", time.time() - start, "timeout")
    except FileNotFoundError:
        return RunResult(False, "", "", 0, "exe not found")

    elapsed = time.time() - start
    if p.returncode != 0:
        return RunResult(False, p.stdout, p.stderr, elapsed, f"exit {p.returncode}")
    return RunResult(True, p.stdout, p.stderr, elapsed)

def build_ot_pattern(cpu_groups: list) -> str | None:
    """['ffn_up','ffn_gate'] -> 'ffn_(up|gate)\\.weight=CPU'"""
    if not cpu_groups:
        return None
    names = "|".join(sorted(cpu_groups))
    return rf"({names})\.weight=CPU"