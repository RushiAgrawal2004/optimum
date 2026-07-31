import re
from dataclasses import dataclass
from pathlib import Path
import runner
from config import PERPLEXITY

@dataclass
class QualityResult:
    kl_mean: float
    kl_p99: float
    top1_agree: float

def parse_kl_output(text: str) -> QualityResult | None:
    """Pure function. Test against a saved output file."""
    def grab(pattern):
        m = re.search(pattern, text, re.IGNORECASE)
        return float(m.group(1)) if m else None

    kl_mean = grab(r"mean\s+kld\s*[:=]\s*([\d.eE+-]+)")
    kl_p99  = grab(r"99\.?\d*%\s*kld\s*[:=]\s*([\d.eE+-]+)")
    agree   = grab(r"same\s+top\s*p\s*[:=]\s*([\d.]+)")

    if kl_mean is None:
        return None
    if agree is not None and agree > 1.5:      # printed as a percentage
        agree = agree / 100.0
    return QualityResult(kl_mean, kl_p99 or kl_mean, agree or 0.0)

def measure_kl(model: Path, ref_kld: Path, calib: Path,
               ngl: int, ot: str = None, ctk: str = "f16") -> QualityResult | None:
    args = ["-m", model, "-f", calib,
            "--kl-divergence-base", ref_kld, "--kl-divergence",
            "-ngl", ngl, "-c", 512, "-ctk", ctk, "-ctv", ctk]
    if ot:
        args += ["-ot", ot]
    res = runner.run(PERPLEXITY, args, timeout=900)
    if not res.ok:
        return None
    return parse_kl_output(res.stdout + res.stderr)

def measure_kl_default(model: Path, ref_kld: Path, calib: Path) -> QualityResult | None:
    """Runs llama-perplexity with none of -ngl/-ctk/-ctv/-ot set — llama.cpp's own
    compiled-in defaults, not a NativeTune choice."""
    args = ["-m", model, "-f", calib,
            "--kl-divergence-base", ref_kld, "--kl-divergence"]
    res = runner.run(PERPLEXITY, args, timeout=900)
    if not res.ok:
        return None
    return parse_kl_output(res.stdout + res.stderr)

def quality_score(q: QualityResult) -> float:
    """One number, 0 to 1. Higher is better."""
    if q.top1_agree > 0:
        return q.top1_agree
    return 1.0 / (1.0 + q.kl_mean)
