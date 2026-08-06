# Optimum

Auto-tunes `llama.cpp` for your GPU and model. It measures real speed **and**
real answer-quality on your own machine, then launches the model with the
settings that won.

---

## Install

Windows, PowerShell:

```powershell
irm https://optimumtune.github.io/install.ps1 | iex
```

Open a new terminal, then check it worked:

```powershell
optimum probe
```

---

## Setup

Optimum does not ship `llama.cpp` or any models — you supply both. Put them in
`%LOCALAPPDATA%\nativetune\`:

| Path | What goes there |
|---|---|
| `llama\` | `llama.cpp` binaries (`llama-bench.exe`, `llama-perplexity.exe`, `llama-server.exe`, ...) |
| `models\` | your `.gguf` model files |
| `data\calib-50kb.txt` | ~50KB of plain text for quality measurement (the wikitext-2 test set works well) |

You also need a **reference model** — a Q8 or F16 quant of the same model.
Quality is measured as damage relative to it, so it needs something more
precise to compare against.

---

## Commands

| Command | What it does |
|---|---|
| `probe` | Show GPU name, free/usable VRAM, free RAM |
| `inspect <model>` | Show architecture, layer count, and tensor-group sizes |
| `sensitivity <model>` | Measure how much speed each tensor group is worth per MB |
| `tune <model> --ref-model <ref>` | Full pipeline — measure, build candidates, benchmark speed + quality, pick the best above the quality floor |
| `default <model> --ref-model <ref>` | Measure `llama.cpp` with zero settings overridden — the untuned baseline |
| `predict <model>` | Estimate speed/quality for a setting combo without running it |
| `report` | List recorded runs |
| `start` | Open the local dashboard — charts, metrics, launch commands |
| `serve <model>` | Launch `llama-server.exe` with the best known settings and open its web UI |

Common flags: `--min-quality 0.97`, `--ctx 4096`, `--port`, `--limit`.
Run `optimum <command> --help` for the full list.

---

## Quick start

```powershell
optimum tune qwen1.5b-q4.gguf --ref-model qwen1.5b-q8.gguf --min-quality 0.9
optimum default qwen1.5b-q4.gguf --ref-model qwen1.5b-q8.gguf   # baseline to compare against
optimum start                                                    # view results as charts
optimum serve qwen1.5b-q4.gguf                                   # run it for real
```

Every measurement is saved to `results.db`, so nothing is ever re-run
unnecessarily.

---

## Configuration

All paths are overridable by environment variable:

| Variable | Default |
|---|---|
| `NATIVETUNE_HOME` | root for everything below |
| `NATIVETUNE_LLAMA_DIR` | `<home>\llama` |
| `NATIVETUNE_MODEL_DIR` | `<home>\models` |
| `NATIVETUNE_CACHE_DIR` | `<home>\cache` |
| `NATIVETUNE_DB_PATH` | `<home>\results.db` |
| `NATIVETUNE_CALIB_FILE` | `<home>\data\calib-50kb.txt` |

See [`app/config.py`](app/config.py).

---

## Development

Run from a clone instead of the installer:

```powershell
cd app
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install nvidia-ml-py psutil gguf pytest requests
python cli.py probe        # same commands, via cli.py
```

Tests:

```powershell
python -m pytest -q    # fast, no GPU
python sanity.py       # slow, needs the GPU
```

---

## Limitations

- **Not tested on a model too large for VRAM** — the case this project exists
  for. Everything tested so far fit entirely on-GPU.
- **MoE models**: `-ot` tensor-pinning won't match their tensor names
  (`ffn_gate_exps` vs the expected `ffn_gate`).
- **Mamba/SSM architectures** report a mislabeled "floor" sensitivity value.
- **`predict` is nearest-neighbor, not a trained model.** It reads model
  *shape* (file size, layer count), not the weights — so it cannot tell a
  good quant from a damaged one of similar size. Use it to shortlist
  settings, never as a substitute for `tune`.

[`CHANGELOG.md`](CHANGELOG.md) has the full history.
