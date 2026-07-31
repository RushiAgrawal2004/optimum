"""Local dashboard: serves a single self-contained HTML page (no external
requests, no CDN) showing hardware, model metrics, tensor-group sensitivity,
and the speed/quality trade-off for every recorded run. Reads straight from
results.db and cache/, so it reflects whatever tune.py has produced so far."""
from __future__ import annotations

import html
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import db
import frontier
import hardware
from config import LLAMA_DIR, TENSOR_GROUPS


def _quote(s: str) -> str:
    return f'"{s}"' if " " in s else s


def _command_for(model_path: str | None, run: dict) -> str | None:
    """The exact llama-server.exe command line for one recorded run, built the
    same way report.launch_command() does but usable directly on a raw db row."""
    if not model_path:
        return None
    parts = [str(LLAMA_DIR / "llama-server.exe"), "-m", model_path]
    if run.get("ngl") is not None:
        parts += ["-ngl", str(run["ngl"])]
    if run.get("threads") is not None:
        parts += ["-t", str(run["threads"])]
    if run.get("ctk") and run["ctk"] != "default":
        parts += ["-ctk", run["ctk"], "-ctv", run["ctk"]]
    cmd = " ".join(_quote(p) for p in parts)
    if run.get("ot_pattern"):
        cmd += f' -ot "{run["ot_pattern"]}"'
    return cmd


def _latest_sensitivity_by_model(sens_rows: list[dict]) -> dict[str, list[dict]]:
    latest_ts: dict[tuple[str, str], str] = {}
    for r in sens_rows:
        k = (r["model_hash"], r["gpu"])
        if k not in latest_ts or r["ts"] > latest_ts[k]:
            latest_ts[k] = r["ts"]
    out: dict[str, list[dict]] = {}
    for r in sens_rows:
        if latest_ts.get((r["model_hash"], r["gpu"])) == r["ts"]:
            out.setdefault(r["model_hash"], []).append(r)
    for rows in out.values():
        rows.sort(key=lambda r: -r["value_per_mb"])
    return out


def _frontier_labels(runs: list[dict]) -> set[int]:
    """Row ids of runs on the speed/quality frontier, within their model group."""
    kept: set[int] = set()
    by_model: dict[str, list[dict]] = {}
    for r in runs:
        if r["gen_tps"] is not None and r["quality"] is not None:
            by_model.setdefault(r["model_hash"], []).append(r)
    for group in by_model.values():
        wrapped = [SimpleNamespace(gen_tps=r["gen_tps"], quality=r["quality"], _id=r["id"]) for r in group]
        for w in frontier.frontier(wrapped):
            kept.add(w._id)
    return kept


def gather() -> dict:
    hw = hardware.probe()
    con = db.connect()
    models = {m["model_hash"]: m for m in db.load_all(con, "models")}
    runs = db.load_all(con, "runs")
    sens_by_model = _latest_sensitivity_by_model(db.load_all(con, "sensitivity"))
    on_frontier = _frontier_labels(runs)

    best_id = None
    best_tps = -1
    for r in runs:
        if r["id"] in on_frontier and r["quality"] and r["quality"] >= 0.8 and r["gen_tps"] > best_tps:
            best_id, best_tps = r["id"], r["gen_tps"]

    default_tps_by_model: dict[str, float] = {}
    for r in runs:
        if r["label"] == "llama.cpp default" and r["gen_tps"]:
            default_tps_by_model[r["model_hash"]] = r["gen_tps"]

    for r in runs:
        r["on_frontier"] = r["id"] in on_frontier
        r["is_best"] = r["id"] == best_id
        r["is_default"] = r["label"] == "llama.cpp default"
        model_path = models.get(r["model_hash"], {}).get("path")
        r["command"] = _command_for(model_path, r)

    best = None
    before_after = None
    if best_id is not None:
        best = next(r for r in runs if r["id"] == best_id)
        default_row = next(
            (r for r in runs if r["is_default"] and r["model_hash"] == best["model_hash"]),
            None,
        )
        default_tps = default_row["gen_tps"] if default_row else None
        best["speedup_pct"] = (
            round(100 * (best["gen_tps"] / default_tps - 1), 1)
            if default_tps else None
        )
        if default_row:
            before_after = {
                "before": {"label": default_row["label"], "gen_tps": default_row["gen_tps"],
                          "quality": default_row["quality"]},
                "after": {"label": best["label"], "gen_tps": best["gen_tps"],
                         "quality": best["quality"]},
            }

    return {
        "hw": {
            "name": hw.name, "free_mb": hw.free_mb, "total_mb": hw.total_mb,
            "usable_mb": hardware.usable_vram_mb(hw),
            "ram_free_mb": hw.ram_free_mb, "ram_total_mb": hw.ram_total_mb,
        },
        "models": models,
        "sensitivity": sens_by_model,
        "runs": runs,
        "best": best,
        "before_after": before_after,
        "tensor_groups": TENSOR_GROUPS,
    }


def render(data: dict) -> str:
    payload = json.dumps(data)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Optimum — NativeTune Dashboard</title>
<style>
  :root {{
    color-scheme: light;
    --surface-1:      #fcfcfb;
    --page:           #f9f9f7;
    --text-primary:   #0b0b0b;
    --text-secondary: #52514e;
    --text-muted:     #898781;
    --grid:           #e1e0d9;
    --axis:           #c3c2b7;
    --border:         rgba(11,11,11,0.10);
    --series-1:       #2a78d6;   /* blue: kept / frontier */
    --series-dominated: #c3c2b7; /* muted: dropped */
    --good:           #0ca30c;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:where(:not([data-theme="light"])) {{
      color-scheme: dark;
      --surface-1:      #1a1a19;
      --page:           #0d0d0d;
      --text-primary:   #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted:     #898781;
      --grid:           #2c2c2a;
      --axis:           #383835;
      --border:         rgba(255,255,255,0.10);
      --series-1:       #3987e5;
      --series-dominated: #383835;
      --good:           #0ca30c;
    }}
  }}
  :root[data-theme="dark"] {{
    color-scheme: dark;
    --surface-1:      #1a1a19;
    --page:           #0d0d0d;
    --text-primary:   #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted:     #898781;
    --grid:           #2c2c2a;
    --axis:           #383835;
    --border:         rgba(255,255,255,0.10);
    --series-1:       #3987e5;
    --series-dominated: #383835;
    --good:           #0ca30c;
  }}

  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--page); color: var(--text-primary);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  }}
  header {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 20px 28px; border-bottom: 1px solid var(--border);
  }}
  header h1 {{ font-size: 18px; margin: 0; font-weight: 600; }}
  header .sub {{ color: var(--text-secondary); font-size: 13px; margin-top: 2px; }}
  main {{ max-width: 1080px; margin: 0 auto; padding: 24px 28px 64px; }}
  section {{ margin-bottom: 40px; }}
  section > h2 {{ font-size: 14px; font-weight: 600; color: var(--text-secondary);
                  text-transform: uppercase; letter-spacing: 0.04em; margin: 0 0 14px; }}

  .tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 12px; }}
  .tile {{ background: var(--surface-1); border: 1px solid var(--border);
           border-radius: 8px; padding: 14px 16px; }}
  .tile .label {{ font-size: 12px; color: var(--text-secondary); }}
  .tile .value {{ font-size: 22px; font-weight: 600; margin-top: 4px; }}

  .card {{ background: var(--surface-1); border: 1px solid var(--border);
           border-radius: 10px; padding: 20px; }}

  .bar-row {{ display: flex; align-items: center; gap: 10px; margin-bottom: 4px; }}
  .bar-label {{ width: 110px; font-size: 12px; color: var(--text-secondary);
                text-align: right; flex-shrink: 0; }}
  .bar-track {{ flex: 1; }}
  .bar-fill {{ height: 20px; background: var(--series-1);
               border-radius: 0 4px 4px 0; display: flex; align-items: center;
               min-width: 2px; }}
  .bar-fill.default {{ background: var(--series-dominated); }}
  .bar-fill.tuned {{ background: var(--series-1); }}
  .ba-headline {{ font-size: 16px; font-weight: 600; margin-bottom: 16px; }}
  .ba-headline .good {{ color: var(--good); }}
  .bar-value {{ font-size: 12px; color: var(--text-secondary); margin-left: 8px;
                white-space: nowrap; }}

  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--grid); }}
  th {{ color: var(--text-secondary); font-weight: 600; font-size: 12px;
        text-transform: uppercase; letter-spacing: 0.03em; }}
  td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  tr.best td {{ color: var(--good); font-weight: 600; }}

  .legend {{ display: flex; gap: 18px; margin-bottom: 10px; font-size: 12px;
             color: var(--text-secondary); }}
  .legend .swatch {{ display: inline-block; width: 10px; height: 10px;
                      border-radius: 50%; margin-right: 6px; vertical-align: middle; }}

  #tooltip {{ position: fixed; pointer-events: none; background: var(--text-primary);
              color: var(--page); font-size: 12px; padding: 6px 9px; border-radius: 6px;
              opacity: 0; transform: translate(-50%, -130%); transition: opacity 0.08s;
              white-space: nowrap; z-index: 10; }}

  .theme-toggle {{ border: 1px solid var(--border); background: var(--surface-1);
                    color: var(--text-primary); border-radius: 6px; padding: 6px 12px;
                    font-size: 12px; cursor: pointer; }}

  .empty {{ color: var(--text-muted); font-size: 13px; padding: 20px; text-align: center; }}

  .best-pick {{ border: 1px solid var(--good); }}
  .best-pick .kicker {{ font-size: 12px; color: var(--good); font-weight: 600;
                         text-transform: uppercase; letter-spacing: 0.04em; }}
  .best-pick .headline {{ font-size: 20px; font-weight: 600; margin: 4px 0 2px; }}
  .best-pick .sub {{ font-size: 13px; color: var(--text-secondary); margin-bottom: 14px; }}
  .cmd-row {{ display: flex; align-items: stretch; gap: 8px; }}
  .cmd-box {{ flex: 1; background: var(--page); border: 1px solid var(--border);
              border-radius: 6px; padding: 10px 12px; font-family: ui-monospace, monospace;
              font-size: 12.5px; color: var(--text-primary); overflow-x: auto;
              white-space: pre; }}
  .copy-btn {{ border: 1px solid var(--border); background: var(--surface-1);
               color: var(--text-primary); border-radius: 6px; padding: 0 14px;
               font-size: 12px; cursor: pointer; white-space: nowrap; }}
  td.cmd {{ font-family: ui-monospace, monospace; font-size: 11.5px;
            max-width: 360px; overflow-wrap: anywhere; }}
  .row-copy {{ border: 1px solid var(--border); background: var(--surface-1);
               color: var(--text-secondary); border-radius: 5px; padding: 2px 8px;
               font-size: 11px; cursor: pointer; margin-left: 8px; }}
</style>
</head>
<body>
<header>
  <div>
    <h1>Optimum</h1>
    <div class="sub">local llama.cpp auto-tuner — speed &amp; quality, measured on your own GPU</div>
  </div>
  <button class="theme-toggle" onclick="toggleTheme()">Toggle theme</button>
</header>
<main id="app"></main>
<div id="tooltip"></div>
<script>
const DATA = {payload};
</script>
<script>
function toggleTheme() {{
  const root = document.documentElement;
  const cur = root.getAttribute('data-theme');
  root.setAttribute('data-theme', cur === 'dark' ? 'light' : 'dark');
}}

function fmt(n, d) {{ return (n === null || n === undefined) ? '—' : Number(n).toFixed(d); }}

function copyText(text, btn) {{
  const done = () => {{
    const orig = btn.textContent;
    btn.textContent = 'Copied!';
    setTimeout(() => {{ btn.textContent = orig; }}, 1200);
  }};
  if (navigator.clipboard && navigator.clipboard.writeText) {{
    navigator.clipboard.writeText(text).then(done).catch(done);
  }} else {{
    done();
  }}
}}

function el(tag, attrs, children) {{
  const e = document.createElement(tag);
  for (const k in (attrs || {{}})) e.setAttribute(k, attrs[k]);
  (children || []).forEach(c => e.appendChild(typeof c === 'string' ? document.createTextNode(c) : c));
  return e;
}}

function statTile(label, value) {{
  const t = el('div', {{class: 'tile'}});
  t.appendChild(el('div', {{class: 'label'}}, [label]));
  t.appendChild(el('div', {{class: 'value'}}, [value]));
  return t;
}}

function renderHardware(hw) {{
  const s = el('section');
  s.appendChild(el('h2', {{}}, ['Hardware']));
  const tiles = el('div', {{class: 'tiles'}});
  tiles.appendChild(statTile('GPU', hw.name));
  tiles.appendChild(statTile('VRAM free', fmt(hw.free_mb, 0) + ' MB'));
  tiles.appendChild(statTile('VRAM usable', fmt(hw.usable_mb, 0) + ' MB'));
  tiles.appendChild(statTile('RAM free', fmt(hw.ram_free_mb, 0) + ' MB'));
  s.appendChild(tiles);
  return s;
}}

function renderModel(hash, model, sensRows) {{
  const s = el('section');
  s.appendChild(el('h2', {{}}, ['Model — ' + (model ? model.arch : hash.slice(0,8))]));

  if (model) {{
    const tiles = el('div', {{class: 'tiles'}});
    tiles.appendChild(statTile('Architecture', model.arch));
    tiles.appendChild(statTile('Layers', String(model.n_layers)));
    tiles.appendChild(statTile('File size', fmt(model.file_size_mb, 0) + ' MB'));
    s.appendChild(tiles);
  }}

  if (sensRows && sensRows.length) {{
    const card = el('div', {{class: 'card', style: 'margin-top:14px'}});
    card.appendChild(el('div', {{style: 'font-size:13px;color:var(--text-secondary);margin-bottom:12px'}},
      ['Tensor-group value — speed gained per GB if that group is kept on GPU']));
    const maxVal = Math.max(...sensRows.map(r => r.value_per_mb), 0.001);
    sensRows.forEach(r => {{
      const row = el('div', {{class: 'bar-row'}});
      row.appendChild(el('div', {{class: 'bar-label'}}, [r.group_name]));
      const track = el('div', {{class: 'bar-track'}});
      const pct = Math.max(2, 100 * Math.max(r.value_per_mb, 0) / maxVal);
      const fill = el('div', {{class: 'bar-fill', style: 'width:' + pct + '%'}});
      track.appendChild(fill);
      row.appendChild(track);
      row.appendChild(el('div', {{class: 'bar-value'}},
        [fmt(r.value_per_mb, 1) + '/GB · ' + fmt(r.size_mb, 0) + ' MB']));
      card.appendChild(row);
    }});

    const tbl = el('table', {{style: 'margin-top:16px'}});
    const thead = el('tr', {{}}, [
      el('th', {{}}, ['Group']), el('th', {{class:'num'}}, ['Size (MB)']),
      el('th', {{class:'num'}}, ['Gain (tok/s)']), el('th', {{class:'num'}}, ['Value/GB'])]);
    tbl.appendChild(el('thead', {{}}, [thead]));
    const tbody = el('tbody');
    sensRows.forEach(r => {{
      tbody.appendChild(el('tr', {{}}, [
        el('td', {{}}, [r.group_name]),
        el('td', {{class:'num'}}, [fmt(r.size_mb, 0)]),
        el('td', {{class:'num'}}, [fmt(r.gain_tps, 2)]),
        el('td', {{class:'num'}}, [fmt(r.value_per_mb, 1)]),
      ]));
    }});
    tbl.appendChild(tbody);
    card.appendChild(tbl);
    s.appendChild(card);
  }}
  return s;
}}

function metricPair(title, beforeVal, afterVal, unit, decimals) {{
  const wrap = el('div', {{style: 'margin-bottom:22px'}});
  wrap.appendChild(el('div', {{style: 'font-size:13px;color:var(--text-secondary);margin-bottom:8px'}}, [title]));
  const maxVal = Math.max(beforeVal || 0, afterVal || 0, 0.001);
  [['default', 'Before', beforeVal], ['tuned', 'After', afterVal]].forEach(([cls, lbl, val]) => {{
    const row = el('div', {{class: 'bar-row'}});
    row.appendChild(el('div', {{class: 'bar-label'}}, [lbl]));
    const track = el('div', {{class: 'bar-track'}});
    const pct = Math.max(2, 100 * Math.max(val || 0, 0) / maxVal);
    track.appendChild(el('div', {{class: 'bar-fill ' + cls, style: 'width:' + pct + '%'}}));
    row.appendChild(track);
    row.appendChild(el('div', {{class: 'bar-value'}}, [fmt(val, decimals) + ' ' + unit]));
    wrap.appendChild(row);
  }});
  return wrap;
}}

function renderBeforeAfter(ba) {{
  const s = el('section');
  s.appendChild(el('h2', {{}}, ['Before vs. after tuning']));
  const card = el('div', {{class: 'card'}});

  if (!ba) {{
    card.appendChild(el('div', {{class: 'empty'}}, [
      'Run "python cli.py default <model>" to measure the untouched baseline, then it will compare here against your tuned pick.']));
    s.appendChild(card);
    return s;
  }}

  const legend = el('div', {{class: 'legend'}}, [
    el('span', {{}}, [el('span', {{class:'swatch', style:'background:var(--series-dominated)'}}), 'Before — ' + ba.before.label]),
    el('span', {{}}, [el('span', {{class:'swatch', style:'background:var(--series-1)'}}), 'After — ' + ba.after.label]),
  ]);
  card.appendChild(legend);

  const speedPct = ba.before.gen_tps ? Math.round(1000 * (ba.after.gen_tps / ba.before.gen_tps - 1)) / 10 : null;
  const qDelta = (ba.after.quality != null && ba.before.quality != null) ? ba.after.quality - ba.before.quality : null;
  let headline = '';
  if (speedPct !== null) {{
    headline += (speedPct >= 0 ? '+' : '') + speedPct + '% speed';
  }}
  if (qDelta !== null) {{
    headline += (headline ? ', ' : '') + (qDelta >= 0 ? '+' : '') + fmt(qDelta, 4) + ' quality';
  }}
  if (headline) {{
    card.appendChild(el('div', {{class: 'ba-headline'}}, [el('span', {{class: 'good'}}, [headline]),
      ' from tuning, on the same model and GPU']));
  }}

  card.appendChild(metricPair('Generation speed', ba.before.gen_tps, ba.after.gen_tps, 'tok/s', 1));
  card.appendChild(metricPair('Quality score', ba.before.quality, ba.after.quality, '', 4));

  s.appendChild(card);
  return s;
}}

function renderBestPick(best) {{
  const s = el('section');
  s.appendChild(el('h2', {{}}, ['Recommended launch command']));
  const card = el('div', {{class: 'card best-pick'}});

  if (!best) {{
    card.appendChild(el('div', {{class: 'empty'}},
      ['No winning candidate yet — run "python cli.py tune <model>".']));
    s.appendChild(card);
    return s;
  }}

  card.appendChild(el('div', {{class: 'kicker'}}, ['Best pick — ' + best.label]));
  card.appendChild(el('div', {{class: 'headline'}},
    [fmt(best.gen_tps,1) + ' tok/s at quality ' + fmt(best.quality,4)]));

  let subText = 'ngl=' + best.ngl + ' · threads=' + best.threads + ' · ctk=' + best.ctk;
  if (best.speedup_pct !== null && best.speedup_pct !== undefined) {{
    subText += '  —  ' + (best.speedup_pct >= 0 ? '+' : '') + best.speedup_pct +
      '% vs. untouched llama.cpp default';
  }}
  card.appendChild(el('div', {{class: 'sub'}}, [subText]));

  if (best.command) {{
    const row = el('div', {{class: 'cmd-row'}});
    const box = el('div', {{class: 'cmd-box'}}, [best.command]);
    const btn = el('button', {{class: 'copy-btn'}}, ['Copy']);
    btn.addEventListener('click', () => copyText(best.command, btn));
    row.appendChild(box);
    row.appendChild(btn);
    card.appendChild(row);
  }}

  s.appendChild(card);
  return s;
}}

function renderScatter(runs) {{
  const s = el('section');
  s.appendChild(el('h2', {{}}, ['Speed vs. quality']));
  const card = el('div', {{class: 'card'}});

  if (!runs.length) {{
    card.appendChild(el('div', {{class: 'empty'}}, ['No runs recorded yet — run "python cli.py tune <model>".']));
    s.appendChild(card);
    return s;
  }}

  const legend = el('div', {{class: 'legend'}}, [
    el('span', {{}}, [el('span', {{class:'swatch', style:'background:var(--series-1)'}}), 'Kept (on frontier)']),
    el('span', {{}}, [el('span', {{class:'swatch', style:'background:var(--series-dominated)'}}), 'Dropped (dominated)']),
  ]);
  card.appendChild(legend);

  const W = 640, H = 360, PAD = 44;
  const xs = runs.map(r => r.gen_tps), ys = runs.map(r => r.quality);
  const xMin = 0, xMax = Math.max(...xs) * 1.1;
  const yMin = Math.min(0.5, Math.min(...ys) - 0.05), yMax = 1.0;
  const sx = v => PAD + (W - 2*PAD) * (v - xMin) / (xMax - xMin || 1);
  const sy = v => H - PAD - (H - 2*PAD) * (v - yMin) / (yMax - yMin || 1);

  const svgNS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(svgNS, 'svg');
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
  svg.setAttribute('width', '100%');
  svg.style.maxWidth = '640px';
  svg.style.display = 'block';

  function line(x1,y1,x2,y2,cls) {{
    const l = document.createElementNS(svgNS, 'line');
    l.setAttribute('x1',x1); l.setAttribute('y1',y1); l.setAttribute('x2',x2); l.setAttribute('y2',y2);
    l.setAttribute('stroke', 'var(--grid)'); l.setAttribute('stroke-width', '1');
    svg.appendChild(l);
  }}
  // gridlines
  for (let i = 0; i <= 4; i++) {{
    const gy = PAD + i * (H - 2*PAD) / 4;
    line(PAD, gy, W-PAD, gy);
  }}
  // axes
  const axis = document.createElementNS(svgNS, 'line');
  axis.setAttribute('x1', PAD); axis.setAttribute('y1', H-PAD);
  axis.setAttribute('x2', W-PAD); axis.setAttribute('y2', H-PAD);
  axis.setAttribute('stroke', 'var(--axis)'); axis.setAttribute('stroke-width', '1');
  svg.appendChild(axis);
  const axisY = document.createElementNS(svgNS, 'line');
  axisY.setAttribute('x1', PAD); axisY.setAttribute('y1', PAD);
  axisY.setAttribute('x2', PAD); axisY.setAttribute('y2', H-PAD);
  axisY.setAttribute('stroke', 'var(--axis)'); axisY.setAttribute('stroke-width', '1');
  svg.appendChild(axisY);

  const xLabel = document.createElementNS(svgNS, 'text');
  xLabel.setAttribute('x', W/2); xLabel.setAttribute('y', H - 8);
  xLabel.setAttribute('fill', 'var(--text-muted)'); xLabel.setAttribute('font-size', '11');
  xLabel.setAttribute('text-anchor', 'middle'); xLabel.textContent = 'tokens / sec';
  svg.appendChild(xLabel);

  const tooltip = document.getElementById('tooltip');

  runs.forEach(r => {{
    if (r.gen_tps == null || r.quality == null) return;
    const cx = sx(r.gen_tps), cy = sy(r.quality);
    const color = r.on_frontier ? 'var(--series-1)' : 'var(--series-dominated)';

    // hit target (bigger, invisible) for a comfortable hover area
    const hit = document.createElementNS(svgNS, 'circle');
    hit.setAttribute('cx', cx); hit.setAttribute('cy', cy); hit.setAttribute('r', 14);
    hit.setAttribute('fill', 'transparent');
    hit.style.cursor = 'pointer';

    const dot = document.createElementNS(svgNS, 'circle');
    dot.setAttribute('cx', cx); dot.setAttribute('cy', cy);
    dot.setAttribute('r', (r.is_best || r.is_default) ? 7 : 5);
    dot.setAttribute('fill', color);
    dot.setAttribute('stroke', 'var(--surface-1)');
    dot.setAttribute('stroke-width', r.is_default ? '3' : '2');

    const show = (evt) => {{
      tooltip.innerHTML = '<b>' + r.label + '</b><br>' + fmt(r.gen_tps,1) + ' tok/s · quality ' + fmt(r.quality,4) +
        '<br>ngl=' + r.ngl + ' t=' + r.threads + ' ctk=' + r.ctk;
      tooltip.style.left = evt.clientX + 'px';
      tooltip.style.top = evt.clientY + 'px';
      tooltip.style.opacity = 1;
    }};
    const hide = () => {{ tooltip.style.opacity = 0; }};
    hit.addEventListener('mousemove', show);
    hit.addEventListener('mouseleave', hide);
    dot.addEventListener('mousemove', show);
    dot.addEventListener('mouseleave', hide);

    svg.appendChild(hit);
    svg.appendChild(dot);

    if (r.is_best || r.is_default) {{
      const lbl = document.createElementNS(svgNS, 'text');
      lbl.setAttribute('x', cx + 10); lbl.setAttribute('y', cy - 10);
      lbl.setAttribute('fill', 'var(--text-primary)'); lbl.setAttribute('font-size', '11');
      lbl.setAttribute('font-weight', '600');
      lbl.textContent = r.is_best ? 'best pick' : 'untouched default';
      svg.appendChild(lbl);
    }}
  }});

  card.appendChild(svg);

  const tbl = el('table', {{style: 'margin-top:20px'}});
  tbl.appendChild(el('thead', {{}}, [el('tr', {{}}, [
    el('th', {{}}, ['Label']), el('th', {{class:'num'}}, ['tok/s']),
    el('th', {{class:'num'}}, ['Quality']), el('th', {{}}, ['ngl']),
    el('th', {{}}, ['threads']), el('th', {{}}, ['ctk']), el('th', {{}}, ['-ot']),
    el('th', {{}}, ['Command'])])]));
  const tbody = el('tbody');
  runs.slice().sort((a,b) => (b.gen_tps||0) - (a.gen_tps||0)).forEach(r => {{
    const cmdCell = el('td', {{class: 'cmd'}}, [r.command || '—']);
    if (r.command) {{
      const btn = el('button', {{class: 'row-copy'}}, ['Copy']);
      btn.addEventListener('click', () => copyText(r.command, btn));
      cmdCell.appendChild(btn);
    }}
    const tr = el('tr', r.is_best ? {{class: 'best'}} : {{}}, [
      el('td', {{}}, [r.label + (r.is_best ? ' ← pick' : '')]),
      el('td', {{class:'num'}}, [fmt(r.gen_tps,1)]),
      el('td', {{class:'num'}}, [fmt(r.quality,4)]),
      el('td', {{}}, [String(r.ngl)]),
      el('td', {{}}, [String(r.threads)]),
      el('td', {{}}, [r.ctk]),
      el('td', {{}}, [r.ot_pattern || '—']),
      cmdCell,
    ]);
    tbody.appendChild(tr);
  }});
  tbl.appendChild(tbody);
  card.appendChild(tbl);

  s.appendChild(card);
  return s;
}}

const app = document.getElementById('app');
app.appendChild(renderHardware(DATA.hw));
app.appendChild(renderBeforeAfter(DATA.before_after));

const modelHashes = Object.keys(DATA.models).length ? Object.keys(DATA.models)
  : [...new Set(DATA.runs.map(r => r.model_hash))];
if (!modelHashes.length) {{
  const s = document.createElement('section');
  s.innerHTML = '<div class="card empty">No model measured yet — run "python cli.py tune &lt;model&gt;" first.</div>';
  app.appendChild(s);
}} else {{
  modelHashes.forEach(h => {{
    app.appendChild(renderModel(h, DATA.models[h], DATA.sensitivity[h]));
  }});
}}

app.appendChild(renderBestPick(DATA.best));
app.appendChild(renderScatter(DATA.runs));
</script>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        body = render(gather()).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve(port: int = 8765, open_browser: bool = True) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"Optimum dashboard running at {url}  (Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    serve()
