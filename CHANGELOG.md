# Changelog

Every commit on this repo, explained in plain language — what changed and why it
mattered, oldest first.

---

## `c55f7f8` / `ce81ebb` / `e6bfac0` — Starting the repo

The GitHub repo was created with just a one-line `README.md`. On top of that we
laid out the empty file structure for the whole project — one file per job
(`config.py` for settings, `hardware.py` for reading the GPU, `sensitivity.py`
for the core idea, and so on), most of them empty placeholders to be filled in
later. `config.py` and `hardware.py` already had working code from the start:
`config.py` holds every path the project needs (where llama.cpp lives, where
models live, where the database goes), and `hardware.py` reads how much GPU
memory is actually free right now using `pynvml`. A `.gitignore` was added so
the big stuff — the 1.7GB of llama.cpp binaries, the multi-GB model files, the
Python virtual environment — never gets pushed to GitHub, only the source code.
Then the repo's own starter README was merged in so both histories lined up.

## `658e124` — Measuring what each part of the model is worth

This is the first real piece of logic: `sensitivity.py`. It answers the
question "if I only have room for some of this model on my GPU, which parts
should I keep there?" It does this by testing one tensor group at a time (the
attention tensors, the feed-forward tensors, etc.) — put everything on the CPU
as a slow "floor" measurement, then put just one group back on the GPU and see
how much speed comes back. Divide that speed gain by the group's size in
megabytes, and you get a "value per MB" for every part of the model. This
turned out to be the whole idea behind the project: small tensor groups often
buy a lot of speed, while huge ones buy comparatively little.

## `e14eef1` — Turning those values into an actual plan

`packer.py` takes the value-per-MB numbers from sensitivity and decides what
actually goes on the GPU versus the CPU. It's a simple, well-known strategy:
sort everything by value, keep adding to the GPU until you run out of room. It
also doesn't stop at one plan — it generates a handful of nearby variations
(one setting safer, one setting greedier, different thread counts) so there
are several real options to test, not just one guess.

## `e653b44` — Actually testing the candidates, and deciding

`evaluate.py` takes each candidate plan from the packer and runs it for real —
once for speed, once for quality — and records both numbers together.
`tune.py` is the single function that ties the whole pipeline together end to
end: check the hardware, look inside the model, measure sensitivity, build
candidates, evaluate them all, and print the answer.

## `122c2ef` — Throwing away the losers

`frontier.py` looks at every candidate that got evaluated and throws out any
that another candidate beats on both speed *and* quality at once — there's no
reason anyone would ever pick a candidate like that. What's left is handed a
quality floor (set by the user), and the fastest one that clears that floor
wins.

## `e9a1643` — Filling in everything quality-, storage-, and safety-related

This was the big one — seven files at once:
- **`speed.py`** runs `llama-bench` and turns its output into tokens-per-second.
- **`reference.py`** builds a "before squeezing" baseline from a
  higher-precision model, so later comparisons have something honest to
  compare against.
- **`quality.py`** runs `llama-perplexity`'s KL-divergence comparison against
  that baseline and turns it into one quality score.
- **`db.py`** saves every result to a local SQLite database, so nothing is
  ever lost or has to be re-measured.
- **`cache.py`** is a small helper to avoid recomputing anything expensive
  twice.
- **`report.py`** turns results into readable text and a ready-to-run
  `llama-server` command.
- **`sanity.py`** is the trust layer — checks like "do repeated runs of the
  same settings give the same speed" and "does a deliberately broken model
  actually score badly" — designed to catch the case where the numbers *look*
  fine but are quietly meaningless.

Real captured output from `llama-bench` and `llama-perplexity` was saved as
test fixtures, and a `pytest.ini` was added so the test suite could find the
project's own code.

## `244db29` — Connecting the pieces, and a real command line

Up to this point, `db.py`, `cache.py`, and `report.py` existed but `tune.py`
never actually called them — every result vanished the moment the program
exited. This commit wired `tune.py` up to actually save each run to the
database and print through the proper report formatter. It also added
`cli.py`, the first real command-line entry point (`probe`, `inspect`,
`sensitivity`, `tune`, `report`) instead of only being able to run each file
on its own.

## `404f3ec` — Fixing a check that would have always failed, plus real test data

While testing the "does the scorer notice a badly broken model" sanity check
against an actual ruined model, it turned out the original threshold was
wrong: a badly quantized model still guesses the *same* next word as a good
model surprisingly often (common words like "the" and "and" are easy
regardless), so its score never drops anywhere near the old absolute cutoff.
The check was rewritten to compare the ruined model against a known-good one
*relatively*, instead of expecting an unrealistic absolute floor. Real
`llama-bench` and `llama-perplexity` output was also captured and saved as
test fixtures, replacing made-up example text.

## `59e7780` — Four real bugs, found only by actually running the program

Every module had been written and unit-tested, but never run start-to-finish
against the real GPU and real model files. Doing that for the first time
immediately turned up four bugs that no amount of unit testing alone would
have caught:
- A newer version of numpy refused to convert certain values to plain Python
  numbers the old way, crashing `model_info.py` the moment it read a model
  file.
- One file referred to a result field by a slightly wrong name
  (`vram_mb` instead of `vram_used_mb`), crashing every single evaluation.
- Loading saved results for the "models" table crashed because that table
  doesn't have a column named `id` like the others do.
- VRAM-used measurements were quietly always reading as zero, because the
  measurement was taken *after* the benchmark program had already closed and
  released its memory — documented rather than silently left broken, since
  fixing it properly needs a different measurement approach.

## `ced9aae` — A real webpage showing the results

Added `optimum start`, a command that launches a local dashboard in your
browser. It shows your GPU and RAM, the model's tensor-group breakdown as a
bar chart, and every recorded run plotted as speed vs. quality — with the
settings that survived the frontier check in one color and the ones that got
beaten in another. Everything is built from plain HTML, CSS and JavaScript
with no internet connection or outside library needed, and it re-reads the
database fresh every time the page loads.

## `68f167a` — A true "no tuning at all" baseline to compare against

Added `optimum default`, which runs llama.cpp with absolutely none of its
speed-related settings changed — no `-ngl`, no `-t`, no `-ctk`, nothing — so
it behaves exactly as it would for someone who just downloaded llama.cpp and
ran it. That result gets saved next to the tuned candidates, so the dashboard
and reports can show, with real numbers, exactly how much tuning actually
bought you. On the model tested here: **84.3 tokens/sec untouched versus
102.5 tokens/sec tuned, at identical answer quality** — about a fifth more
speed, for free, just from picking better settings.
