# Autonomous Data Quality Monitoring Agent — Dimuon Events

An agent (Claude API) that plays the role of a particle-detector Data
Quality Monitoring (DQM) shift worker: given successive chunks of dimuon
event data, it decides which statistical checks are worth running,
distinguishes real problems from statistical noise, and writes a
plain-language shift report — the same judgment a real ATLAS/CMS shift
crew makes, automated.

Validated end-to-end against **real CMS 2012 collision data**
(`Run2012BC_DoubleMuParked`). Four real statistical bugs were found and
fixed along the way, and — running the fully-fixed pipeline on real,
completely unmodified data — the agent found and correctly characterized
what looks like a genuine, reproducible composition shift in the real
dataset. See [`NOTES.md`](./NOTES.md) for the full chronological
debugging log, and [`LAB_MANUAL.md`](./LAB_MANUAL.md) for background
theory and procedure. This file is the results summary.

## Headline results

**Statistical validation, on real dimuon events from the CMS Open Data
Portal:**
- 0/12 false alarms on unperturbed real data (specificity baseline)
- Detects a muon momentum-scale miscalibration down to ~1.0-1.5% at a
  chunk size of 1500 events, quantified via a full detection curve

**Live-agent behavioral validation:**
- Correctly identifies a background-only anomaly outside the Z-peak
  window, after a system-prompt fix closed a real detection blind spot
- Correctly recognizes a recurring problem across non-adjacent chunks and
  escalates severity appropriately, using a compact context digest — not
  a full conversation transcript
- Formally confirms a real, developing drift using a proper statistical
  trend test, after a bug in the original threshold logic was found and
  fixed

**A real finding, not just a validated capability:** run twice,
independently, on real unmodified CMS data with zero synthetic
modification, the agent flagged 6/10 and then 9/10 chunks — both driven
by the same specific signal (Z-peak event-share depletion). This was
ruled out as a statistical artifact via a 300-run Monte Carlo stress test
(0/300 simulated null runs produced 6+ flagged chunks under correctly-
calibrated statistics) and independently corroborated by an offline
numerical scan of the raw file, which found real, reproducible elevated
regions at specific positions in the dataset. See "The real-data finding"
below for the full story, including where the investigation honestly hit
its limit.

![3D spectrum drift visualization](./spectrum_waterfall_3d.html)
*(open directly in a browser — interactive, drag to rotate)*

![Detection sensitivity curve](./sensitivity_curve.png)
![Real CMS dimuon spectrum](./dimuon_spectrum_real_data.png)

## The real-data finding

This is the most substantive result in the project, so it gets its own
section rather than a bullet point.

**Setup:** `run_real_data_no_injection.py` runs real CMS data through the
live agent with zero synthetic modification of any kind — pure specificity
testing with the actual, current toolkit (unlike an earlier, KS-only
check from before several of the bug fixes below existed).

**First run** (chunks from the start of the file): 6 of 10 chunks flagged
watch/anomaly, all via the same test (Z-peak occupancy, binomial
proportion), clustered in a temporally coherent block (chunks 5-9), then
resolving by chunk 10. Not scattered noise — a specific, recurring signal.

**Ruling out a bug before trusting it:** the occupancy test was
stress-tested the same way an earlier chi-square bug had been caught —
500 trials under a true null gave a 6.4% false-positive rate (correctly
calibrated). A full Monte Carlo of the actual per-chunk decision procedure
(6 correlated tests + Benjamini-Hochberg correction, 300 simulated
10-chunk runs under a true null) produced **zero** runs with 6+ flagged
chunks. The result cannot be explained by chance or a residual bug.

**Replication, on an independent, non-overlapping slice further into the
file:** 9 of 10 chunks flagged, even more extreme (p-values to 10⁻²¹),
with no recovery — a stable new composition, not a drift. A closer look
suggested the *reference* chunk (not the 9 "anomalous" ones) was the
actual outlier: 9 samples agreeing with each other and 1 disagreeing is
more parsimoniously explained by an atypical reference than by 9
independent chunks drifting the same way.

**Independent offline confirmation:** a from-scratch numerical scan of
Z-peak fraction across 500,000 raw file entries — built, debugged (the
first version had a real x-axis labeling bug, caught and fixed), and run
entirely separately from the live agent's judgment — found real,
multi-window elevated regions at specific raw-entry positions, matching
both live-agent runs' reference points almost exactly.

**Where it honestly stops:** tried to trace the physical cause via CMS's
standard `run`/`luminosityBlock` provenance branches. This specific
reduced educational file only contains 6 branches total (muon kinematics
and charge) — no provenance information at all. The investigation
concludes with strong, replicated, stress-tested statistical evidence of
a real effect in the data, without a confirmed physical mechanism. This
mirrors what a real DQM shift worker does: detect and characterize
correctly, then hand off to an expert with the right context — not
necessarily diagnose root cause personally.

## Four real bugs found and fixed

| # | Bug | How it was found | Fix |
|---|---|---|---|
| 1 | `check_occupancy` used a naive ratio threshold, couldn't distinguish "fewer events overall" from "events migrating out of one region" | A real run flagged an unfounded "concern"; turned out not significant when actually tested | Poisson test (total count) + binomial proportion test (regional share) |
| 2 | Agent never checked the full mass spectrum, only the Z-peak window — structurally blind to any problem outside it | Deliberately designed adversarial test (background-only anomaly, Z-peak untouched) | System prompt requires `region='full_spectrum'` every chunk, no exceptions |
| 3 | Chi-square test could return a statistic of ~1,000,000 from a single stray event, and separately had a **~50% false-positive rate** (should be ~5%) — using a one-sample GOF test where a two-sample homogeneity test was needed | A live 12-chunk real-data run produced a nonsensical "recurring catastrophic anomaly" | Two-sample chi-square test of homogeneity |
| 4 | Drift-detection boolean essentially never fired — even given an obvious, agent-identified +3.4 GeV drift over 12 real chunks | Observed repeatedly across live runs, traced to two compounding bugs (a hidden history-length gate, and a threshold that measured itself against a self-inflated baseline) | Real linear regression significance test (`scipy.stats.linregress`) |

Each has a dedicated regression test tied to the specific real case that
revealed it.

## Structure

```
dqm_agent/
├── dimuon/
│   ├── kinematics.py               relativistic invariant-mass math
│   ├── generate.py                 physically-exact synthetic dimuon events
│   ├── real_data_loader.py         loads real CMS Open Data, with entry_start
│   │                                 offset + raw-entry-index tracking
│   ├── real_data_false_positive_test.py   specificity check on real data
│   ├── real_data_sensitivity_test.py      single-case detection on real data
│   └── real_data_sensitivity_curve.py     full detection-threshold curve
├── monitors/
│   ├── distribution_shift.py       KS test, two-sample chi-square, Benjamini-Hochberg
│   ├── occupancy.py                Poisson (total) + binomial (regional) tests
│   └── drift_tracker.py            linear-regression trend significance test
├── agent/
│   ├── data_context.py             holds current chunk's raw data
│   ├── tools.py                    tool schemas + dispatcher
│   └── loop.py                     Claude tool-use loop, prompt caching, cost-aware history
├── demo/
│   ├── test_full_spectrum_blind_spot.py   adversarial test for bug #2
│   ├── test_recurring_anomaly.py          recurrence-recognition test
│   ├── test_drift_threshold_fix.py        live confirmation of bug #4's fix
│   ├── run_real_data_no_injection.py      the real-data specificity experiment
│   ├── run_real_data_no_injection_offset.py   the replication check
│   ├── scan_z_peak_fraction.py            offline confirmation scan (free, no API)
│   ├── inspect_run_boundaries.py          provenance-branch check (free, no API)
│   ├── dashboard.html                     interactive shift-by-shift dashboard
│   ├── spectrum_waterfall_3d.html         3D visualization of the drift
│   ├── agent_story.html                   plain-language workflow + decision replay
│   └── export_shift_json.py               converts a run into dashboard-ready JSON
├── tests/                          fast, free, deterministic (no API calls)
├── LAB_MANUAL.md                   background, theory, full procedure
├── NOTES.md                        chronological debugging log
└── README.md                       this file
```

## Quick start

```bash
pip install -r requirements.txt
```

**Free/fast test suite** (no API calls, no network, ~2 seconds):
```bash
PYTHONPATH=. pytest tests/ -v
```

**Live agent run** (costs real API calls — start small):
```bash
export ANTHROPIC_API_KEY=your_key_here
PYTHONPATH=. python3 demo/test_drift_threshold_fix.py
```

**Free offline diagnostics** (no API key needed, just the downloaded
`.root` file):
```bash
PYTHONPATH=. python3 demo/scan_z_peak_fraction.py Run2012BC_DoubleMuParked_Muons.root
PYTHONPATH=. python3 demo/inspect_run_boundaries.py Run2012BC_DoubleMuParked_Muons.root
```

**Visualize a run:**
```bash
open demo/dashboard.html          # load an exported JSON
open demo/spectrum_waterfall_3d.html
open demo/agent_story.html        # non-expert-friendly walkthrough
```

## Why dimuon events, and why real data

Rather than an invented scenario, this project uses real dimuon invariant
mass — the same quantity the official CMS Open Data outreach example
computes, and the same quantity real experiments use to calibrate muon
momentum scale off the known Z boson mass (91.19 GeV). Testing against
real CMS 2012 collision data, not just synthetic data engineered to pass,
is what actually surfaced bugs #3 and #4 above, and is what led to the
real-data finding described above — synthetic-only testing would very
likely have caught neither.

## Cost lessons (also real, also worth knowing)

An early 12-chunk live run cost several dollars, traced to `run_shift()`
threading the full raw conversation transcript forward across chunks —
cost grew roughly quadratically with shift length. Fixed by having each
chunk start fresh, with a short plain-text digest of recent verdicts
instead of the full transcript, plus prompt caching on the system prompt
and tool schemas. All subsequent live runs, including the two 10-chunk
real-data experiments above, benefited from this fix. See `NOTES.md` for
the full story.

## Status

- Kinematics — validated against hand-computed cases and real CMS data
- Statistics layer — validated on synthetic and real data, all four bugs
  above fixed and regression-tested
- Full agent loop — validated live on synthetic scenarios (blind-spot,
  recurrence, drift-threshold) and on real, unmodified CMS data
- Cost-aware design — no-history-carryover + prompt caching, confirmed
  via mocked tests (zero API cost) before real validation
- Real-data composition-shift finding — replicated, stress-tested,
  independently confirmed offline; physical mechanism not traceable with
  this specific reduced dataset (no provenance branches available)
- A "narrative trap" guardrail (don't claim a periodic pattern from only
  2 occurrences) is in the system prompt and shows up correctly in live
  transcripts, but has no automated regression test — a language-quality
  property, not a computation to assert on
