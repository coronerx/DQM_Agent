# Autonomous Data Quality Monitoring Agent — Dimuon Events

An agent (Claude API) that plays the role of a particle-detector Data
Quality Monitoring (DQM) shift worker: given successive chunks of dimuon
event data, it decides which statistical checks are worth running,
distinguishes real problems from statistical noise, and writes a
plain-language shift report — the same judgment a real ATLAS/CMS shift
crew makes, automated.

Validated end-to-end against **real CMS 2012 collision data**
(`Run2012BC_DoubleMuParked`) and, separately, against live agent runs on
synthetic data designed to exercise specific behaviors. Four real bugs
were found and fixed along the way — not simulated ones. See
[`NOTES.md`](./NOTES.md) for the full chronological debugging log, and
[`LAB_MANUAL.md`](./LAB_MANUAL.md) for background theory and procedure.
This file is the results summary.

## Headline results

Using real dimuon events from the CMS Open Data Portal:

- **0 / 12 false alarms** on unperturbed real data (specificity baseline)
- **Detects a muon momentum-scale miscalibration down to ~1.0–1.5%** at a
  chunk size of 1500 events, quantified via a full detection curve

Using live agent runs (synthetic and real data):

- **Correctly identifies a background-only anomaly** confined outside the
  Z-peak window, after a system-prompt fix closed a real detection blind
  spot (see findings below)
- **Correctly recognizes a recurring problem** across non-adjacent chunks
  and escalates severity appropriately, using only a compact context
  digest — not a full conversation transcript
- **Formally confirms a real, developing drift** (momentum-scale
  miscalibration) using a proper statistical trend test, after a bug in
  the original threshold logic was found and fixed

**[Open the interactive 3D spectrum drift visualization →](./demo/spectrum_waterfall_3d.html)**
*(GitHub won't render this inline — it's a standalone interactive page. Click through.)*

![Detection sensitivity curve](./sensitivity_curve.png)
![Real CMS dimuon spectrum](./dimuon_spectrum_real_data.png)

## Four real bugs found and fixed

This project's most substantive engineering content came from running the
agent for real and finding out where the statistics were actually wrong —
not from getting everything right on the first attempt.

| # | Bug | How it was found | Fix |
|---|---|---|---|
| 1 | `check_occupancy` used a naive ratio threshold, couldn't distinguish "fewer events overall" from "events migrating out of one region" | Manual review after a real run flagged an unfounded "concern" | Replaced with a Poisson test (total count) + binomial proportion test (regional share) |
| 2 | Agent never checked the full mass spectrum, only the Z-peak window — structurally blind to any problem outside it | Deliberately designed adversarial test (background-only anomaly, Z-peak untouched) | System prompt now requires `region='full_spectrum'` every chunk, no exceptions |
| 3 | Chi-square test could return a statistic of ~1,000,000 from a single stray event, and separately had a **~50% false-positive rate** (should be ~5%) — the whole implementation used the wrong statistical test (one-sample GOF instead of two-sample homogeneity) | Live 12-chunk real-data run produced a nonsensical "recurring catastrophic anomaly" the agent spent multiple chunks investigating | Switched to a proper two-sample chi-square test of homogeneity |
| 4 | Drift-detection boolean essentially never fired — even given an obvious, agent-identified +3.4 GeV drift over 12 real chunks | Repeatedly observed across several live runs before being traced to two compounding bugs | Replaced heuristic threshold with a real linear regression significance test |

Each of these has a dedicated regression test (see `tests/`) tied
specifically to the real case that revealed it.

## Structure

```
dqm_agent/
├── dimuon/
│   ├── kinematics.py               relativistic invariant-mass math
│   ├── generate.py                 physically-exact synthetic dimuon events
│   ├── real_data_loader.py         loads real CMS Open Data (NanoAOD)
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
│   ├── dashboard.html                     interactive shift-by-shift dashboard
│   ├── spectrum_waterfall_3d.html         3D visualization of the drift
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

**Free/fast test suite** (no API calls, no network, runs in ~2 seconds):
```bash
PYTHONPATH=. pytest tests/ -v
```

**Live agent run** (costs real API calls — start small):
```bash
export ANTHROPIC_API_KEY=your_key_here
PYTHONPATH=. python3 demo/test_drift_threshold_fix.py
```

**Visualize a run:**
```bash
PYTHONPATH=. python3 demo/export_shift_json.py   # after a run_shift() call
open demo/dashboard.html                          # load the exported JSON
open demo/spectrum_waterfall_3d.html               # interactive 3D view
```

## Why dimuon events, and why real data

Rather than an invented scenario, this project uses real dimuon invariant
mass — the same quantity the official CMS Open Data outreach example
computes, and the same quantity real experiments use to calibrate muon
momentum scale off the known Z boson mass (91.19 GeV). Testing against
real CMS 2012 collision data (not just synthetic data engineered to pass)
is what actually surfaced bugs #3 and #4 above — synthetic-only testing
would very likely have missed both.

## Cost lessons (also real, also worth knowing)

An early 12-chunk live run cost several dollars, traced to `run_shift()`
threading the full raw conversation transcript forward across chunks —
cost grew roughly quadratically with shift length. Fixed by having each
chunk start fresh, with a short plain-text digest of recent verdicts
instead of the full transcript, plus prompt caching on the (large,
per-call-identical) system prompt and tool schemas. See `NOTES.md` for
the full story.

## Status

- Kinematics — validated against hand-computed cases and real CMS data
- Statistics layer — validated on synthetic and real data, all four
  bugs above fixed and regression-tested
- Full agent loop — validated live on both synthetic (blind-spot,
  recurrence, drift-threshold scenarios) and real CMS data
- Cost-aware design — no-history-carryover + prompt caching, confirmed
  working via mocked tests (zero API cost) before real validation
- A "narrative trap" guardrail (don't claim a periodic pattern from
  only 2 occurrences) was added to the system prompt and shows up
  correctly in live transcripts, but has no automated regression test —
  it's a language-quality property, not a computation to assert on
