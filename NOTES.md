# Project notes — running log

Kept as I went, not reconstructed afterward. This is the honest version of
the debugging process; `README.md` has the polished summary.

## Kinematics + generator

- Built `kinematics.py` (relativistic invariant mass from pT/eta/phi) and
  `generate.py` (physically exact synthetic dimuon events: sample target
  mass, construct in CM frame, apply a random longitudinal boost — boosts
  preserve invariant mass exactly, so this is real relativistic kinematics,
  not a curve-fit approximation).
- Test that actually mattered: generate events at a target mass, recompute
  the mass from the generated 4-vectors, confirm it round-trips to
  floating-point precision. If this ever fails after touching
  `generate.py`, nothing downstream can be trusted.

## First real-data validation

- Downloaded real CMS 2012 `Run2012BC_DoubleMuParked` data via
  `cernopendata-client` (my first hand-guessed download URL was wrong —
  hit a stub page, 10KB instead of 2GB — the client resolves the real,
  current URL properly, don't hand-guess CERN portal links).
- Ran `kinematics.py` on real muon momenta. Mean mass 33.34 GeV, matching
  the CMS tutorial's own documented note that "the bump at 30 GeV is not
  a resonance but a trigger effect" — a real, specific, previously
  unknown-to-me artifact of this exact dataset, correctly reproduced.
- Full spectrum plot: clean J/psi, Upsilon, the ~30 GeV trigger bump, and
  Z peak all showed up in the right places.

## Specificity/sensitivity validation (standalone scripts, pre-agent)

- 0/12 false alarms on real, unperturbed data (chunk_size=1500, KS test
  only — chi-square fix didn't exist yet).
- Injected a synthetic 3% momentum miscalibration into one real chunk —
  caught cleanly, only that chunk flagged.
- Full sensitivity curve: detects down to ~1.0-1.5% miscalibration at
  n=1500 events/chunk.

## Bug #1 — occupancy check was checking the wrong thing

- Original `check_occupancy` was a single ratio threshold. A real agent
  run described a Z-peak event-count drop (189 vs 202) as a "concern" —
  turned out NOT statistically significant (p=0.28) when actually tested.
- Fixed with Poisson (total count) + binomial proportion (regional share)
  tests. Regression test uses the exact 189/202 case that started this.

## Bug #2 — full-spectrum blind spot

- The agent, as originally prompted, only ever chose `region='z_peak'` —
  never `'full_spectrum'`. Built a deliberately adversarial test: inject
  an anomaly ONLY in the background continuum, leave the Z-peak window
  byte-identical to normal. Confirmed the agent missed it completely.
- Fixed by making `full_spectrum` mandatory every chunk in the system
  prompt. Reran the same adversarial scenario — correctly caught, with
  the right physical reasoning about a background-only effect.

## Cost lesson — the $6 run

- First 12-chunk live run cost ~$6. Traced it: `run_shift()` was
  threading the FULL raw message transcript forward across chunks, so
  chunk 12 was resending chunks 1-11's entire tool-call history on every
  single API call. Cost grew roughly quadratically with chunk count.
- Fixed: each chunk starts fresh; a short plain-text digest of recent
  verdicts replaces the full transcript. Added prompt caching on the
  system prompt + tools. Confirmed via mocked tests (free) that history
  really resets before spending anything real to confirm no regression.

## Bug #3 — the chi-square test was fundamentally wrong

- That same $6 run produced a "recurring catastrophic full-spectrum
  anomaly" (chi-square p=0.0, statistic ~1-2 million) the agent spent
  several chunks treating as a second, independent failure mode.
- Reproduced directly: a SINGLE stray event in a mass bin where the
  reference histogram had zero events blew the statistic to ~1,000,000
  from one data point (old code floored expected counts at 1e-6 instead
  of excluding empty-reference bins).
- Fixed that, then while writing a regression test, found something
  bigger: even on completely matching distributions, false-positive rate
  was ~50% instead of the nominal 5%. Root cause: using
  `scipy.stats.chisquare` (one-sample GOF, treats "expected" as fixed and
  noiseless) when the reference is itself a noisy finite sample. Replaced
  with a proper two-sample chi-square test of homogeneity. Verified: false
  positive rate ~3.5-5%, single-event case stays sane, real shifts still
  caught decisively.
- Most of the $6 run's "second anomaly" narrative was very likely this
  artifact the whole time, not real physics. Reconstructed that run's
  JSON export with the caveat explicitly annotated rather than pretending
  the original interpretation was right.

## Bug #4 — drift threshold essentially never fired

- Confirmed on the real 12-chunk run: Z-peak mean mass rose 90.478 ->
  93.845 GeV, an obvious drift the agent itself narrated correctly using
  raw numbers — but `looks_like_sustained_drift` stayed False across all
  12 chunks.
- Two compounding bugs: (1) a gating condition required 10+ chunks of
  history before the boolean could be True at all, regardless of trend
  strength; (2) even past that gate, the threshold compared slope against
  the raw standard deviation of a window that INCLUDED the drifting
  values — a real trend inflates its own yardstick, self-defeating
  exactly when there's real drift to catch.
- Fixed with `scipy.stats.linregress` — tests whether the slope is
  significant given the RESIDUAL scatter around the fitted trend line.
  Verified against the real 12-chunk trajectory: now fires True starting
  at chunk 5, the earliest statistically possible point. False-positive
  rate on flat noise: 5.0% across 200 trials.
- Built a small, cheap 6-chunk synthetic scenario (offline-validated
  first to find an injection strength that reliably crosses the
  threshold) to confirm live. Fired exactly at chunk 5; the agent's own
  language changed too — started citing "drift p=4.2e-4" as formal
  evidence instead of describing raw numbers informally.

## Recurring-anomaly test — first attempt failed by design, not by bug

- Wanted to test whether the agent recognizes a RECURRING problem
  (non-adjacent chunks) and escalates severity accordingly. First
  attempt (default `anomaly_fraction=0.15`): chunk_2 came back NORMAL —
  its injected anomaly, by chance (different random seed), wasn't strong
  enough to survive BH correction, so the test never got to ask what it
  was designed to ask.
- Checked offline across several seeds/strengths: 0.15 only triggered
  ~25% of the time by luck; 0.20 triggered reliably across every seed
  tested. Reran with 0.20 — worked exactly as intended: chunk_2 WATCH,
  chunk_4 explicitly named as a recurrence, severity escalated to
  ANOMALY, chunks 5-6 correctly de-escalated once the pattern didn't
  continue.

## Narrative-trap guardrail

- Chunk_6 of that recurring-anomaly run confidently described an
  "every-other-chunk cadence" from exactly 2 occurrences with only 2
  clean chunks around them. Hedged, but more pattern than 2 data points
  supports.
- Added an explicit system-prompt rule: don't name a periodic/cyclic
  pattern from fewer than 3 occurrences. Confirmed in a later run
  (drift-threshold scenario, chunks 3-4 had a similar 2-occurrence
  situation): agent explicitly said "with only two occurrences it is too
  early to call it a systematic trend." No automated test for this one —
  it's a language-quality property, not a numeric assertion.

## Visualization pass

- Built `dashboard.html` (chunk-by-chunk severity timeline + p-value/mean
  mass charts, loads a JSON export), `spectrum_waterfall_3d.html` (3D
  Plotly surface of the mass spectrum shifting across chunks, seeded from
  the same synthetic data as the drift-threshold-fix run), and
  `agent_story.html` (plain-language workflow diagram + step-by-step
  replay of a real run's actual recorded decisions, for a non-HEP
  audience).
- Added a source toggle (real vs. synthetic sample data) to
  `agent_story.html` — discovered along the way that real and synthetic
  chunks share the same `chunk_1..N` naming, so a single shared histogram
  file would have silently shown the wrong spectrum shape depending on
  which source was selected. Split into `chunk_histograms_real.json` and
  `chunk_histograms_sample.json`, matched per source explicitly.
- Deployed to GitHub Pages. Learned the hard way not to commit the 2GB
  `.root` file (added `.gitignore`).

## The partial-trailing-chunk bug

- `run_real_data_drift_shift.py`'s 12-chunk run included a partial final
  chunk (827 of an expected 1500 events) because 13 chunks were actually
  needed (1 reference + 12 test) but only 18,827 real events were loaded
  — a shortfall the user caught by checking my arithmetic (18,000 < 18,827
  doesn't mean 13 chunks fit; needed 19,500). The agent correctly but
  misleadingly read the partial chunk as a catastrophic -17sigma
  occupancy collapse — a real reaction to a fake artifact, not a genuine
  detector event.
- Fixed: drop any trailing chunk smaller than `chunk_size`, and raised
  `max_events` to 80,000 so enough full chunks remain after dropping.
  Validated the drop-logic offline with synthetic fixture data before
  trusting it.

## The real-data-only experiment (no injection)

- Built `run_real_data_no_injection.py`: real data, ZERO synthetic
  modification, through the FULL current pipeline (KS + fixed two-sample
  chi-square + both occupancy tests + fixed drift test) — a genuinely new
  experiment nothing before had run (earlier false-positive checks used
  an older, partial toolkit; every live-agent real-data run since had a
  synthetic drift injected on top).
- First run (`entry_start=0`, 10 chunks): **6/10 flagged**, all driven by
  the same specific signal (Z-peak occupancy depletion), in a temporally
  coherent block (chunks 5-9), resolving by chunk 10. Not scattered noise
  — a real, specific, recurring signal.
- Before trusting this as "real": stress-tested `binomial_proportion_test`
  the same way chi-square had been (500 trials under a true null) — 6.4%
  false-positive rate, correctly calibrated, no hidden bug. Then ran a
  Monte Carlo of the FULL per-chunk decision procedure (6 correlated
  tests + BH correction, 300 simulated 10-chunk runs under a true null):
  **0/300** produced 6+ flagged chunks. 6/10 cannot be chance or a
  statistical bug given correctly-calibrated tests.

## Replication check — even more extreme, and a subtler read

- Added `entry_start` support to `load_events()`/`build_scenario()` to
  pull an independent, non-overlapping slice (entries starting at
  200,000, well past the ~80,000 the first run covered).
- Result: **9/10 flagged**, far more extreme (p-values to 10^-21) than
  the first run, with NO recovery — stable at a new composition, not
  drifting.
- Key realization: chunk_1 (this run's own reference) sat at 27.5% Z-peak
  share; chunks 2-10 all clustered tightly around 17-20%. When 9 out of
  10 samples agree with each other and only the reference disagrees, the
  more parsimonious read isn't "9 consecutive real chunks independently
  drifted the same way" — it's that the REFERENCE landed in an unusual
  pocket of the data. `DoubleMuParked` is a CMS "data parking" stream
  specifically designed to vary composition across different run
  periods, so this isn't a stretch explanation.

## Scanning for where the composition actually shifts

- Built `scan_z_peak_fraction.py`: offline, free, no API cost, maps
  Z-peak fraction across a wide stretch of the file.
- First version had a real bug: bucketed by POST-SELECTION event count
  and labeled the x-axis "raw entry number" — since only ~37.65% of raw
  entries pass the opposite-charge-dimuon cut, this silently distorted
  the position mapping. The scan's actual endpoint (~185,000 on the
  mislabeled axis) matched `500,000 x 0.3765` almost exactly, confirming
  the bug rather than a data problem. The reference marker line for
  `entry_start=200,000` was consequently in the wrong place too.
- Fixed properly: added `return_raw_entry_index` to `load_events()`,
  tracking the TRUE raw entry index through both selection masks
  (validated the sequential-masking logic offline with plain numpy
  before trusting it on real data). Rebucketed by fixed raw-entry
  windows instead of post-selection count.
- Corrected scan (`z_peak_fraction_scan_v2.png`): at raw entry 200,000,
  fraction = 0.275, n=1496 — matches the second live-agent run's reported
  reference almost exactly, confirming the fix. Found real, multi-window
  elevated regions: entries 196,000-208,000 (three consecutive windows,
  0.265/0.275/0.263, then a sharp drop back to baseline at 208,000), and
  a larger one spanning ~324,000-420,000 (multiple peaks, including the
  scan's global maximum of 0.309). Real dataset mean/std: 0.195/0.031.
- This is an offline, purely numerical check — independent of the live
  agent's judgment entirely — and it corroborates both live-agent
  findings with real, quantifiable structure.

## Trying to trace the physical cause — hit a real, honest wall

- Checked whether this file retains CMS's standard `run` /
  `luminosityBlock` / `event` provenance branches (would let us identify
  the actual CMS run number responsible and look up real trigger/
  luminosity conditions).
- Result: this file has only 6 branches total — `nMuon`, `Muon_pt`,
  `Muon_eta`, `Muon_phi`, `Muon_mass`, `Muon_charge`. No provenance
  information whatsoever. This specific "reduced NanoAOD" educational
  file cannot be traced to a specific run this way — not a bug, just a
  genuine limit of what this particular dataset contains.
- Decided to stop here rather than chase the full non-reduced AOD dataset
  (a much heavier lift — GBs-to-TBs scale, full CMSSW environment) for
  what it would add. The finding stands on its statistical merits
  (replication + stress-tested false-positive rate + independent offline
  confirmation) even without a confirmed physical mechanism. This mirrors
  what a real DQM shift worker actually does: detect and characterize
  correctly, then hand off to the right expert with the right context —
  not necessarily diagnose root cause personally.

## Open, not yet chased

- The confirmed real-data composition-shift regions have never been run
  back through the LIVE AGENT with `entry_start` set to land the
  reference squarely in a "normal" baseline region and test chunks
  spanning a known transition (e.g. entries 190,000-210,000) — would let
  the agent's own drift/occupancy tools characterize the transition shape
  directly (sharp step vs. gradual) rather than inferring it from an
  external offline scan.
- Whether the narrative-trap guardrail holds up under more pressure —
  e.g. 4-5 borderline recurrences instead of a clean 2-vs-3 case.
- A scenario combining two simultaneous real problems in one chunk.
