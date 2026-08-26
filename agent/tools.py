"""
tools.py — tool definitions Claude sees (TOOLS) and the dispatcher that
executes them (execute_tool), wired to the dimuon pipeline specifically.

Two design choices here are direct consequences of real findings from
LAB_MANUAL.md Section 5, not arbitrary:
  - check_occupancy exists as its OWN tool because run_distribution_test
    (shape-based) provably cannot detect a pure event-count drop.
  - run_distribution_test takes a `region` parameter so the agent can choose
    the Z-peak window specifically, since that was empirically far more
    sensitive to a real momentum-scale miscalibration than the full
    spectrum (~1-1.5% detection threshold vs. an unusable whole-spectrum
    mean in the synthetic tests).
"""

import numpy as np

from monitors import distribution_shift, drift_tracker, occupancy
from reference import reference_distributions
from agent import data_context

Z_MASS_GEV = 91.1876
Z_WINDOW_GEV = 10.0
MIN_EVENTS_FOR_TEST = 5

TOOLS = [
    {
        "name": "run_distribution_test",
        "description": (
            "Compare the current chunk's dimuon invariant-mass distribution "
            "against the reference, using both a KS test and a chi-square "
            "test. Choose region='z_peak' to restrict the comparison to "
            "events near the Z resonance (91.19 +/- 10 GeV) -- this is far "
            "more sensitive to a momentum-scale miscalibration specifically, "
            "since it excludes the heavy-tailed background continuum that "
            "otherwise dominates the noise. Prefer 'z_peak' when checking "
            "for a calibration-style shift; use 'full_spectrum' only for a "
            "general shape check unrelated to calibration."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "region": {"type": "string", "enum": ["full_spectrum", "z_peak"]},
            },
            "required": ["region"],
        },
    },
    {
        "name": "check_occupancy",
        "description": (
            "Check whether event counts are consistent with the reference, "
            "using the statistically correct test for the question being "
            "asked. region='total' runs a Poisson count test on the "
            "chunk's total event count (a counting-process question: did "
            "fewer/more events arrive overall than expected?). "
            "region='z_peak' runs a binomial proportion test on the SHARE "
            "of events falling in the Z-peak window specifically -- a "
            "DIFFERENT question from total occupancy. A chunk can have a "
            "completely normal total event count while events are still "
            "migrating out of the Z-peak window specifically (or into it) "
            "-- only the region-specific test can catch that pattern; a "
            "raw before/after comparison of the region's event count is "
            "not a substitute for this test, since chunk-to-chunk event "
            "counts fluctuate on their own and an apparent drop is often "
            "not statistically significant. run_distribution_test cannot "
            "detect either of these either way: shape-based tests cannot "
            "see a pure count change if the surviving events still look "
            "normally distributed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "region": {"type": "string", "enum": ["total", "z_peak"]},
            },
            "required": ["region"],
        },
    },
    {
        "name": "apply_multiple_testing_correction",
        "description": (
            "Apply Benjamini-Hochberg FDR correction to a batch of raw "
            "p-values from tests run this chunk (e.g. full_spectrum and "
            "z_peak tests together), before deciding which, if any, are "
            "real enough to escalate. Always call this before escalating "
            "any finding based on a p-value, if more than one test was run "
            "this chunk."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pvalues": {
                    "type": "object",
                    "description": "Mapping from a label to its raw p-value, e.g. {'full_spectrum:ks': 0.4, 'z_peak:ks': 0.02}.",
                    "additionalProperties": {"type": "number"},
                },
                "alpha": {"type": "number", "description": "Significance level, default 0.05."},
            },
            "required": ["pvalues"],
        },
    },
    {
        "name": "check_drift",
        "description": (
            "Record this chunk's value for a tracked quantity (e.g. the "
            "chunk's Z-peak mean mass) and check whether it fits a "
            "developing trend across chunks, rather than being a one-off "
            "fluctuation. Use a consistent label across chunks in the same "
            "shift, e.g. 'z_peak_mean_mass', so the rolling history is "
            "tracked correctly."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "channel": {"type": "string"},
                "value": {"type": "number"},
            },
            "required": ["channel", "value"],
        },
    },
    {
        "name": "escalate_finding",
        "description": (
            "Record a finding in the shift report -- either an anomaly "
            "worth flagging or an explicit note that everything looks "
            "normal. Call this once per chunk to close out the analysis."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "severity": {"type": "string", "enum": ["normal", "watch", "anomaly"]},
                "summary": {"type": "string"},
            },
            "required": ["severity", "summary"],
        },
    },
]


def execute_tool(name: str, tool_input: dict):
    if name == "run_distribution_test":
        region = tool_input["region"]
        chunk_mass = data_context.get_reco_mass()
        reference_mass = reference_distributions.get_reference_mass()

        if region == "z_peak":
            chunk_vals = chunk_mass[np.abs(chunk_mass - Z_MASS_GEV) < Z_WINDOW_GEV]
            ref_vals = reference_mass[np.abs(reference_mass - Z_MASS_GEV) < Z_WINDOW_GEV]
            if len(chunk_vals) < MIN_EVENTS_FOR_TEST or len(ref_vals) < MIN_EVENTS_FOR_TEST:
                return {
                    "region": region,
                    "error": "too few events in the Z-peak window for a reliable test",
                    "chunk_n": int(len(chunk_vals)),
                    "reference_n": int(len(ref_vals)),
                }
        else:
            chunk_vals, ref_vals = chunk_mass, reference_mass

        return {
            "region": region,
            "chunk_n": int(len(chunk_vals)),
            "reference_n": int(len(ref_vals)),
            "ks_test": distribution_shift.ks_test(region, chunk_vals, ref_vals),
            "chi_square_test": distribution_shift.chi_square_test(region, chunk_vals, ref_vals),
        }

    if name == "check_occupancy":
        region = tool_input["region"]

        if region == "total":
            chunk_n = data_context.get_n_events()
            reference_n = reference_distributions.get_reference_n_events()
            result = occupancy.poisson_count_test(chunk_n, reference_n)
            result["region"] = "total"
            return result

        if region == "z_peak":
            chunk_mass = data_context.get_reco_mass()
            reference_mass = reference_distributions.get_reference_mass()

            chunk_n_total = len(chunk_mass)
            chunk_n_z = int(np.sum(np.abs(chunk_mass - Z_MASS_GEV) < Z_WINDOW_GEV))
            reference_n_total = len(reference_mass)
            reference_n_z = int(np.sum(np.abs(reference_mass - Z_MASS_GEV) < Z_WINDOW_GEV))
            reference_p = reference_n_z / reference_n_total if reference_n_total else float("nan")

            result = occupancy.binomial_proportion_test(chunk_n_z, chunk_n_total, reference_p)
            result["region"] = "z_peak"
            return result

        raise ValueError(f"Unknown occupancy region: {region}")

    if name == "apply_multiple_testing_correction":
        return distribution_shift.benjamini_hochberg(
            tool_input["pvalues"], alpha=tool_input.get("alpha", 0.05)
        )

    if name == "check_drift":
        return drift_tracker.record_and_check(tool_input["channel"], tool_input["value"])

    if name == "escalate_finding":
        return {"logged": True, **tool_input}

    raise ValueError(f"Unknown tool: {name}")
