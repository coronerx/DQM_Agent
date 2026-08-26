"""
demo/export_shift_json.py — converts run_shift()'s output into a clean JSON
file for the dashboard (demo/dashboard.html) to load.

Pulls p-values, severities, and mean-mass values out of each chunk's
tool_calls (real Python objects, captured directly in loop.py) rather than
re-parsing the prose report -- so this stays correct even if the report
text's phrasing changes.
"""

import json


def _extract_pvalues(tool_calls: list[dict]) -> dict:
    """Flatten every test's p-value into one dict, keyed like
    'full_spectrum:ks', matching the same label scheme the agent itself
    uses when calling apply_multiple_testing_correction."""
    pvalues = {}
    for call in tool_calls:
        if call["tool"] == "run_distribution_test":
            region = call["result"].get("region")
            if "ks_test" in call["result"]:
                pvalues[f"{region}:ks"] = call["result"]["ks_test"]["p_value"]
            if "chi_square_test" in call["result"]:
                pvalues[f"{region}:chi2"] = call["result"]["chi_square_test"]["p_value"]
        elif call["tool"] == "check_occupancy":
            region = call["result"].get("region")
            pvalues[f"occupancy:{region}"] = call["result"].get("p_value")
    return pvalues


def _extract_drift(tool_calls: list[dict]) -> dict:
    """Pull check_drift results (slope, trend p-value, sustained-drift
    flag) keyed by channel, e.g. 'z_peak_mean_mass'."""
    drift = {}
    for call in tool_calls:
        if call["tool"] == "check_drift":
            result = call["result"]
            channel = result.get("channel")
            if channel and result.get("trend_detectable"):
                drift[channel] = {
                    "slope": result.get("linear_trend_slope"),
                    "trend_p_value": result.get("trend_p_value"),
                    "looks_like_sustained_drift": result.get("looks_like_sustained_drift"),
                }
    return drift


def _extract_severity_and_summary(tool_calls: list[dict]):
    for call in tool_calls:
        if call["tool"] == "escalate_finding":
            return call["input"].get("severity"), call["input"].get("summary")
    return None, None


def _extract_mean_masses(chunk_summary: dict) -> dict:
    return {
        "full_spectrum": chunk_summary.get("mean_mass_full_spectrum"),
        "z_peak": chunk_summary.get("mean_mass_near_z_peak"),
    }


def _extract_steps(tool_calls: list[dict]) -> list[dict]:
    """Preserve the REAL order the agent actually called tools in, with a
    short plain-language caption per step -- used by demo/agent_story.html
    to replay the actual decision sequence rather than a guessed one.
    Older exports won't have this field; the story page falls back to a
    representative fixed order in that case."""
    captions = {
        "run_distribution_test": lambda inp, res: (
            f"Checking the {'overall' if inp.get('region') == 'full_spectrum' else 'Z-peak-region'} "
            f"shape of this batch against the reference"
        ),
        "check_occupancy": lambda inp, res: (
            "Checking the total event count" if inp.get("region") == "total"
            else "Checking the share of events landing near the Z mass"
        ),
        "apply_multiple_testing_correction": lambda inp, res: (
            "Adjusting the significance bar since several checks were run this chunk"
        ),
        "check_drift": lambda inp, res: (
            "Comparing against recent chunks for a developing trend, not just this one"
        ),
        "escalate_finding": lambda inp, res: "Writing the final verdict",
    }
    steps = []
    for call in tool_calls:
        caption_fn = captions.get(call["tool"])
        caption = caption_fn(call["input"], call["result"]) if caption_fn else call["tool"]
        steps.append({
            "tool": call["tool"],
            "input": call["input"],
            "result": call["result"],
            "caption": caption,
        })
    return steps


def export_reports_to_json(reports: list[dict], path: str) -> None:
    chunks_out = []
    for r in reports:
        severity, summary = _extract_severity_and_summary(r["tool_calls"])
        chunks_out.append({
            "chunk_index": r["chunk_index"],
            "chunk_id": r["chunk_id"],
            "severity": severity or "unknown",
            "summary": summary or r.get("report", ""),
            "pvalues": _extract_pvalues(r["tool_calls"]),
            "drift": _extract_drift(r["tool_calls"]),
            "mean_mass": _extract_mean_masses(r["chunk_summary"]),
            "n_events": r["chunk_summary"].get("n_events"),
            "n_events_near_z_peak": r["chunk_summary"].get("n_events_near_z_peak"),
            "steps": _extract_steps(r["tool_calls"]),
        })

    with open(path, "w") as f:
        json.dump({"chunks": chunks_out}, f, indent=2)

    print(f"Exported {len(chunks_out)} chunks to {path}")
