"""
loop.py — core agentic loop for the dimuon DQM agent.

For each incoming chunk of dimuon events, Claude:
  1. Reads a lightweight summary (event count, mean mass, Z-peak stats) --
     never the raw per-event array.
  2. Decides which check(s) are worth running given that summary.
  3. Interprets results, applying multiple-testing correction before
     trusting any p-value, and checking drift history before treating a
     single flag as real.
  4. Produces a shift-report entry via escalate_finding.
"""

import anthropic

from agent.tools import TOOLS, execute_tool

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-6"

# Prompt caching: build this ONCE, not on every call. TOOLS is
# byte-identical across every API call in an entire shift -- marking the
# last tool with cache_control tells the API to cache everything up to and
# including it (system prompt + full tool list), so later calls that still
# start with this exact prefix get a large discount on those input tokens
# instead of paying full price every single round.
_TOOLS_WITH_CACHE = TOOLS[:-1] + [{**TOOLS[-1], "cache_control": {"type": "ephemeral"}}] if TOOLS else TOOLS

SYSTEM_PROMPT = """You are an automated Data Quality Monitoring (DQM) shifter \
for the CMS dimuon channel, reviewing chunks of dimuon event data the way a \
human shift-crew member would.

For each chunk you receive:
- Call run_distribution_test with region='full_spectrum' at least once on \
  every chunk, no exceptions. The Z-peak window only covers events near \
  91 GeV -- it is structurally blind to anomalies confined to the \
  background continuum (e.g. a localized excess or artifact elsewhere in \
  the spectrum). Checking only the Z-peak region, even if it looks clean, \
  tells you nothing about the rest of the spectrum. Run full_spectrum \
  regardless of what the Z-peak check shows.
- In addition, call run_distribution_test with region='z_peak' when you are \
  checking for a possible momentum-scale miscalibration specifically -- \
  that window is far more sensitive to that kind of shift than the full \
  spectrum, because it excludes the noisy background continuum. Both \
  regions serve different purposes; running one does not substitute for \
  the other.
- Call check_occupancy with region='total' every chunk to check the total \
  event count via a proper Poisson test. Also call check_occupancy with \
  region='z_peak' -- this runs a binomial proportion test on the SHARE of \
  events falling in the Z-peak window, a genuinely different question from \
  total occupancy (a chunk can have normal total occupancy while events \
  are migrating into or out of the Z-peak window specifically). Do NOT \
  judge whether a region's event count "looks low" by eyeballing the raw \
  number against a previous chunk -- chunk-to-chunk counts fluctuate on \
  their own, and an apparent drop is frequently not statistically \
  significant. Always run the actual test and read its p-value before \
  describing a count change as a concern.
- run_distribution_test and check_occupancy answer DIFFERENT questions too. \
  A shape-based test (KS/chi-square) cannot detect a chunk that has lost a \
  large fraction of its events but whose surviving events still look \
  normally distributed -- always run check_occupancy separately, not only \
  when a distribution test flags something.
- You will typically run several tests per chunk (full_spectrum, z_peak \
  distribution tests, and both occupancy checks). Always apply \
  apply_multiple_testing_correction to the full batch of p-values collected \
  this chunk before treating any of them as evidence -- never act on a raw, \
  uncorrected p-value.
- When a result survives correction, check its drift history with \
  check_drift before deciding it's a real, developing problem rather than a \
  one-off fluctuation. Use a consistent label (e.g. 'z_peak_mean_mass' or \
  'full_spectrum_mean_mass') so the rolling trend is tracked correctly \
  across chunks.
- Do not describe a periodic, alternating, or cyclic pattern (e.g. "every \
  other chunk," "an even-chunk cadence") unless it has actually recurred \
  at least 3 times. Two occurrences, however suggestive, is not a pattern \
  -- it is two data points, and could easily be coincidental. If you \
  notice a recurrence at only 2 occurrences, say plainly that it's too \
  early to call it a pattern and state the sample size, rather than \
  naming a cadence or trend that sounds more established than the \
  evidence supports.
- When you escalate, state the concrete reason in plain language -- which \
  check flagged it, what the deviation was, whether it looks like a \
  one-off or a developing trend. Do not just report a flag/pass verdict.
- If nothing is worth escalating, say so briefly. A quiet chunk is a \
  normal and useful outcome, not something to pad with unnecessary detail.

Always call escalate_finding once per chunk to close out your analysis.
"""


def analyze_chunk(chunk_summary: dict, context_digest: str | None = None) -> dict:
    """Run the agent loop on a single chunk. Assumes agent.data_context has
    already been set (via set_chunk) with this chunk's raw reco_mass array
    before this is called.

    context_digest: optional short plain-text summary of recent chunks'
    verdicts (NOT the raw transcript) -- gives the agent enough continuity
    to reference "last chunk" without resending the full prior history's
    tool-call JSON on every single call."""
    messages = []
    summary_text = f"New chunk summary:\n{chunk_summary}"
    if context_digest:
        summary_text = f"{context_digest}\n\n{summary_text}"
    print(f"  >> sending: {summary_text}", flush=True)
    messages.append({"role": "user", "content": summary_text})

    # Captures each tool call as a real Python object (not the stringified
    # version sent to the API), so downstream consumers -- like a dashboard
    # -- can read p-values and severities directly without re-parsing text.
    tool_calls: list[dict] = []

    round_num = 0
    while True:
        round_num += 1
        print(f"    [round {round_num}] calling Claude...", flush=True)
        response = client.messages.create(
            model=MODEL,
            max_tokens=1500,
            system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            tools=_TOOLS_WITH_CACHE,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            final_text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            return {"report": final_text, "messages": messages, "tool_calls": tool_calls}

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"    [round {round_num}] tool call: {block.name}({block.input})", flush=True)
            result = execute_tool(block.name, block.input)
            print(f"    [round {round_num}] -> {result}", flush=True)
            tool_calls.append({"tool": block.name, "input": block.input, "result": result})
            tool_results.append({
                "type": "tool_result", "tool_use_id": block.id, "content": str(result),
            })
        messages.append({"role": "user", "content": tool_results})


def run_shift(chunks: list[tuple[str, dict, dict]]) -> list[dict]:
    """
    chunks: list of (chunk_id, chunk_data, chunk_summary) tuples, where
    chunk_data has a "reco_mass" key (matches dimuon.generate.generate_chunk
    and dimuon.real_data_loader.chunk_events output shape).

    Each chunk starts with a FRESH message history, not the full transcript
    of every prior chunk -- carrying the whole raw history forward made
    cost grow roughly quadratically with shift length (chunk 12 was
    resending chunks 1-11's entire tool-call transcript on every API call).
    Cross-chunk trend continuity doesn't need that: drift_tracker already
    persists rolling state independently of the conversation, and a short
    plain-text digest of recent verdicts (below) gives the agent enough
    narrative context to reference "last chunk" without re-sending
    thousands of tokens of old tool call JSON.
    """
    from agent import data_context

    recent_verdicts: list[str] = []  # short digest, not the raw transcript
    reports = []
    for i, (chunk_id, chunk_data, chunk_summary) in enumerate(chunks):
        data_context.set_chunk(chunk_id, chunk_data)

        digest = None
        if recent_verdicts:
            digest = "Recent chunk verdicts (for context, not re-verified):\n" + "\n".join(recent_verdicts[-3:])

        result = analyze_chunk(chunk_summary, context_digest=digest)
        reports.append({
            "chunk_index": i, "chunk_id": chunk_id, "chunk_summary": chunk_summary, **result,
        })

        severity, summary = None, None
        for call in result["tool_calls"]:
            if call["tool"] == "escalate_finding":
                severity = call["input"].get("severity")
                summary = call["input"].get("summary", "")
        short_summary = (summary[:150] + "...") if summary and len(summary) > 150 else summary
        recent_verdicts.append(f"{chunk_id}: {severity} -- {short_summary}")

    return reports
