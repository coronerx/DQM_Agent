"""
tests/test_loop_wiring.py — validates agent/loop.py's orchestration logic
using FAKE Claude responses (unittest.mock), so wiring bugs (tool_result
formatting, message bookkeeping, loop termination) get caught for free,
before spending a single real API call finding them.

Run with: PYTHONPATH=. pytest tests/test_loop_wiring.py -v
"""

from unittest.mock import patch, MagicMock

import numpy as np

from agent.loop import analyze_chunk
from agent import data_context
from reference import reference_distributions
from dimuon.generate import generate_chunk


def _fake_tool_use_block(name, input_dict, block_id="tool_1"):
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.input = input_dict
    block.id = block_id
    return block


def _fake_text_block(text):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _setup_reference_and_chunk():
    reference = generate_chunk(500, np.random.default_rng(0), momentum_scale=1.0)
    reference_distributions.build_reference(reference)
    chunk = generate_chunk(500, np.random.default_rng(1), momentum_scale=1.0)
    data_context.set_chunk("test_chunk", chunk)
    return chunk


def test_loop_handles_single_tool_call_then_final_text():
    _setup_reference_and_chunk()

    tool_call = _fake_tool_use_block("check_occupancy", {"region": "total"})
    first_response = MagicMock(stop_reason="tool_use", content=[tool_call])

    final_text = _fake_text_block("Occupancy looks normal, nothing to escalate.")
    second_response = MagicMock(stop_reason="end_turn", content=[final_text])

    with patch("agent.loop.client.messages.create",
               side_effect=[first_response, second_response]) as mock_create:
        result = analyze_chunk({"chunk_id": "test_chunk", "n_events": 500})

    assert "normal" in result["report"]
    assert mock_create.call_count == 2
    # Confirm the tool_result made it back into the message history in the
    # shape the API expects (a user-role message with a tool_result block).
    tool_result_messages = [
        m for m in result["messages"]
        if isinstance(m.get("content"), list)
        and any(isinstance(c, dict) and c.get("type") == "tool_result" for c in m["content"])
    ]
    assert len(tool_result_messages) == 1
    assert tool_result_messages[0]["content"][0]["tool_use_id"] == "tool_1"


def test_loop_handles_multiple_tool_calls_in_one_turn():
    """Claude can request more than one tool in a single response (e.g. both
    run_distribution_test regions) -- confirm every tool_use block gets
    executed and every result gets a matching tool_result."""
    _setup_reference_and_chunk()

    call_a = _fake_tool_use_block("run_distribution_test", {"region": "full_spectrum"}, "id_a")
    call_b = _fake_tool_use_block("run_distribution_test", {"region": "z_peak"}, "id_b")
    first_response = MagicMock(stop_reason="tool_use", content=[call_a, call_b])

    second_response = MagicMock(stop_reason="end_turn",
                                  content=[_fake_text_block("Both regions look consistent.")])

    with patch("agent.loop.client.messages.create",
               side_effect=[first_response, second_response]):
        result = analyze_chunk({"chunk_id": "test_chunk", "n_events": 500})

    tool_result_msg = result["messages"][-2]  # the user-role tool_result message
    assert len(tool_result_msg["content"]) == 2
    ids = {block["tool_use_id"] for block in tool_result_msg["content"]}
    assert ids == {"id_a", "id_b"}


def test_loop_terminates_immediately_with_no_tool_calls():
    """If Claude responds with only text (decides nothing needs checking),
    the loop should end after one API call, not hang or error."""
    response = MagicMock(stop_reason="end_turn",
                          content=[_fake_text_block("Nothing unusual, skipping checks.")])

    with patch("agent.loop.client.messages.create", side_effect=[response]) as mock_create:
        result = analyze_chunk({"chunk_id": "quiet_chunk", "n_events": 500})

    assert mock_create.call_count == 1
    assert "skipping" in result["report"]


def test_check_occupancy_tool_actually_executes_through_the_loop():
    """A slightly stronger version of the first test -- confirm the tool's
    REAL return value (not just that the call happened) reaches the second
    API call's message history correctly."""
    chunk = _setup_reference_and_chunk()

    tool_call = _fake_tool_use_block("check_occupancy", {"region": "total"})
    first_response = MagicMock(stop_reason="tool_use", content=[tool_call])
    second_response = MagicMock(stop_reason="end_turn",
                                  content=[_fake_text_block("ok")])

    with patch("agent.loop.client.messages.create",
               side_effect=[first_response, second_response]):
        result = analyze_chunk({"chunk_id": "test_chunk", "n_events": len(chunk["reco_mass"])})

    tool_result_content = result["messages"][-2]["content"][0]["content"]
    # The real check_occupancy output should be a stringified dict containing
    # these keys -- confirms execute_tool actually ran the real Poisson test,
    # not a stub.
    assert "p_value" in tool_result_content
    assert "expected_n" in tool_result_content


def test_tool_calls_are_captured_as_structured_data():
    """analyze_chunk should return a 'tool_calls' list with REAL Python
    objects (not stringified), so downstream consumers (e.g. a dashboard)
    can read p-values directly without re-parsing text out of the
    transcript."""
    _setup_reference_and_chunk()

    call_a = _fake_tool_use_block("run_distribution_test", {"region": "z_peak"}, "id_a")
    first_response = MagicMock(stop_reason="tool_use", content=[call_a])
    second_response = MagicMock(stop_reason="end_turn",
                                  content=[_fake_text_block("done")])

    with patch("agent.loop.client.messages.create",
               side_effect=[first_response, second_response]):
        result = analyze_chunk({"chunk_id": "test_chunk"})

    assert "tool_calls" in result
    assert len(result["tool_calls"]) == 1
    entry = result["tool_calls"][0]
    assert entry["tool"] == "run_distribution_test"
    assert entry["input"] == {"region": "z_peak"}
    # This is the real, unstringified dict from execute_tool -- confirms a
    # consumer can do entry["result"]["ks_test"]["p_value"] directly.
    assert "ks_test" in entry["result"]
    assert isinstance(entry["result"]["ks_test"]["p_value"], float)


def test_analyze_chunk_starts_fresh_each_call_not_carrying_history():
    """Cost regression test: analyze_chunk should NOT accumulate a growing
    transcript across separate calls -- each call's message list should
    start from just this chunk's own summary, not include anything from
    a 'previous' call. This is what keeps a long shift's cost from growing
    quadratically (a real ~$6 / 12-chunk bill was traced to this)."""
    _setup_reference_and_chunk()

    response = MagicMock(stop_reason="end_turn", content=[_fake_text_block("ok")])
    with patch("agent.loop.client.messages.create", side_effect=[response]):
        result = analyze_chunk({"chunk_id": "chunk_1"})
    # Only this chunk's own summary message plus the assistant reply --
    # no leftover history from anywhere else.
    assert len(result["messages"]) == 2

    with patch("agent.loop.client.messages.create", side_effect=[response]):
        result2 = analyze_chunk({"chunk_id": "chunk_2"})
    # A second, independent call should ALSO start fresh at 2 messages,
    # not 4 -- confirms no hidden shared state carries over between calls.
    assert len(result2["messages"]) == 2


def test_context_digest_is_short_text_not_raw_history():
    """run_shift's digest mechanism should pass a short plain-text summary
    of past verdicts, not the full raw tool-call transcript -- confirms
    the actual cost-saving replacement is working, not just that history
    carryover was removed."""
    _setup_reference_and_chunk()

    tool_call = _fake_tool_use_block("escalate_finding",
                                       {"severity": "normal", "summary": "All clear."})
    first_response = MagicMock(stop_reason="tool_use", content=[tool_call])
    second_response = MagicMock(stop_reason="end_turn", content=[_fake_text_block("ok")])

    captured_summary_texts = []
    original_create_args = []

    def fake_create(**kwargs):
        original_create_args.append(kwargs)
        if len(original_create_args) == 1:
            return first_response
        return second_response

    with patch("agent.loop.client.messages.create", side_effect=fake_create):
        analyze_chunk({"chunk_id": "chunk_2"}, context_digest="chunk_1: normal -- All clear.")

    first_call_messages = original_create_args[0]["messages"]
    sent_text = first_call_messages[0]["content"]
    assert "chunk_1: normal" in sent_text
    # The digest is short -- nowhere near the size of a full raw transcript
    # (tool call JSON, tool results, etc.) that history carryover produced.
    assert len(sent_text) < 500
