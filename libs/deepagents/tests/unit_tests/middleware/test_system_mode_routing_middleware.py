"""Unit tests for system-mode routing middleware."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from deepagents.middleware.system_mode_routing import SystemModeConfig, SystemModeRoutingMiddleware


def test_before_agent_threshold_routing_deterministic() -> None:
    middleware = SystemModeRoutingMiddleware(config=SystemModeConfig(deterministic_threshold=0.85, clarification_threshold=0.50))

    result = middleware.before_agent(
        {"confidence_level": 0.91},
        runtime=None,  # type: ignore[arg-type]
        config={},  # type: ignore[arg-type]
    )

    assert result is not None
    assert result["execution_mode"] == "deterministic"
    assert result["routing_confidence"] == 0.91


def test_before_agent_escalates_clarification_to_emergent_on_round_limit() -> None:
    middleware = SystemModeRoutingMiddleware(config=SystemModeConfig(deterministic_threshold=0.85, clarification_threshold=0.50, max_clarification_rounds=3))

    result = middleware.before_agent(
        {
            "confidence_level": 0.70,
            "clarification_round": 3,
            "max_clarification_rounds": 3,
        },
        runtime=None,  # type: ignore[arg-type]
        config={},  # type: ignore[arg-type]
    )

    assert result is not None
    assert result["execution_mode"] == "emergent"
    assert result["routing_reason"] == "clarification_round_limit_reached"


def test_before_model_forced_tool_calls_bypasses_model_node() -> None:
    """forced_tool_calls: before_model injects AIMessage and sets jump_to='tools'."""
    middleware = SystemModeRoutingMiddleware()

    state = {
        "messages": [HumanMessage(content="analyze costs")],
        "execution_mode": "deterministic",
        "routing_confidence": 0.95,
        "_system_mode_decision": {
            "mode": "deterministic",
            "confidence": 0.95,
            "workflow_id": "cost_analyzer",
            "reason": "fast_track",
            "forced_tool_calls": [
                {"id": "tc1", "name": "task", "args": {"name": "cost_analyzer"}, "type": "tool_call"}
            ],
            "state_update": {"inferred_workflow": "cost_analyzer"},
        },
    }

    result = middleware.before_model(state, runtime=None, config={})  # type: ignore[arg-type]

    assert result is not None
    assert result["jump_to"] == "tools"
    assert result["execution_mode"] == "deterministic"
    assert result["routing_workflow_id"] == "cost_analyzer"
    assert result["inferred_workflow"] == "cost_analyzer"
    assert len(result["messages"]) == 1
    msg = result["messages"][0]
    assert isinstance(msg, AIMessage)
    assert msg.tool_calls[0]["name"] == "task"


def test_before_model_assistant_message_bypasses_model_and_tools() -> None:
    """assistant_message: before_model injects AIMessage and sets jump_to='end'."""
    middleware = SystemModeRoutingMiddleware()

    state = {
        "messages": [HumanMessage(content="hello")],
        "execution_mode": "deterministic",
        "_system_mode_decision": {
            "mode": "deterministic",
            "confidence": 0.95,
            "workflow_id": "data_analyzer",
            "assistant_message": "System-1 route selected: data_analyzer",
            "state_update": {"inferred_workflow": "data_analyzer"},
        },
    }

    result = middleware.before_model(state, runtime=None, config={})  # type: ignore[arg-type]

    assert result is not None
    assert result["jump_to"] == "end"
    assert result["execution_mode"] == "deterministic"
    assert result["inferred_workflow"] == "data_analyzer"
    assert len(result["messages"]) == 1
    msg = result["messages"][0]
    assert isinstance(msg, AIMessage)
    assert msg.content == "System-1 route selected: data_analyzer"
    assert not msg.tool_calls


def test_before_model_passthrough_for_non_deterministic_mode() -> None:
    """Emergent/clarification mode: before_model returns None (falls through to model)."""
    middleware = SystemModeRoutingMiddleware()

    for mode in ("emergent", "clarification", None):
        state = {
            "messages": [HumanMessage(content="help")],
            "execution_mode": mode,
            "_system_mode_decision": {"mode": mode},
        }
        result = middleware.before_model(state, runtime=None, config={})  # type: ignore[arg-type]
        assert result is None, f"Expected None for mode={mode!r}"


def test_before_model_passthrough_when_no_bypass_fields() -> None:
    """Deterministic mode but no forced_tool_calls or assistant_message: no bypass."""
    middleware = SystemModeRoutingMiddleware()

    state = {
        "messages": [HumanMessage(content="hello")],
        "execution_mode": "deterministic",
        "_system_mode_decision": {
            "mode": "deterministic",
            "confidence": 0.95,
            "workflow_id": "some_workflow",
            # no forced_tool_calls, no assistant_message
        },
    }

    result = middleware.before_model(state, runtime=None, config={})  # type: ignore[arg-type]
    assert result is None


@pytest.mark.asyncio
async def test_abefore_model_delegates_to_sync() -> None:
    """abefore_model produces the same result as before_model."""
    middleware = SystemModeRoutingMiddleware()

    state = {
        "messages": [HumanMessage(content="analyze costs")],
        "execution_mode": "deterministic",
        "_system_mode_decision": {
            "mode": "deterministic",
            "confidence": 0.9,
            "forced_tool_calls": [
                {"id": "tc1", "name": "task", "args": {"name": "cost_analyzer"}, "type": "tool_call"}
            ],
        },
    }

    sync_result = middleware.before_model(state, runtime=None, config={})  # type: ignore[arg-type]
    async_result = await middleware.abefore_model(state, runtime=None, config={})  # type: ignore[arg-type]

    assert sync_result == async_result


def test_before_model_has_can_jump_to_metadata() -> None:
    """Verify @hook_config sets __can_jump_to__ so the graph creates conditional edges."""
    method = SystemModeRoutingMiddleware.before_model
    assert hasattr(method, "__can_jump_to__")
    assert set(method.__can_jump_to__) == {"tools", "end"}
