"""Unit tests for system-mode routing middleware."""

from langchain.agents.middleware.types import ExtendedModelResponse, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage

from deepagents.middleware.system_mode_routing import SystemModeConfig, SystemModeRoutingMiddleware
from tests.unit_tests.chat_model import GenericFakeChatModel


def _build_request(state: dict):
    model = GenericFakeChatModel(messages=iter([AIMessage(content="llm-response")]))
    return ModelRequest(
        model=model,
        messages=[HumanMessage(content="hello")],
        state=state,
    )


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


def test_wrap_model_call_can_bypass_llm_for_deterministic_mode() -> None:
    middleware = SystemModeRoutingMiddleware()

    request = _build_request(
        {
            "execution_mode": "deterministic",
            "routing_confidence": 0.95,
            "_system_mode_decision": {
                "mode": "deterministic",
                "confidence": 0.95,
                "workflow_id": "data_analyzer",
                "assistant_message": "System-1 route selected: data_analyzer",
                "state_update": {"inferred_workflow": "data_analyzer"},
            },
        }
    )

    called = {"handler": False}

    def handler(_: ModelRequest):
        called["handler"] = True
        return ModelResponse(result=[AIMessage(content="should-not-be-used")])

    response = middleware.wrap_model_call(request, handler)

    assert isinstance(response, ExtendedModelResponse)
    assert called["handler"] is False
    assert response.model_response.result[0].content == "System-1 route selected: data_analyzer"
    assert response.command is not None
    assert response.command.update is not None
    assert response.command.update["execution_mode"] == "deterministic"
    assert response.command.update["inferred_workflow"] == "data_analyzer"
