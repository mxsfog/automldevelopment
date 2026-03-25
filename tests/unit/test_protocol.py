"""Тесты для RunnerProtocol и AgentMessage."""

from uaf.runner.protocol import AgentMessage


def test_agent_message_creation() -> None:
    msg = AgentMessage(role="assistant", content="hello")
    assert msg.role == "assistant"
    assert msg.content == "hello"
    assert msg.metadata == {}


def test_agent_message_with_metadata() -> None:
    msg = AgentMessage(role="result", content="done", metadata={"stop_reason": "end_turn"})
    assert msg.metadata["stop_reason"] == "end_turn"


def test_agent_message_roles() -> None:
    for role in ("assistant", "tool_use", "tool_result", "system", "result", "error"):
        msg = AgentMessage(role=role, content="")
        assert msg.role == role
