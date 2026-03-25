"""Протокол и типы для runner-абстракции UAF."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass
class AgentMessage:
    """Единый формат сообщения от любого runner backend."""

    role: str  # "assistant" | "tool_use" | "tool_result" | "system" | "result"
    content: str
    metadata: dict[str, object] = field(default_factory=dict)


@runtime_checkable
class RunnerProtocol(Protocol):
    """Интерфейс runner для UAF session controller.

    Две реализации: AgentSDKRunner (claude-agent-sdk) и SubprocessRunner (fallback).
    """

    async def run(self, prompt: str, session_dir: Path) -> AsyncIterator[AgentMessage]:
        """Запустить агента и итерировать по сообщениям."""
        ...

    async def stop(self) -> None:
        """Остановить агента."""
        ...
