"""AgentSDKRunner — запуск Claude через claude-agent-sdk."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import anyio
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    ResultMessage,
    SystemMessage,
    TextBlock,
)

from uaf import MLFLOW_DEFAULT_URI
from uaf.runner.hooks import HookState, create_deny_list_hook, create_save_reminder_hook
from uaf.runner.protocol import AgentMessage
from uaf.runner.tools import create_uaf_tools

logger = logging.getLogger(__name__)


class AgentSDKRunner:
    """Runner через claude-agent-sdk.

    Заменяет ClaudeCodeRunner (subprocess.Popen) на async SDK API.
    """

    def __init__(
        self,
        session_dir: Path,
        session_id: str,
        claude_model: str = "claude-opus-4-6",
        mlflow_tracking_uri: str = MLFLOW_DEFAULT_URI,
        mlflow_experiment_name: str | None = None,
        mlflow_experiment_id: str | None = None,
        budget_status_file: Path | None = None,
        max_turns: int = 200,
        system_prompt: str | None = None,
        target_metric: str = "metric",
        train_data_path: Path | None = None,
        leakage_high_proba: bool = False,
    ) -> None:
        self.session_dir = session_dir
        self.session_id = session_id
        self.claude_model = claude_model
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_experiment_name = mlflow_experiment_name or f"uaf/{session_id}"
        self.mlflow_experiment_id = mlflow_experiment_id
        self.budget_status_file = budget_status_file or session_dir.parent / "budget_status.json"
        self.max_turns = max_turns
        self.system_prompt = system_prompt
        self.target_metric = target_metric
        self.train_data_path = train_data_path
        self.leakage_high_proba = leakage_high_proba

        self._client: ClaudeSDKClient | None = None
        self._hook_state = HookState()
        self._heartbeat_path = session_dir / ".heartbeat"

    def run_sync(self, prompt: str, session_dir: Path | None = None) -> list[AgentMessage]:
        """Sync wrapper для вызова из _do_executing."""
        return anyio.from_thread.run(self._run_async, prompt, session_dir or self.session_dir)

    async def run_chunk(
        self,
        prompt: str,
        session_dir: Path,
        resume_id: str | None = None,
        chunk_max_turns: int | None = None,
    ) -> list[AgentMessage]:
        """Запустить один chunk агента с возможностью resume.

        Args:
            prompt: промпт для агента.
            session_dir: рабочая директория.
            resume_id: session_id для resume (None = новая сессия).
            chunk_max_turns: max_turns для этого chunk (None = self.max_turns).
        """
        return await self._run_async(
            prompt, session_dir, resume_id, chunk_max_turns
        )

    async def _run_async(
        self,
        prompt: str,
        session_dir: Path,
        resume_id: str | None = None,
        chunk_max_turns: int | None = None,
    ) -> list[AgentMessage]:
        """Запустить агента через ClaudeSDKClient с custom tools."""
        messages: list[AgentMessage] = []
        effective_max_turns = chunk_max_turns or self.max_turns

        uaf_tools_server = create_uaf_tools(
            session_dir=session_dir,
            budget_status_file=self.budget_status_file,
            mlflow_tracking_uri=self.mlflow_tracking_uri,
            experiment_id=self.mlflow_experiment_id,
            target_metric=self.target_metric,
            train_data_path=self.train_data_path,
            leakage_high_proba=self.leakage_high_proba,
        )

        env_vars = {
            "MLFLOW_TRACKING_URI": self.mlflow_tracking_uri,
            "MLFLOW_EXPERIMENT_NAME": self.mlflow_experiment_name,
            "UAF_SESSION_ID": self.session_id,
            "UAF_SESSION_DIR": str(session_dir),
            "UAF_BUDGET_STATUS_FILE": str(self.budget_status_file),
            "UAF_TARGET_METRIC": self.target_metric,
            # Workaround SDK Issue #730: stdin timeout kills hooks/MCP after 60s
            "CLAUDE_CODE_STREAM_CLOSE_TIMEOUT": "3600000",
        }

        # Маппинг алиасов — bundled CLI требует полные model ID
        model_aliases: dict[str, str] = {
            "claude-opus-4": "claude-opus-4-6",
            "opus": "claude-opus-4-6",
            "sonnet": "claude-sonnet-4-6",
            "haiku": "claude-haiku-4-5",
        }
        resolved_model = model_aliases.get(self.claude_model, self.claude_model)

        short_system = (
            "You are an autonomous ML researcher. "
            "Work iteratively: one step at a time, run, analyze, then next. "
            "Never stop until check_budget returns can_stop=true."
        )

        resume_opts: dict[str, Any] = {}
        if resume_id:
            resume_opts["resume"] = resume_id

        options = ClaudeAgentOptions(
            model=resolved_model,
            cwd=str(session_dir),
            system_prompt=short_system,
            max_turns=effective_max_turns,
            permission_mode="acceptEdits",
            **resume_opts,
            allowed_tools=[
                "Bash", "Read", "Edit", "Write", "Glob", "Grep",
                "WebSearch", "WebFetch",
                "mcp__uaf-tools__save_pipeline",
                "mcp__uaf-tools__check_budget",
                "mcp__uaf-tools__get_experiment_memory",
                "mcp__uaf-tools__log_experiment_result",
                "mcp__uaf-tools__check_leakage",
            ],
            mcp_servers={"uaf-tools": uaf_tools_server},
            hooks={
                "PreToolUse": [
                    HookMatcher(
                        matcher="Bash|Write",
                        hooks=[create_deny_list_hook(self._hook_state)],
                    )
                ],
                "PostToolUse": [
                    HookMatcher(
                        matcher=".*",
                        hooks=[
                            create_save_reminder_hook(
                                self._hook_state, self.max_turns
                            )
                        ],
                    )
                ],
            },
            env=env_vars,
        )

        logger.info(
            "[SDK] Starting agent: model=%s, max_turns=%d, cwd=%s",
            self.claude_model,
            self.max_turns,
            session_dir,
        )

        try:
            async with ClaudeSDKClient(options=options) as client:
                self._client = client
                await client.query(prompt)

                async for message in client.receive_response():
                    self._update_heartbeat(message)

                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                messages.append(
                                    AgentMessage(
                                        role="assistant",
                                        content=block.text,
                                        metadata={
                                            "usage": message.usage or {}
                                        },
                                    )
                                )

                    elif isinstance(message, ResultMessage):
                        messages.append(
                            AgentMessage(
                                role="result",
                                content=message.result or "",
                                metadata={
                                    "stop_reason": message.stop_reason,
                                    "session_id": getattr(
                                        message, "session_id", None
                                    ),
                                    "num_turns": getattr(
                                        message, "num_turns", None
                                    ),
                                    "total_cost_usd": getattr(
                                        message, "total_cost_usd", None
                                    ),
                                },
                            )
                        )
                        logger.info(
                            "[SDK] Chunk finished: stop=%s, session=%s",
                            message.stop_reason,
                            getattr(message, "session_id", "?"),
                        )

                    elif isinstance(message, SystemMessage):
                        messages.append(
                            AgentMessage(
                                role="system",
                                content=str(message.data),
                                metadata={"subtype": message.subtype},
                            )
                        )

        except Exception as exc:
            logger.error("[SDK] Agent error: %s", exc)
            messages.append(
                AgentMessage(
                    role="error", content=str(exc),
                    metadata={"exception": type(exc).__name__},
                )
            )

        finally:
            self._client = None
            self._cleanup_heartbeat()

        return messages

    async def stop(self) -> None:
        """Остановить агента."""
        if self._client:
            try:
                await self._client.interrupt()
            except Exception as exc:
                logger.warning("[SDK] Interrupt failed: %s", exc)

    def _update_heartbeat(self, message: Any) -> None:
        """Обновить heartbeat file для BudgetController."""
        try:
            tool_name = ""
            if hasattr(message, "content") and isinstance(message.content, list):
                for block in message.content:
                    if hasattr(block, "name"):
                        tool_name = block.name
                        break

            data = {
                "timestamp": time.time(),
                "tool": tool_name,
                "session_id": self.session_id,
            }
            self._heartbeat_path.write_text(json.dumps(data))
        except Exception:
            pass

    def _cleanup_heartbeat(self) -> None:
        """Удалить heartbeat file после завершения."""
        try:
            if self._heartbeat_path.exists():
                self._heartbeat_path.unlink()
        except Exception:
            pass
