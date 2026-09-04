"""Claude Code CLI 适配器（车道三：子进程 RPC，外部黑盒 Agent 第一个真实现）。

链路（对齐 dev01 实测环境）::

    本适配器 ──tsh ssh──► dev01: claude -p --output-format json [--resume <sid>]
                              │  cwd = tau2-bench（.claude/settings.local.json 生效）
                              ▼
                    anthropic_proxy(:8022) ──► vLLM(:8021, Qwen3.8-27B)

工具桥（文本协议，借鉴 tau2-bench 的 ClaudeCodeAgent 方案）：环境工具不打进 CC 的
MCP，而是在附加系统提示里给工具目录，约定 CC 需要域操作时输出 JSON 块::

    ```json
    {"tool_call": {"name": "...", "arguments": {...}}}
    ```

适配器解析该块 -> 在本地执行（经 runner 录制包装）-> 结果以 [TOOL RESULT] 文本
经 --resume 回传。CC 自己的循环（上下文管理、内置工具、重试）原样保留——那正是被测对象。

模型访问不需要 Anthropic 适配：8022 代理已把 Anthropic 格式转成 OpenAI 格式打 vLLM；
本框架自身仍只走 OpenAI SDK（LLMClient 接缝预留未来 Provider）。
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ahedd.adapters import register_adapter
from ahedd.adapters.base import AgentResult, TaskInput
from ahedd.env.base import default_diff
from ahedd.env.tools import ToolDefinition, ToolRegistry
from ahedd.trace.schema import Usage

if TYPE_CHECKING:
    from ahedd.trace.schema import TrajectoryRecorder

TOOL_CALL_RE = re.compile(r"```json\s*(\{.*?tool_call.*?\})\s*```", re.DOTALL)

_BRIEF_TEMPLATE = (
    "# DOMAIN TOOLS (system-provided; NOT your built-in tools)\n"
    "{catalog}\n\n"
    "# TOOL CALL PROTOCOL\n"
    "When you need to perform a domain action, output ONLY a single JSON block:\n"
    '```json\n{{"tool_call": {{"name": "<tool_name>", "arguments": {{...}}}}}}\n```\n'
    "The system will execute it and send you the result. One tool call per turn.\n"
    "Do NOT use your built-in bash/file tools to simulate domain operations.\n"
    "When responding to the user, output plain text WITHOUT any JSON block."
)


class ClaudeCodeAdapter:
    name = "claude-code"

    def __init__(
        self,
        *,
        workdir: str,
        ssh_target: str | None = None,
        node_bin: str = "~/.nvm/versions/node/v22.22.0/bin",
        timeout: int = 300,
        max_turns: int = 20,
        system_prompt: str | None = None,
        tool_mode: str = "text",
        mcp_server_name: str = "ahedd",
        mcp_url: str = "http://127.0.0.1:8023/mcp",
        events_file: str | None = None,
        events_ssh_target: str | None = None,
    ) -> None:
        """:param ssh_target: 形如 "research@nm-zhipu-a800-develop01"；None 则本地子进程执行。
        :param workdir: claude 的工作目录（.claude/settings.local.json 生效范围）。
        :param node_bin: claude 所在的 nvm bin 目录（远端 PATH 注入）。
        :param tool_mode: "text" = 文本协议桥（兜底，任何 CLI 可用）；
            "mcp" = 原生工具调用：经 --mcp-config 连接我们的环境 MCP server（推荐，测真实 CC 工作方式）。
        :param mcp_url / mcp_server_name / events_file: MCP 模式参数（events_file 为 server 事件日志，
            适配器读回合并进统一轨迹并提供环境终态 diff）。
        :param events_ssh_target: 事件日志在远端（MCP server 与 CC 同机运行）时，
            经 tsh ssh cat 拉取；None 则读本地文件。
        """
        self.workdir = workdir
        self.ssh_target = ssh_target
        self.node_bin = node_bin
        self.timeout = timeout
        self.max_turns = max_turns
        self.system_prompt = system_prompt
        self.tool_mode = tool_mode
        self.mcp_server_name = mcp_server_name
        self.mcp_url = mcp_url
        self.events_file = events_file
        self.events_ssh_target = events_ssh_target

    # ---- 子进程执行（阻塞，由 run 内 to_thread 包装） ----

    def _call_claude(
        self,
        prompt: str,
        session_id: str | None,
        append_system: str | None,
        extra_flags: list[str] | None = None,
    ) -> dict[str, Any]:
        flags = list(extra_flags or [])
        if self.ssh_target:
            script = (
                f"export PATH={self.node_bin}:$PATH && cd {shlex.quote(self.workdir)} "
                "&& claude -p --output-format json"
            )
            if session_id:
                script += f" --resume {shlex.quote(session_id)}"
            if append_system:
                b64 = base64.b64encode(append_system.encode("utf-8")).decode()
                script += f' --append-system-prompt "$(echo {b64} | base64 -d)"'
            script += "".join(f" {shlex.quote(f)}" for f in flags)
            cmd: list[str] = ["tsh", "ssh", self.ssh_target, script]
            env = None
        else:
            cmd = ["claude", "-p", "--output-format", "json"]
            if session_id:
                cmd += ["--resume", session_id]
            if append_system:
                cmd += ["--append-system-prompt", append_system]
            cmd += flags
            env = dict(os.environ)
            env["PATH"] = os.path.expanduser(self.node_bin) + os.pathsep + env.get("PATH", "")

        try:
            proc = subprocess.run(
                cmd, input=prompt, capture_output=True, text=True,
                timeout=self.timeout, env=env, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"claude CLI timed out after {self.timeout}s") from exc
        if proc.returncode != 0:
            raise RuntimeError(f"claude CLI failed ({proc.returncode}): {proc.stderr[-500:]}")
        try:
            data = json.loads(proc.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError) as exc:
            raise RuntimeError(f"claude output not JSON: {proc.stdout[:300]}") from exc
        if data.get("is_error"):
            raise RuntimeError(f"claude reported error: {str(data.get('result'))[:300]}")
        return data

    # ---- 适配器主循环 ----

    async def run(
        self,
        task: TaskInput,
        tools: list[ToolDefinition],
        recorder: TrajectoryRecorder | None = None,
    ) -> AgentResult:
        if self.tool_mode == "mcp":
            return await self._run_mcp(task, recorder)
        return await self._run_text_protocol(task, tools, recorder)

    async def _run_mcp(self, task: TaskInput, recorder: TrajectoryRecorder | None) -> AgentResult:
        """MCP 模式：CC 原生工具环路。一次 -p 调用内 CC 自行完成全部 MCP 工具调用，
        工具执行发生在我们的 MCP server 进程（事件日志 -> 合并入统一轨迹 + 环境终态 diff）。"""
        mcp_config = json.dumps(
            {"mcpServers": {self.mcp_server_name: {"type": "http", "url": self.mcp_url}}}
        )
        flags = [
            "--mcp-config", mcp_config,
            "--strict-mcp-config",
            # server 级授权：mcp__<server> 允许其全部工具（中间通配符 mcp__x_* 不受支持）
            "--allowedTools", f"mcp__{self.mcp_server_name}",
        ]
        data = await asyncio.to_thread(self._call_claude, task.instruction, None, None, flags)

        raw_usage = data.get("usage") or {}
        usage = Usage(
            input_tokens=int(raw_usage.get("input_tokens", 0) or 0),
            output_tokens=int(raw_usage.get("output_tokens", 0) or 0),
            cost_usd=float(data.get("total_cost_usd", 0.0) or 0.0),
        )
        result_text = (data.get("result") or "").strip()

        env_diff: dict[str, Any] | None = None
        if self.events_file and recorder:
            from ahedd.mcp.server import read_server_events

            events_path = self._fetch_events()
            if events_path:
                server_steps, initial_state, final_state = read_server_events(events_path)
                for step in server_steps:  # 工具事件入轨（位于最终 assistant 之前）
                    step.index = len(recorder.trajectory.steps)
                    recorder.trajectory.steps.append(step)
                if initial_state or final_state:
                    env_diff = default_diff(initial_state, final_state)
            recorder.note("assistant", content=result_text, stop_reason="stop", usage=usage)

        return AgentResult(
            final_message=result_text,
            stop_reason="stop",
            usage_total={
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cost_usd": usage.cost_usd,
            },
            env_diff=env_diff,
        )

    def _fetch_events(self) -> str | None:
        """事件日志路径：本地直接用；远端经 tsh ssh cat 拉到临时文件。"""
        if not self.events_ssh_target:
            return self.events_file
        import tempfile

        proc = subprocess.run(
            ["tsh", "ssh", self.events_ssh_target, f"cat {shlex.quote(self.events_file)}"],
            capture_output=True, text=True, timeout=60, check=False,
        )
        if proc.returncode != 0:
            return None
        tmp_path = Path(tempfile.gettempdir()) / f"ahedd_events_{os.getpid()}.jsonl"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(proc.stdout)
        return str(tmp_path)

    async def _run_text_protocol(
        self,
        task: TaskInput,
        tools: list[ToolDefinition],
        recorder: TrajectoryRecorder | None,
    ) -> AgentResult:
        registry = ToolRegistry(tools)
        brief = _build_brief(registry, self.system_prompt)

        session_id: str | None = None
        text = task.instruction
        total = Usage()
        final_message = ""

        for turn in range(self.max_turns):
            append_system = brief if session_id is None else None
            data = await asyncio.to_thread(self._call_claude, text, session_id, append_system)
            session_id = data.get("session_id") or session_id

            raw_usage = data.get("usage") or {}
            usage = Usage(
                input_tokens=int(raw_usage.get("input_tokens", 0) or 0),
                output_tokens=int(raw_usage.get("output_tokens", 0) or 0),
                cost_usd=float(data.get("total_cost_usd", 0.0) or 0.0),
            )
            total = Usage(
                input_tokens=total.input_tokens + usage.input_tokens,
                output_tokens=total.output_tokens + usage.output_tokens,
                cost_usd=total.cost_usd + usage.cost_usd,
            )
            result_text = (data.get("result") or "").strip()
            if recorder:
                recorder.note("assistant", content=result_text, usage=usage)

            call = _extract_tool_call(result_text)
            if call is None:
                final_message = result_text
                return AgentResult(
                    final_message=final_message,
                    stop_reason="stop",
                    usage_total={
                        "input_tokens": total.input_tokens,
                        "output_tokens": total.output_tokens,
                        "cost_usd": total.cost_usd,
                    },
                )

            name, args = call
            tool = registry.get(name)
            if tool is None:
                if recorder:
                    recorder.note("error", content=f"unknown tool: {name}", tool_name=name, error_kind="agent")
                result: Any = {"ok": False, "error": f"unknown tool: {name}"}
            else:
                try:
                    result = await tool.func(**args)  # runner 已包装：调用即入轨
                except TypeError as exc:
                    if recorder:
                        recorder.note("error", content=f"TypeError: {exc}", tool_name=name, error_kind="agent")
                    result = {"ok": False, "error": f"bad arguments: {exc}"}
            text = f"[TOOL RESULT] {json.dumps(result, ensure_ascii=False, default=str)}\nContinue with the protocol."

        if recorder:
            recorder.note("error", content="max_turns exceeded", stop_reason="max_steps", error_kind="agent")
        return AgentResult(
            final_message="",
            stop_reason="max_steps",
            usage_total={
                "input_tokens": total.input_tokens,
                "output_tokens": total.output_tokens,
                "cost_usd": total.cost_usd,
            },
        )


def _build_brief(registry: ToolRegistry, extra_policy: str | None) -> str:
    catalog = []
    for tool in registry:
        props = tool.parameters.get("properties", {})
        args = ", ".join(f"{k} ({v.get('type', 'any')})" for k, v in props.items())
        catalog.append(f"- {tool.name}({args})\n  {tool.description[:200]}")
    brief = _BRIEF_TEMPLATE.format(catalog="\n".join(catalog))
    if extra_policy:
        brief = f"# DOMAIN POLICY\n{extra_policy}\n\n{brief}"
    return brief


def _extract_tool_call(text: str) -> tuple[str, dict[str, Any]] | None:
    """从结果文本解析 ```json {"tool_call": ...}``` 块（容忍裸 JSON）。"""
    candidates = [m.group(1) for m in TOOL_CALL_RE.finditer(text)]
    start = text.find('{"tool_call"')
    if start != -1:
        end = text.find("}", text.find("arguments", start))
        if end != -1:
            candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            call = payload["tool_call"]
            return str(call["name"]), dict(call.get("arguments", {}) or {})
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return None


@register_adapter("claude-code")
def _factory() -> ClaudeCodeAdapter:
    raise TypeError("claude-code 需要执行参数：请直接构造 ClaudeCodeAdapter(workdir=..., ssh_target=...)")
