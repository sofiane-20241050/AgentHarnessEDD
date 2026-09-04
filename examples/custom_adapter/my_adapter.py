"""15 分钟接入你自己的 Agent Harness —— 最小适配器示例。

被测 Harness 的三种形态，对应三条接入车道（见根 README）：
  ① 进程内 + 有 SDK：实现 AgentAdapter 契约（本示例），工具直注入，轨迹最精确
  ② 进程内 + 有原生工具定义：参考 deepagents/tau 适配器（工具转换 + 事件流消费）
  ③ 外部 CLI/黑盒：参考 claude_code_adapter（子进程驱动 + MCP/文本协议工具桥）

AgentAdapter 契约只有三件事：
  - name：注册名（CLI --adapter 用）
  - run(task, tools, recorder, ...)：消费环境工具完成任务，返回 AgentResult
  - 轨迹采集：runner 已把工具用录制包装好（调用即入轨）；适配器只需补记
    对话侧事件（recorder.note("assistant", ...)）和用量（usage_total）

运行自测::

    python my_adapter.py       # 用 FakeLLM 离线驱动一轮（无需模型端点）
"""

from __future__ import annotations

import asyncio
from typing import Any

from ahedd.adapters import register_adapter
from ahedd.adapters.base import AgentResult, TaskInput
from ahedd.env.tools import ToolDefinition
from ahedd.llm.base import LLMClient
from ahedd.trace.schema import TrajectoryRecorder, Usage


class MyEchoAgentAdapter:
    """最简适配器示例：单次补全 + 执行全部工具调用 + 汇报。

    你的真实 Harness 只需把"单次补全"换成它的驱动方式（SDK 调用 / 事件流 /
    子进程），契约不变。
    """

    name = "my-echo-agent"

    def __init__(self, client: LLMClient, max_turns: int = 20) -> None:
        self.client = client
        self.max_turns = max_turns

    async def run(
        self,
        task: TaskInput,
        tools: list[ToolDefinition],
        recorder: TrajectoryRecorder | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        messages = [
            {"role": "system", "content": "你是业务助手，用工具完成任务。"},
            {"role": "user", "content": task.instruction},
        ]
        openai_tools = [
            {"type": "function",
             "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
            for t in tools
        ]
        by_name = {t.name: t for t in tools}
        total = Usage()

        for _turn in range(self.max_turns):
            resp = await self.client.chat(messages, openai_tools)
            total = Usage(
                input_tokens=total.input_tokens + resp.usage.input_tokens,
                output_tokens=total.output_tokens + resp.usage.output_tokens,
                cost_usd=total.cost_usd + resp.usage.cost_usd,
            )
            messages.append({"role": "assistant", "content": resp.content, "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.name, "arguments": __import__("json").dumps(tc.args, ensure_ascii=False)}}
                for tc in resp.tool_calls
            ]})
            if not resp.tool_calls:
                if recorder:
                    recorder.note("assistant", content=resp.content or "", usage=resp.usage)
                return AgentResult(
                    final_message=resp.content or "",
                    stop_reason="stop",
                    usage_total={"input_tokens": total.input_tokens,
                                 "output_tokens": total.output_tokens,
                                 "cost_usd": total.cost_usd},
                )
            if recorder:
                recorder.note("assistant", content=resp.content or "", usage=resp.usage)
            for tc in resp.tool_calls:
                result = await by_name[tc.name].func(**tc.args)  # runner 已包装：调用即入轨
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": __import__("json").dumps(result, ensure_ascii=False, default=str)})
        return AgentResult(final_message="", stop_reason="max_steps")


register_adapter("my-echo-agent")(lambda: (_ for _ in ()).throw(
    TypeError("my-echo-agent 需要客户端：MyEchoAgentAdapter(make_client(spec))")))

if __name__ == "__main__":
    from ahedd.datasets import get_dataset
    from ahedd.llm.fake import FakeLLMClient
    from ahedd.runner import run_case

    provider = get_dataset("mock")
    case = provider.load("mock")[0]
    adapter = MyEchoAgentAdapter(FakeLLMClient([
        {"tool_calls": [{"id": "c1", "name": "update_address",
                         "args": {"order_id": "ORD_1", "new_address": "北京市朝阳区"}}]},
        {"content": "已为您改好地址。", "finish_reason": "stop"},
    ]))

    outcome = asyncio.run(run_case(
        dataset="mock", adapter=adapter, env=provider.build_environment("mock"),
        case=case, task_id=case.id, instruction=case.instruction, trace_dir="runs/example",
    ))
    print("stop:", outcome.stop_reason, "| steps:", len(outcome.trajectory.steps))
    print("env_diff:", outcome.env_diff)
