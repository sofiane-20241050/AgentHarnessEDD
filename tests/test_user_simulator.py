"""用户模拟器测试：双 FakeLLM（Agent 侧 + 用户侧）离线验证多轮对话模式。"""


from ahedd.adapters.openai_loop import OpenAILoopAdapter
from ahedd.datasets import get_dataset
from ahedd.llm.fake import FakeLLMClient
from ahedd.runner import run_case
from ahedd.user import UserSimulator, is_stop


def _sim_client() -> FakeLLMClient:
    return FakeLLMClient([
        {"content": "你好，帮我点份清淡的米线送到医院"},                                        # 开场白
        {"content": "我下午一点半有手术，十二点前送到吧"},                                      # 回答追问
        {"content": "嗯就这么定了，谢谢 ###STOP###"},                                           # 结束
    ])


async def test_user_simulator_multi_turn(tmp_path) -> None:
    """Agent 终答 -> 模拟器回话 -> 继续；###STOP### 结束；模拟器用量并入总账。"""
    provider = get_dataset("mock")
    case = next(c for c in provider.load("mock") if c.id == "mock_001_change_address")
    case = case.model_copy(update={"instruction": "你今天胃不舒服，想点份清淡米线到医院吃，隐含要求十二点前送达。"})
    env = provider.build_environment("mock")

    # Agent 侧脚本：问一句澄清 -> 改地址 -> 汇报（触发模拟器 STOP）
    agent_client = FakeLLMClient([
        {"content": "请问需要几点前送达？", "finish_reason": "stop"},
        {"tool_calls": [{"id": "c1", "name": "update_address",
                         "args": {"order_id": "ORD_1", "new_address": "北京市朝阳区"}}]},
        {"content": "已安排，十二点前送达北京市朝阳区。", "finish_reason": "stop"},
    ])
    adapter = OpenAILoopAdapter(agent_client)
    simulator = UserSimulator(_sim_client(), case)

    outcome = await run_case(
        dataset="mock",
        adapter=adapter,
        env=env,
        case=case,
        task_id=case.id,
        instruction=case.instruction,
        trace_dir=str(tmp_path),
        user_simulator_factory=lambda c: simulator,
    )

    types = [s.type for s in outcome.trajectory.steps]
    # 首条 user 是模拟器开场白（非原始剧本）；随后形成 user/assistant 交替
    assert outcome.trajectory.steps[0].content == "你好，帮我点份清淡的米线送到医院"
    assert "胃不舒服" not in outcome.trajectory.steps[0].content  # 剧本未直接泄给 Agent
    assert types.count("user") == 2  # 开场白 + 一次追问回答（###STOP### 不入轨）
    assert outcome.stop_reason == "stop"
    assert "北京市朝阳区" in outcome.env_diff["orders"]["after"]["ORD_1"]["address"]
    # 用量总账 = Agent(3 次补全) + 模拟器(3 次补全) = 6 in / 6 out（Fake 每次 1/1）
    assert outcome.trajectory.meta.total_usage.input_tokens == 6
    assert outcome.trajectory.meta.total_usage.output_tokens == 6


def test_is_stop() -> None:
    assert is_stop("好了 ###STOP###")
    assert not is_stop("再帮我看看")
