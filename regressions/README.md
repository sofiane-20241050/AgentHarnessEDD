# regressions/ —— 回归用例资产

Bad Case 沉淀的物理形态（见 `docs/research-report.md` §6）：

```text
regressions/
└── cases/
    └── RC-0007.yaml    # 环境种子 + 用户剧本 + rubric 断言 + 归因标签 + 红基线轨迹
```

- 由 `ahedd freeze <run_id>` 自动生成，人工确认归因标签后入库
- 随 git 版本化：每个用例都是一次可重放的失败
- `ahedd ci` 对全量用例做红绿检查，构成变更门禁（Prompt / 模型 / 工具描述变更必跑）
