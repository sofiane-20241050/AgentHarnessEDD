<div align="center">

[简体中文](README_zh.md) ｜ **English**

# AgentHarnessEDD

**An Evaluation-Driven Development (EDD) framework for AI agent harnesses**

Evaluate · Diagnose · Regress — define what "success" means *before* you build the agent.

`pip install agentharness-edd` ｜ import name / CLI: `ahedd`

🚧 Scaffold stage (v0.1.0): base contracts in place, runtimes landing milestone by milestone

</div>

---

## Why

Agents are probabilistic systems: the same input can produce different outputs across runs, so you cannot "read the code to tell if it's right" the way you would with traditional software. Teams keep running into the same three questions:

1. **How well did the agent do?** — after swapping the model or editing a prompt, did things get better or worse?
2. **Where did it go wrong?** — reasoning breakdown? wrong tool? bad arguments? or it never asked the user to clarify?
3. **Did the fix hold?** — bad cases get fixed once and lost forever, with no regression line of defense.

AgentHarnessEDD answers with **EDD (Evaluation-Driven Development)**: treat agents the way TDD treats code — define the success criteria first, then build; make every failure **attributable**, **replayable**, and **sedimented** as a regression case.

---

## What a single evaluation is made of

This is the project's "base-class map". **Onboarding any evaluation dataset (public benchmark or enterprise-private) boils down to providing six kinds of components** — each has a base class and a registry, all pluggable:

| # | Component | Base class / module | What you provide |
|---|-----------|--------------------|--------------------|
| 1 | **Dataset** | `ahedd.datasets.DatasetProvider` | Tasks: initial instruction + user scenario + rubric assertions (+ optional env seed) |
| 2 | **Environment** | `ahedd.env.Environment` | Tool set + virtual database; deterministic core with snapshot / diff |
| 3 | **Agent under test** | `ahedd.adapters.AgentAdapter` | See "three lanes" below — in-process adapter / MCP / RPC |
| 4 | **LLM endpoints** | `ahedd.config.ModelSpec` | Role-based endpoints: agent, user simulator, judge (all OpenAI-compatible: vLLM / OpenRouter / vendor APIs) |
| 5 | **Scoring** | `ahedd.scoring.Scorer` | Dual channel: rubric sliding-window LLM judge + deterministic assertions (write-ops / trajectory rules) |
| 6 | **Metrics** | `ahedd.scoring.metrics` | Avg@k / Pass@k / Pass^k + per-stage metrics (reasoning / tools / interaction) + efficiency (turns / tokens / cost) |

Plus two cross-cutting layers:

- **Trace (`ahedd.trace`)**: a unified trajectory schema (JSONL on disk — store the trace first, score later; judges are swappable and re-runnable)
- **Flywheel (`ahedd.regression`)**: freeze any failing run into a regression case (`RC-xxxx.yaml`); CI gate against regressions

## Onboarding an evaluation dataset, end to end

Using the first onboarded dataset, **VitaBench** (life-service domains: food delivery / in-store / OTA / cross-scenario, 400 tasks, 66 tools, ICLR 2026), as the example:

```text
① Implement dataset access   class VitaProvider(DatasetProvider)  # tasks, scenarios, rubric assertions
② Attach the environment     provider.build_environment(domain)   # tools + virtual DB (with distractors)
③ Configure the 3 LLM roles  models.yaml: agent / user_simulator / judge
④ Pick the on-ramp lane      ahedd run --dataset vita --adapter openai-loop --model <agent-model>
⑤ Get the report             ahedd report                        # score + per-stage diagnosis + trajectory replay
⑥ Sediment failures          ahedd freeze <run_id>               # bad case -> regression case -> ahedd ci
```

> The framework is not bound to any single benchmark: VitaBench is simply `ahedd`'s first dataset plugin. Enterprise-private datasets onboard through the same base classes (`DatasetProvider` + `Environment`), and run scored separately from public sets.

## Three on-ramp lanes for the agent under test

| Lane | For | Integration | Trace capture |
|------|-----|-------------|---------------|
| **In-process** | Python-native: DeepAgents (LangGraph), tau, any OpenAI-compatible model | tool injection + event stream / callbacks | precise (token-level) |
| **MCP** | Closed clients: Claude Code, Codex CLI, Dify, Coze, … | environment exposed as an MCP server; the agent connects in | protocol-message recording |
| **RPC** | CLI agents (headless mode) | stdio JSON-RPC subprocess driving | protocol-message recording |

Rule of thumb: **in-process > MCP > RPC** — pick by the shape of the agent under test; all lanes share the same environment core and trajectory schema.

## Quick start

```bash
uv venv .venv
uv pip install -e ".[dev,deepagents,tau]"

ahedd datasets list    # registered datasets (built-in: mock self-test domain)
ahedd adapters list    # registered adapters (openai-loop / deepagents / tau)

# configure .env (see .env.example), then run the mock domain with a real model:
ahedd run --dataset mock --adapter openai-loop --disable-thinking
ahedd run --dataset mock --adapter deepagents --disable-thinking
ahedd run --dataset mock --adapter tau
```

## Repository layout

```text
AgentHarnessEDD/
├── src/ahedd/
│   ├── datasets/     # dataset onboarding base classes + registry (tasks / rubrics / scenarios)
│   ├── env/          # environment base class (tools + virtual DB + snapshot/diff)
│   ├── adapters/     # three-lane agent adapters
│   ├── trace/        # unified trajectory schema & recorder
│   ├── scoring/      # dual scoring channels + metrics (Avg@k / Pass@k / Pass^k)
│   ├── regression/   # bad case -> regression-case assets
│   ├── report/       # single-file HTML diagnostic report
│   ├── config.py     # models.yaml: role-based LLM endpoints
│   ├── runner.py     # per-task orchestration (record -> persist -> score, decoupled)
│   └── cli.py        # the ahedd entry point
├── cases/            # cases (dataset-bundled / user-authored)
├── regressions/      # frozen regression-case assets (versioned with git)
├── docs/             # research report and friends
└── tests/
```

## Roadmap

- [x] **D1** `vita` dataset plugin (real VitaBench tasks: rubric scoring with partial credit, loop circuit breaker, first regression case RC-0001 frozen)
- [x] **D2** more lanes: DeepAgents and tau in-process adapters (validated with a real model); trajectory JSONL persistence
- [x] **D3** dual scoring channels + metrics: rubric sliding-window judge (validated with a real model; catches the tau refuses-to-edit bad case), deterministic assertions, trajectory-dynamics metrics (useful-action ratio / error recovery / STS / pareto), offline `ahedd score`
- [x] **D4** single-file HTML diagnostic report + `ahedd freeze` (auto triage: loop/param/unfinished/constraint)
- [x] MCP deepening: deepagents MCP mode, per-case server restart (no cross-case env pollution)
- [ ] dataset entry-points, private-set authoring guide, user simulator, `ahedd ci` gate

## Docs

- [Research report (Chinese): positioning, VitaBench deep dive, competitor analysis, bad-case sedimentation design](docs/research-report.md)

## License

[Apache-2.0](LICENSE)
