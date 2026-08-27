# VisionRestore Agent LangGraph 标准化重构报告

版本：visionrestore-langgraph-v2.1
日期：2026-08-20
范围：API 后端 Agent 编排、状态、工具、检查点、后处理恢复、观测与测试。模型算法、权重和前端交互语义保持兼容。

## 1. 结论

本项目适合采用“一个领域 Agent + LangGraph 状态图 + 专用服务/工具”的结构，不适合当前阶段拆成多个自治 Agent。

原因是 VisionRestore 的任务边界清晰：分析一张已注册 RGB 静态图，规划少量互补候选，分别从原图推理，基于真实输出评估排序，再在必要时等待用户确认去噪或超分。这里需要的是可恢复的确定性编排、并行候选汇总和人工确认，而不是多个 Agent 之间的自然语言协商。

重构后的默认 V2 路径已经切换到 LangGraph。旧单候选路径与旧 V2 多候选 Agent 继续保留为显式兼容入口。

## 2. 重构前的项目结构和已有能力

### 2.1 已有业务能力

- FastAPI v1/v2 任务、文件、报告和 WebSocket 接口。
- React/Vite Web 界面。
- 本地图像退化分析、自然语言意图解析和硬件探测。
- 模型注册表、checkpoint 状态、外部 PyTorch worker 隔离。
- Retinexformer、DarkIR、HVI-CIDNet、FLOL、SCI、Zero-DCE 等增强候选。
- CandidatePlanner、MultiCandidateExecutor、CandidateEvaluator、CandidateRanker 和 ResultSelector。
- 可选多模态模型建议，带本地校验和有限加分。
- 本地受限知识检索，不向外部模型默认发送检索内容。
- ROI/区域硬约束、一次有界质量重试。
- 残余退化诊断、NAFNet 去噪和 Real-ESRGAN 超分确认、重新评分和回滚。
- SQLite 业务持久化、任务报告、模型运行日志和文件 lineage。

### 2.2 重构前的核心问题

1. `EnhancementAgentV2.run` 是一个较长的命令式函数。步骤存在，但没有显式 node/edge、条件路由和可恢复检查点。
2. TaskRecord 同时承担业务实体、运行态、消息和部分流程控制，状态边界不清晰。
3. 图像分析、硬件检查、知识检索和候选执行属于工具调用，但此前没有统一 tool definition、schema、router 和 execution result。
4. 候选规划和执行具备多候选语义，但缺少标准 fan-out/reducer 表达。
5. 后处理确认由 API 直接调用 `PostprocessController`，绕过原 Agent 流程；进程重启后无法从原决策点恢复。
6. 日志以 TaskRecord.logs 为主，缺少结构化 step、run、tool call、duration 和 workflow event。
7. 旧 V2 Agent 难以进行节点级测试，失败恢复和幂等性主要依赖业务代码约定。
8. 项目已有知识检索，但不是 embedding + vector database 方案；如果仅为“看起来工业化”引入向量库，会增加部署复杂度而没有当前收益。
9. 当前不需要 Multi-Agent。若把分析、规划、执行、评估分别包装成自治 Agent，会增加 prompt、消息轮次、非确定性和 GPU 调度冲突。

## 3. 标准 Agent 模块对应关系

| 标准模块 | 当前实现/重构后落点 | 说明 |
|---|---|---|
| LLM 接入层 | `visionrestore/ai/providers.py`、`ai/prompts.py`、`graph/advisory.py`、`apps/api/prompts/` | provider、API 配置、prompt registry 已存在；图只接收受校验的 advisory，不授予最终控制权。 |
| Agent 核心层 | `graph/runtime.py`、`graph/builder.py` | Agent Loop 被显式图替代；Planning、Reasoning、Decision 分别落在候选规划、语义建议/规则、排序选择与条件路由节点。 |
| State 管理 | `graph/state.py`、`schemas/task.py`、LangGraph checkpoint | 图状态保存当前阶段、候选、消息、事件、确认项；TaskRecord 是 API 投影和 durable business record。 |
| Memory | checkpoint、TaskRecord.messages、workflow_events、reports、local knowledge | short-term 是当前图状态/messages；durable memory 是 SQLite 业务记录和报告。未增加泛化用户画像或不可控长期记忆。 |
| Tool | `visionrestore/tools/` | 统一定义、Pydantic input schema、注册表、路由器、执行结果和错误码。 |
| Knowledge | `visionrestore/context/`、`apps/api/knowledge/` | 当前为 allowlist 本地检索。没有 embedding/vector DB，属于有意的轻量实现。 |
| Workflow | `graph/builder.py`、`graph/subgraphs/` | LangGraph 主图、候选执行子图、后处理子图、Send fan-out、reducer、conditional edge、interrupt/resume。 |
| Evaluation | evaluator、candidate_evaluator、result_selector、residual_analyzer、postprocess_controller | 真实输出评分、硬约束淘汰、最终排序、残余诊断、后处理重评和回滚。 |
| Observability | `graph/events.py`、workflow_events、graph_runs、TaskRecord.logs/messages | 结构化 run/step/tool 事件与兼容日志并存。 |
| Config | `core/config.py`、`core/model_config.py`、`config/*.yaml` | API、模型、路由、后处理参数继续由现有配置系统管理。 |

## 4. 重构后目录结构

```text
apps/api/visionrestore/
├── api/
│   └── v2_routes.py                 # HTTP 适配、确认恢复、工作流观测接口
├── graph/
│   ├── __init__.py
│   ├── agent.py                     # LangGraph Agent 门面：run/resume
│   ├── advisory.py                  # 多模态建议编排与本地降级
│   ├── builder.py                   # 主图节点和边装配
│   ├── checkpoint.py                # SQLite checkpointer
│   ├── events.py                    # 结构化 event/message 工厂
│   ├── runtime.py                   # 节点实现、条件路由和 TaskRecord 投影
│   ├── state.py                     # TypedDict、candidate/event reducers
│   └── subgraphs/
│       ├── candidate_execution.py   # 单候选执行子图
│       └── postprocess.py           # interrupt/resume 后处理子图
├── tools/
│   ├── definitions.py               # ToolDefinition/Context/Result
│   ├── schemas.py                   # 工具输入 schema
│   ├── registry.py                  # 工具注册和发现
│   └── router.py                    # 校验、调用、错误归一化、耗时
├── agent/
│   ├── enhancement_agent.py         # legacy single-candidate
│   ├── enhancement_agent_v2.py      # legacy V2 rollback path
│   ├── candidate_planner.py
│   └── intent_parser.py
├── services/                         # 领域服务，不感知 LangGraph
├── adapters/                         # 模型/worker 适配
├── context/                          # 受限本地知识检索
├── schemas/                          # API/领域模型
├── storage/                          # 业务 SQLite
└── core/                             # 配置与模型配置
```

该结构没有新增通用 BaseAgent、AgentFactory、EventBus 或 Repository 抽象层。当前代码量和单进程部署不需要这些层。

## 5. 文件职责和迁移清单

### 5.1 新增图模块

- `graph/state.py`：定义唯一工作流状态契约。候选结果通过 candidate_id reducer 合并，消息和事件按唯一 ID 去重。
- `graph/runtime.py`：承载每个 node 的薄编排逻辑；调用已有领域服务，不复制模型算法。
- `graph/builder.py`：只装配节点、普通边和条件边，便于审查流程拓扑。
- `graph/agent.py`：向 TaskService 暴露 `run` 和 `resume`，隐藏 LangGraph 配置细节。
- `graph/checkpoint.py`：将线程状态持久化到 `data/langgraph/checkpoints.sqlite`。
- `graph/events.py`：为 run/step/tool 生成结构化事件和消息。
- `graph/advisory.py`：从旧 V2 大函数中抽出可选多模态建议流程，继续执行本地校验和 fallback。
- `graph/subgraphs/candidate_execution.py`：每次只执行一个候选，输出严格限制为 candidate_results/events/messages，避免并发覆盖父状态。
- `graph/subgraphs/postprocess.py`：准备确认、interrupt、恢复、执行、重评和下一步路由。

### 5.2 新增工具模块

- `tools/definitions.py`：工具元数据、输入上下文和标准结果。
- `tools/schemas.py`：AnalyzeImage、InspectHardware、RetrieveContext、ExecuteCandidate 输入契约。
- `tools/registry.py`：注册、冲突检查、查询。
- `tools/router.py`：Pydantic 校验、执行、耗时、统一错误码。

### 5.3 修改现有文件

- `services/task_service.py`：多候选默认使用 LangGraph；保留 `single_candidate` 和 `workflow_engine=legacy`；新增带任务级锁的 postprocess resume。
- `api/v2_routes.py`：LangGraph 任务的确认请求恢复原 thread；legacy 任务仍走旧控制器；新增只读 workflow 观测接口。
- `schemas/task.py`：增加 workflow engine/thread/version、events、messages 和 pending_confirmation。
- `storage/database.py`：增加 workflow_event 和 graph_run 实体表。
- `pyproject.toml`：增加 LangGraph 与 SQLite checkpoint 依赖。
- `tests/test_langgraph_workflow.py`：覆盖 fan-out/reducer、原图输入、评分隔离、interrupt/resume 和幂等性。
- `tests/test_tool_router.py`：覆盖 schema validation 和标准工具结果。

### 5.4 保留但不再默认进入的代码

- `EnhancementAgent`：保留旧单候选模式。
- `EnhancementAgentV2`：保留 legacy V2 回滚能力。
- `HierarchicalRouter`：保留兼容和本地评分复用，但不再代表 V2 最终执行决策。

## 6. 主数据流

```text
POST /api/v2/tasks
  -> TaskService 创建 TaskRecord / thread_id
  -> LangGraph initialize
  -> 图像分析工具
  -> 意图解析
  -> 硬件与模型快照工具
  -> 本地知识检索工具
  -> 可选多模态 advisory + 本地校验
  -> CandidatePlanner
  -> Send(candidate A/B/C)
       -> candidate subgraph
       -> 每个候选只读 original input
       -> worker inference
       -> real-output evaluation
  -> reducers 汇总候选
  -> 可选一次 ROI 质量重试（仍从原图）
  -> CandidateRanker / ResultSelector
  -> best result
  -> residual diagnosis
       -> 无后处理：finalize
       -> 需后处理：prepare confirmation -> interrupt
  -> 用户提交 decision
  -> Command(resume)
  -> PostprocessController
  -> re-score -> adopt 或 rollback
  -> finalize + report
```

### 6.1 状态边界

- LangGraph State：一次工作流的运行态和节点间数据。
- Checkpoint：用于 interrupt/resume 和进程内故障后的线程恢复。
- TaskRecord：对 API/UI 暴露的业务投影，也是兼容旧代码的数据结构。
- Entity tables：candidate、ranking、postprocess、workflow event 和 graph run 的审计记录。
- 文件系统：原图、候选输出、后处理输出和报告的 artifact lineage。

## 7. 关键工业约束如何落实

### 7.1 原图输入不变式

CandidatePlanner 输出的每个候选都标记 `input_policy=original_input_only`。fan-out 状态向每个候选传递相同 input_path，候选间不串联输出。ROI 重试也从原图重新执行。

### 7.2 规划分与最终分隔离

- planning_score 只在推理前决定候选是否值得执行。
- final_score 只在真实输出产生后由 CandidateRanker/ResultSelector 写入。
- 任务报告和 metrics 明确记录 `planning_score_is_not_final_score=true`。

### 7.3 多模态模型权限边界

外部模型只能提供 scene、region 和候选建议。建议必须通过 schema 和本地规则验证，只能形成有限 bonus；不能选择不存在的模型/checkpoint，不能直接修改像素，不能绕过区域硬约束、最终评分或用户确认。

### 7.4 后处理人工确认和幂等性

图在 wait node 调用 `interrupt`，确认准备和状态写入发生在 interrupt 前的独立节点。恢复只继续中断后的路径，不重复执行增强候选。TaskService 对同一任务的 resume 使用任务级锁并校验等待状态和 operation。

### 7.5 故障和回滚

- 节点工具错误转换为结构化错误码并使图进入 failed。
- candidate failure 可以与其他 candidate success 同时存在，由 ranker 淘汰失败项。
- 后处理执行后必须重新评分；下降时恢复后处理前最佳结果。
- `workflow_engine=legacy` 可显式回退旧 V2；`single_candidate` 保持旧单候选。

## 8. 为什么不采用 Multi-Agent

当前不建议建立 Planner Agent、Executor Agent、Evaluator Agent、Reflection Agent 等多个自治单元。

### 不采用的原因

- 模型推理和评估服务已有严格接口，不需要 Agent 间自然语言协商。
- 多 Agent 会重复携带图像指标、候选和消息，增加 token 和延迟。
- 多个 Agent 同时调度本地 GPU 容易造成显存竞争。
- 评分和安全边界需要确定性代码，不能由 Agent 自由协商。
- 当前用户任务不是开放式研究或跨领域长任务。

### 何时才考虑 Multi-Agent

只有出现跨任务批处理、独立数据治理/合规审查、远程异构 GPU 调度、需要不同权限边界的外部系统协作，且这些单元有独立 SLA 和失败域时，才应考虑 supervisor + workers。即使届时，也应优先把 worker 设计成确定性服务，而非全都变成 LLM Agent。

## 9. Knowledge 和 Memory 的取舍

### 当前实现

- Short-term memory：LangGraph state、messages、events、checkpoint。
- Durable task memory：TaskRecord、entity tables、报告和 artifacts。
- Knowledge：allowlisted local knowledge + bounded lexical retrieval。

### 暂不引入

- 不引入向量数据库、embedding 服务和通用长期用户记忆。
- 当前知识量小、内容稳定、访问范围明确，关键词/规则检索可审计且部署简单。
- 当知识文档达到数百/数千、召回质量有可量化不足，并且有离线索引生命周期后，再引入 embedding/vector DB。

## 10. 自测范围和验收标准

新增测试验证：

1. ToolRouter 对输入执行 Pydantic 校验并返回 tool call metadata。
2. 多候选通过 LangGraph Send fan-out 执行并由 reducer 汇总。
3. 所有候选 input_path 完全相同，满足原图输入不变式。
4. planning_score 与 final_score 同时存在且来源不同。
5. 最佳候选由真实输出结果决定。
6. 后处理推荐触发 interrupt，TaskRecord 进入 awaiting 状态。
7. Command(resume) 后图完成，增强候选执行次数不增加。
8. 真实 API 后台任务可以进入 completed/failed/awaiting 状态。
9. 完整既有后端测试覆盖 API、模型适配、候选、评估、ROI、知识、多模态和后处理。
10. 前端执行 TypeScript/Vite production build。

### 10.1 本次实际结果

- 后端：`python -m pytest -q`，73 项测试全部通过。
- 新增图/工具测试：4 项通过，覆盖双候选 fan-out、原图输入、评分隔离、interrupt/resume 幂等性、空重试计划回退和工具 schema。
- API 烟雾测试：后台 LangGraph 任务成功进入允许的终态/确认态。
- 前端：`npm run build` 通过，TypeScript 编译和 Vite production build 成功，1578 个模块完成转换。
- 模型注册验证：脚本退出码 0；Retinexformer、DarkIR、HVI-CIDNet、FLOL、SCI、Zero-DCE、LPDM、NAFNet、Real-ESRGAN 注册状态可读取。`realesr_general_x4v3` 权重仍缺失，x2/x4 plus 状态正常。
- 静态检查：关键 Python 模块 py_compile 通过；`git diff --check` 无新增空白错误（仅 Windows LF/CRLF 提示）。
- 架构图结构校验：0 errors、0 warnings。
- 已知非阻断警告：Starlette TestClient/httpx2 迁移提醒；Pillow `Image.getdata` 将在未来版本弃用。

## 11. 当前仍存在的限制

- 任务后台执行仍使用进程内 ThreadPoolExecutor，单进程重启后不会自动重新提交 queued 任务；checkpoint 主要解决图内恢复和人工确认。
- SQLite checkpointer 适合当前单机规模，不是多实例分布式锁和高吞吐队列。
- 候选 fan-out 在图语义上并行，但默认 `max_concurrency=1`，有意避免单 GPU 显存争用。
- workflow events 已持久化，但 UI 仍主要消费 TaskRecord/WebSocket 投影，尚未做完整 trace 时间线视图。
- long-term memory、embedding 和 vector DB 未实现，这是当前阶段的范围选择，不是缺陷。
- 外部多模态 provider 的可用性、费用和延迟仍由第三方服务决定。

## 12. 后续建议

### P0：上线前

- 在真实模型环境跑一组固定低照度样本，核对每个模型实际显存、耗时、输出路径和 final score。
- 备份现有业务 SQLite，并验证一次进程重启后的 awaiting task resume。
- 为 `GET /api/v2/tasks/{task_id}/workflow` 增加前端只读时间线展示。

### P1：稳定性

- 启动时扫描 queued/running 但没有活跃 worker 的任务，标记 interrupted 或按策略恢复。
- 为 checkpoint 文件增加维护策略和旧 thread 清理命令。
- 给 workflow event 增加统一 error category 和 retryable 标志。

### P2：达到明确规模后再做

- 当单机队列成为瓶颈时再引入独立任务队列和 GPU worker，不需要改变 LangGraph 业务图。
- 当知识召回有数据证明不足时再增加 embedding/vector DB。
- 只有出现真正独立的权限、SLA 和失败域时才考虑 Multi-Agent supervisor。

## 13. 验收结论

本次重构完成了“标准化 LangGraph”，但没有把项目变成框架样板工程。Agent 仍是一个面向低照度恢复的领域 Agent；LangGraph 负责状态、条件路由、候选 fan-out、检查点和人工确认恢复；已有 service/adapter 继续负责确定性业务能力。该结构与当前规模匹配，也为未来队列、分布式 worker、向量检索或受控 Multi-Agent 留出了清晰边界。
