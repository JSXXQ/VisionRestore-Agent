<div align="center">

<img src="docs/social-preview.png" alt="VisionRestore Agent — 多模型协同低照度图像增强系统" width="100%"/>

<br/>

[![许可证: MIT](https://img.shields.io/badge/license-MIT-6366f1?style=for-the-badge)](LICENSE)
[![当前版本: V2.3](https://img.shields.io/badge/version-V2.3-22d3ee?style=for-the-badge)](CHANGELOG.md)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-22d3ee?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-76b900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React + Vite](https://img.shields.io/badge/React%20%2B%20Vite-TypeScript-61dafb?style=for-the-badge&logo=react&logoColor=black)](https://react.dev/)
[![LangGraph](https://img.shields.io/badge/LangGraph-多%20candidate-8b5cf6?style=for-the-badge)](https://langchain-ai.github.io/langgraph/)
[![本地优先](https://img.shields.io/badge/%E6%9C%AC%E5%9C%B0%E4%BC%98%E5%85%88-%E6%97%A0%E4%BA%91-22c55e?style=for-the-badge)](#-windows-%E5%90%AF%E5%8A%A8)
[![Windows](https://img.shields.io/badge/%E5%B9%B3%E5%8F%B0-Windows%2011-0078d4?style=for-the-badge&logo=windows&logoColor=white)](#-windows-%E5%90%AF%E5%8A%A8)

[English](README.md) · 中文 · [架构总览](docs/architecture.md)

</div>

---

## ✨ 项目定位

**VisionRestore Agent** 是一个由 LangGraph 编排的、**多候选并行**的低照度图像增强 Agent。默认工作流**不预先选定一个模型**，而是：

1. 用 `ImageAnalyzer` 解析图像退化
2. 用 `IntentParser`（本地规则）解析用户意图
3. 检查硬件与模型就绪状态
4. 让 `CandidatePlanner` 规划 **1–3 个互补的候选**
5. 让 `MultiCandidateExecutor` **同时**在原始图上跑这些候选
6. 用 `CandidateEvaluator` 对**真实输出**打分
7. `CandidateRanker` 选最优，再交给 `ResultSelector`
8. 必要时通过 `interrupt/resume` 等你确认 denoise / 超分辨率
9. 给你最终结果 + 完整 lineage

整条链路不依赖任何云端 API Key（多模态 AI 默认为 `disabled`，仅作建议）。你的图像**永远不离开本机**。

---

## 🎯 V2.3 相比传统工具

| 传统工具 | VisionRestore Agent V2.3 |
|---|---|
| 写死一个模型 | **多候选 fan-out**，对真实输出打分 |
| 黑盒决策 | 两层确定性路由 + 22 步状态机 |
| 必须在云端推理 | **本地 CUDA 优先**，CPU 也能跑 SCI / FLOL |
| 写死一个 LLM | 多模态 AI 走 `disabled/openai/openai_compatible/anthropic/gemini` 五种槽位 |
| 单语言 | **中英双语** UI、prompt、报告 |
| 串行后处理 | 真正支持 **interrupt / resume**，差则回滚 |

---

## 🖼️ 界面截图

<div align="center">

### 🛠️ 增强工作台
<img src="docs/screenshots/ui-workbench.jpg" alt="增强工作台" width="95%"/>
<sub>上传 → 描述需求 → 多模型 fan-out → 左右对比 → 选最佳 → 保存 / 导出报告</sub>

<br/><br/>

### 🧠 模型与权重
<img src="docs/screenshots/ui-model-center.jpg" alt="模型与权重" width="95%"/>
<sub>10 个 worker、24 个权重。健康检查、占用空间、可路由状态一目了然</sub>

<br/><br/>

### ⚙️ 系统设置
<img src="docs/screenshots/ui-system-settings.jpg" alt="系统设置" width="95%"/>
<sub>硬件 / CUDA / PyTorch / IntentParser / LLM 实时状态；多模态 Key 写入 `.env` 不回显</sub>

</div>

---

## 🧠 架构（V2.3）

```
┌──────────────────────────────────────────────────────────────────────┐
│  用户请求 → ImageAnalyzer → IntentParser → HardwareInspector         │
│                          │                                           │
│                          ▼                                           │
│          CandidatePlanner  (local + knowledge + AI bonus)             │
│                          │                                           │
│            ┌─────────────┼─────────────┐                             │
│            ▼             ▼             ▼                             │
│         Worker A       Worker B       Worker C   (并行、真实推理)     │
│            │             │             │                             │
│            └─────────────┼─────────────┘                             │
│                          ▼                                           │
│     CandidateEvaluator → CandidateRanker → ResultSelector             │
│                          │                                           │
│                          ▼                                           │
│   ResidualDegradationAnalyzer → [interrupt] → NAFNet / Real-ESRGAN   │
│                          │                                           │
│                          ▼                                           │
│                  最终结果 + lineage                                   │
└──────────────────────────────────────────────────────────────────────┘
```

完整数据流图：[docs/diagrams/visionrestore_v22_dataflow.png](docs/diagrams/visionrestore_v22_dataflow.png)

**硬约束**（写在测试里）：

- 所有候选**只读原始图**，禁止串行增强
- `final_score` **只看真实输出**；knowledge 调整限定在 `[-5, +5]`，不进入 final_score
- 手动选模型时 **knowledge 与 AI 都不参与调整**
- 后处理需要**你显式确认**；若质量分下降则自动回滚

详见 [docs/agent_workflow.md](docs/agent_workflow.md)（22 个状态）。

---

## 🧩 模型矩阵

| 类别 | Worker | 状态 | 权重 / 变体 |
|---|---|---|---|
| 增强（自动池） | **Retinexformer** | ✅ 真实推理 | LOL-v2-real · SDSD-indoor · SDSD-outdoor · NTIRE |
| 增强（自动池） | **DarkIR** | ✅ 真实推理 | real-lol · LOLBlur · LOLBlur width64 · AR-LOL |
| 增强（自动池） | **HVI-CIDNet** | ✅ 真实推理 | SiCe · FiveK · LOLBlur · SID |
| 增强（自动池） | **FLOL** | ✅ 真实推理 | LOLv2-Real · UHD-LL |
| 增强（自动池） | **SCI** | ✅ 真实推理 | easy · medium · difficult |
| 基线 / 手动 | **Zero-DCE** | ✅ 真实推理 | Epoch99（不进自动池） |
| 后处理（手动） | **NAFNet** | ✅ 真实推理 | SIDD width32 · SIDD width64（去噪） |
| 后处理（手动） | **Real-ESRGAN** | ✅ 真实推理 | v2（动漫 / 照片）· v4（照片） |
| 辅助（受限） | **LPDM** | ⚠️ 真实推理（自检有限） | LOL |
| 辅助（暂未接入） | **MambaIR** | ⛔ 源码未 vendored | — |

**正式 V2 自动池只包含**：`retinexformer` · `darkir` · `hvi_cidnet` · `flol` · `sci`。
`NAFNet` 仅作 `experimental/manual_only` 后处理去噪；`Zero-DCE` 仅作 legacy / manual 基线，**不参与自动候选竞争**。

---

## 🪜 分层路由

**第一层（架构）**：

- 质量 + CUDA + 显存充足 → Retinexformer
- 均衡 + 显存充足 → Retinexformer
- 速度优先 / 仅 CPU / 低显存 → SCI
- Zero-DCE 仅作为手动 / 对比 / 兜底，**不**作为自动首选

**第二层（权重）** — 顺序来自 `config/routing_rules.yaml`：

- 未知场景：LOL-v2-real → NTIRE → SDSD-outdoor → SDSD-indoor
- 室内：SDSD-indoor → LOL-v2-real → NTIRE → SDSD-outdoor
- 室外：SDSD-outdoor → LOL-v2-real → NTIRE → SDSD-indoor

SCI 走**亮度统计投票**（均值、中位数、25% 分位、暗像素比例、动态范围），不会用亮度伪造室内/室外语义。

如果场景无法判断 → 显示「场景未知」，**默认 LOL-v2-real**，不靠亮度统计假装语义。

自动模式**最多 1 主 + 1 兜底**；主结果通过质量检查就跳过兜底。

---

## 🚀 Windows 启动

**环境要求**：Windows 10/11 · Python 3.12+ · CUDA 显卡（RTX 3060 12GB 已验证）· Node.js 20+ · Git

```powershell
cd E:\codex_project\VisionRestore-Agent
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start.ps1
```

启动后：

- 前端页面：<http://127.0.0.1:5173>
- 本地 FastAPI：<http://127.0.0.1:8000>
- API 文档：<http://127.0.0.1:8000/docs>

> **模型权重不随仓库分发**。首次启动会运行 `scripts\download_models.ps1`，路径写在 `config\models.local.yaml`（已 gitignore）。详见 [docs/model_download_checklist.md](docs/model_download_checklist.md)。

停止：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\stop.ps1
```

---

## 🧪 模型与代码验证

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\check_environment.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\validate_models.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_smoke_test.ps1
.\.venv\Scripts\python.exe -m pytest
npm.cmd run build --prefix apps\web
```

**已验证的真实推理**（`E:\anconda\envs\pytorch\python.exe`，PyTorch 2.5.1，CUDA 12.1，RTX 3060 12GB）：

| Worker · 权重 | 状态 |
|---|---|
| Retinexformer · LOL-v2-real | ✔ 通过 |
| Retinexformer · SDSD-outdoor | ✔ 通过 |
| DarkIR · real-lol | ✔ 通过 |
| HVI-CIDNet · SiCe | ✔ 通过 |
| FLOL · LOLv2-Real | ✔ 通过 |
| SCI · medium | ✔ 通过 |
| Zero-DCE · Epoch99 | ✔ 通过 |
| NAFNet · SIDD width32 | ✔ 通过（去噪） |
| Real-ESRGAN · v2 | ✔ 通过（超分辨率） |

输出图像经过尺寸断言，**保持输入宽高一致**。

---

## 🔌 CLI

```powershell
# 仅分析
.\.venv\Scripts\python.exe -m visionrestore.cli analyze --input data\cache\test_images\low_light.png

# 自动 V2 多候选
.\.venv\Scripts\python.exe -m visionrestore.cli enhance `
    --input input.jpg `
    --request "自然增强暗部，保护高光，质量优先" `
    --mode auto `
    --output output.png

# 手动指定模型 + 权重
.\.venv\Scripts\python.exe -m visionrestore.cli enhance `
    --input input.jpg `
    --model retinexformer --weight lol_v2_real `
    --output output.png

# 多候选对比
.\.venv\Scripts\python.exe -m visionrestore.cli compare `
    --input input.jpg `
    --output compare.png `
    --candidates retinexformer:lol_v2_real retinexformer:sdsd_outdoor sci:difficult
```

---

## 🌐 API 速览

```
GET  /api/v1/health
GET  /api/v1/system
GET  /api/v1/models
GET  /api/v1/models/{model}/weights
POST /api/v1/intent/parse
POST /api/v1/images
POST /api/v1/tasks            # 创建后立即返回 task_id
GET  /api/v1/tasks/{id}        # 轮询 / WebSocket
GET  /api/v1/history
GET  /api/v1/files/{id}
GET  /api/v1/settings
PUT  /api/v1/settings

# 可选多模态（默认 disabled）
GET  /api/v1/ai/providers
POST /api/v1/ai/providers/{provider_id}/health-check
POST /api/v1/ai/analyze
GET  /api/v1/ai/settings
PUT  /api/v1/ai/settings
POST /api/v1/ai/test
```

完整 API：[docs/api.md](docs/api.md)

---

## 🤖 多模态 AI（可选）

可选，**不是必须**。后端提供 `disabled`、`openai`、`openai_compatible`、`anthropic`、`gemini` 五个供应商槽位。`openai_compatible` 可以连接任何兼容 OpenAI `/chat/completions` 格式的厂商 Base URL。

- 默认 `disabled`
- 外部 API **只做语义分析和路由建议**，不直接控制本地模型推理
- 完整 Key 写入本地 `.env`，前端**永不回显**

在前端「系统设置」页可直接配置。

---

## 📏 评价指标

第一版没有 GT 上传，**不计算 PSNR / SSIM / LPIPS**。系统只显示无参考指标：

- 亮度变化、暗像素变化、过曝变化、对比度、动态范围、清晰度、噪声、色偏、熵
- 结构保持估计、推理耗时、显存占用、文件大小

> 无参考指标只能辅助判断，不能完全替代人工主观评价。

---

## ⚠️ 显存不足和分辨率规则

**本项目不是超分辨率项目。** 用户保存的结果**必须与输入图像宽高一致**。模型内部如需填充，由 adapter / runner 处理，推理后裁剪回原始尺寸。高分辨率分块推理接口已保留，后续可继续完善加权融合策略。

---

## 📚 文档地图

| 主题 | 文档 |
|---|---|
| V2 多候选工作流（22 状态） | [docs/agent_workflow.md](docs/agent_workflow.md) |
| LangGraph 重构报告 | [docs/langgraph_refactor_report.md](docs/langgraph_refactor_report.md) |
| 确定性两层路由 | [docs/hierarchical_routing.md](docs/hierarchical_routing.md) |
| 候选规划与评分 | [docs/multi_candidate_planning.md](docs/multi_candidate_planning.md) |
| 候选评分细节 | [docs/candidate_scoring.md](docs/candidate_scoring.md) |
| 候选规划知识库 | [docs/rag_context.md](docs/rag_context.md) |
| 图像分析 | [docs/image_analysis.md](docs/image_analysis.md) |
| 模型适配器契约 | [docs/model_adapter.md](docs/model_adapter.md) |
| 模型接入清单 | [docs/model_integration.md](docs/model_integration.md) |
| Real-ESRGAN 接入 | [docs/realesrgan_integration.md](docs/realesrgan_integration.md) |
| Real-ESRGAN + IQA 验收 | [docs/realesrgan_iqa_acceptance_report.md](docs/realesrgan_iqa_acceptance_report.md) |
| IQA 集成 | [docs/iqa_integration.md](docs/iqa_integration.md) |
| 后处理工作流 | [docs/postprocess_workflow.md](docs/postprocess_workflow.md) |
| 多模态 AI 供应商 | [docs/multimodal_ai.md](docs/multimodal_ai.md) |
| 模型环境隔离 | [docs/model_environment_isolation.md](docs/model_environment_isolation.md) |
| 安全 / 上传加固 | [docs/security.md](docs/security.md) |
| 故障排查 | [docs/troubleshooting.md](docs/troubleshooting.md) |
| V2.2 → V2.3 重构报告 | [docs/VisionRestore_Agent_V2.2_重构与轻量知识增强报告.docx](docs/VisionRestore_Agent_V2.2_重构与轻量知识增强报告.docx) |
| 开发日志 | [docs/development_log.md](docs/development_log.md) |

---

## 🛣️ 版本路线

- ✅ **V2.0** — LangGraph 多候选编排 + 持久化 checkpoint
- ✅ **V2.1** — 分层路由 + 确定性评分 + IQA 集成
- ✅ **V2.2** — 区域约束 + 残差分析 + 手动后处理控制器
- ✅ **V2.3** — 轻量知识增强 + NAFNet / Real-ESRGAN 后处理 + 厂商无关多模态 AI
- 🔜 **V2.4** — WebP / AVIF 管线、tile 超分加权融合、自动化测试向 20 类扩展

---

## 🤝 贡献

欢迎 PR 与 Issue。开工前请：

1. 先读 [docs/architecture.md](docs/architecture.md) 和 [docs/agent_workflow.md](docs/agent_workflow.md)
2. 使用 [`.github/ISSUE_TEMPLATE/`](.github/ISSUE_TEMPLATE/) 里的模板
3. 确保 `./.venv/Scripts/python.exe -m pytest` 全绿 + `npm.cmd run build --prefix apps\web` 通过

新增 model worker 时，请严格遵守 [docs/model_adapter.md](docs/model_adapter.md) 的契约：worker 必须读**原始输入**，输出**真实结果**，由 `CandidateEvaluator` 评分——**不要**让 planner 替代 evaluator 决定 final_score。

---

## 📜 许可证

[MIT](LICENSE)。第三方模型代码与权重（Retinexformer、SCI、Zero-DCE 等）的许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)，上游的「非商业 / 学术研究」条款在源码层仍然适用。

---

## 🙏 致谢

本项目站在以下开源工作的肩膀上。如使用对应权重，请引用原论文：

- **Retinexformer** — Cai et al., *Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement*, ICCV 2023
- **SCI** — Ma et al., *Toward Fast, Flexible, and Robust Low-Light Image Enhancement*, CVPR 2022
- **Zero-DCE** — Guo et al., *Zero-Reference Deep Curve Estimation for Low-light Image Enhancement*, CVPR 2020
- **DarkIR · HVI-CIDNet · FLOL · NAFNet · Real-ESRGAN · LPDM · MambaIR** 见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)

---

## 🚧 已知局限

- 健康检查结果当前按请求返回，尚未持久化到每个 checkpoint 状态
- UI 已按参考图重做布局和背景，但拖动分割、同步缩放和平移仍可继续增强
- 自动化测试已覆盖核心链路，但尚未扩展到说明中列出的全部 20 类测试

---

<div align="center">

<sub>在 Windows 上用心打造 · RTX 3060 12GB 已验证 · 图像永远不离开你的机器</sub>

</div>
