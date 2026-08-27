# VisionRestore Agent V2.3 决策评分修正与验证报告

日期：2026-08-24

## 1. 结论

本轮已完成模型选择机制和真实结果评价机制的增量修正。项目仍为单 Agent、LangGraph 多候选闭环架构，没有引入 Multi-Agent、完整 RAG、新模型算法或新权重。

真实固定 30 张 LOLv2 Real Test 复验结果：

| 指标 | 固定 Retinexformer | Agent | Agent 增量 |
|---|---:|---:|---:|
| PSNR | 22.749873 dB | 23.565943 dB | +0.816071 dB |
| SSIM | 0.838730 | 0.852388 | +0.013658 |
| FinalScore | 90.218333 | 90.608667 | +0.390333 |

PSNR 95% CI 为 `[0.030783, 1.792089]`，SSIM 95% CI 为 `[0.002749, 0.026559]`，两者均高于零。本轮 Agent 任务成功 30/30，固定基线成功 30/30。

## 2. 保持不变的架构边界

- 保留 `LangGraphEnhancementAgentV2` 主图、候选 `Send` 扇出、SQLite checkpoint 和 interrupt/resume。
- 保留 `CandidatePlanner -> MultiCandidateExecutor -> CandidateEvaluator -> CandidateRanker -> ResultSelector`。
- 所有增强候选仍从同一原始输入独立执行，不串联增强模型。
- `PlanningScore` 仅用于执行前候选规划，`FinalScore` 仅来自真实模型输出。
- Knowledge 只向 LLM 提供模型能力参照，不作为独立加分项。
- 不按模型名称修改 FinalScore，不给 DarkIR 奖励，也不给 HVI-CIDNet 惩罚。

## 3. PlanningScore 修正

有效外部语义评分存在时：

`PlanningScore = 0.5 × LocalScore + 0.5 × LLMScore`

外部 LLM 被禁用、不可用、低置信、格式错误或本地校验失败时：

`PlanningScore = LocalScore`

其中：

- LocalScore 为 0-100 的本地模型先验和输入退化匹配评分。
- LLMScore 为 0-100 的模型家族语义适配评分。
- LLM 使用 allowlisted `model_roles` Knowledge 作为能力定义标准。
- LLM 不选择 checkpoint；自动模式使用模型家族配置中的默认健康 checkpoint。
- 模型健康和硬件资源属于执行门禁，不再伪装成质量分数。

## 4. FinalScore 通用修正

最终公式仍保持四层：

`FinalScore = ImageQualityScore + RestorationScore + ConstraintScore + StabilityScore`

本轮修正集中在无参考本地评价的可解释性：

1. 增加目标亮度恢复，区分“真正恢复到合理亮度”和“仅比原图变亮”。
2. 使用输出图像的绝对色偏，而不只判断色偏是否比输入继续恶化。
3. 使用配置化有效细节曲线，同时惩罚过度平滑和异常高频。
4. 降低单纯平滑带来的噪声控制奖励，避免去噪分掩盖亮度、色彩和细节损失。
5. 增强稳定性层中的欠曝、绝对色偏、过曝、伪影和数值完整性检查。
6. 所有阈值和权重均位于 `config/scoring_rules.yaml`，评价代码不识别模型名称。

## 5. 修正前后对比

同一固定 30 张样本、同一候选模型和固定 Retinexformer 基线：

| 项目 | 修正前真实运行 | 修正后真实运行 |
|---|---:|---:|
| Agent 成功 | 30/30 | 30/30 |
| Retinexformer 入选 | 20 | 21 |
| DarkIR 入选 | 5 | 8 |
| HVI-CIDNet 入选 | 5 | 1 |
| PSNR 增量 | -0.580621 dB | +0.816071 dB |
| SSIM 增量 | -0.021311 | +0.013658 |
| FinalScore 上升但 PSNR/SSIM 同降 | 4 | 1 |

DarkIR 的 8 个入选样本平均相对固定 Retinexformer：

- PSNR：`+3.070321 dB`
- SSIM：`+0.057880`
- SSIM 提升：8/8
- PSNR 与 SSIM 同时下降：0/8

这与人工观察到 DarkIR 在本测试域部分图像中恢复更好的判断一致，但运行时选择仍由统一输出指标产生。

## 6. 历史 100 张候选离线复核

对历史已保存的 100 张真实候选输出重新评分，不重新运行模型：

- Retinexformer 31 张、DarkIR 67 张、HVI-CIDNet 2 张。
- 相对固定 Retinexformer，PSNR `+0.510261 dB`。
- 相对固定 Retinexformer，SSIM `+0.027875`。

该结果仅用于评价规则敏感性检查，不能替代新的 100 张真实推理实验。

## 7. 后处理闭环修正

真实复验暴露了一个 LangGraph 闭环问题：当用户确认 NAFNet 后，若后处理依赖缺失或执行失败，旧逻辑重新进入同一个去噪确认节点，可能形成重复循环。

修正后：

- 保留后处理前最佳增强结果。
- 记录执行失败和错误来源。
- 如果仍需要 SR，则进入 SR 确认；否则直接完成。
- 不重复请求同一个失败的后处理操作。
- 已保留重新评价、分数下降 rollback 和 checkpoint resume 机制。

## 8. 工程与依赖修正

- 为 Retinexformer 和隔离 worker 子进程统一传递项目本地 `data/python_packages`。
- 增加 `einops==0.6.1` 可复现依赖声明。
- 修复评测报告中固定写死 100 张、固定显著性结论、固定候选失败原因和空值格式化问题。
- 评测驱动现在按实际样本数、置信区间和真实失败记录生成结论。

## 9. 自测结果

- CandidateEvaluator/Evaluator 定向测试：14/14 通过。
- PostprocessController/LangGraph 定向测试：11/11 通过。
- 项目完整 pytest：95/95 通过。
- Retinexformer、HVI-CIDNet、DarkIR 真实候选冒烟：均成功。
- 固定 30 张真实 Agent 闭环：30/30 成功。
- 固定基线：30/30 成功。
- 候选级失败：1/90；任务级容错和固定基线均保持有效。

## 10. 结论边界与遗留问题

1. 本轮真实评测使用本地分析模式，因此验证了 LocalScore fallback 和 FinalScore 闭环；LLM 50/50 融合由单元/集成测试覆盖，但没有在这 30 张中调用外部多模态 API。
2. 仍有 1 张近分误选：HVI-CIDNet 的 FinalScore 仅高 0.28，但 PSNR/SSIM 同时小幅下降。没有使用模型专属规则隐藏该问题。
3. NAFNet 当前隔离运行环境仍缺少 `lmdb`。失败后闭环已能安全继续，但本轮没有覆盖成功去噪后的重新评价和 rollback。
4. Real-ESRGAN 在本次成对 PSNR/SSIM 评测中按策略跳过，以保证输出和 Normal 图像严格同尺寸。
5. 历史 100 张结果是离线重评分；如果需要新的 100 张统计结论，应重新执行 100 张真实推理。

## 11. 评测产物

- 真实 30 张逐图结果：`data/eval/lolv2_v23_scorefix_fixed30_seed20260803_20260824/per_image_metrics.csv`
- 真实 30 张汇总：`data/eval/lolv2_v23_scorefix_fixed30_seed20260803_20260824/summary.json`
- 真实 30 张报告：`data/eval/lolv2_v23_scorefix_fixed30_seed20260803_20260824/report.md`
- 历史 100 张离线重评分：`data/eval/lolv2_real_test100_v23_rescore_20260823/summary.json`
- 评分敏感性分析：`data/eval/lolv2_v23_fixed30_seed20260803_20260823/score_calibration_grid.json`

最终结论：VisionRestore Agent 已从“多个增强模型自动选择”推进为“依据输入状态和模型能力规划候选，通过真实输出评价选择结果，并在后处理失败或退化时安全保留/回滚”的单 Agent 闭环视觉恢复系统。
