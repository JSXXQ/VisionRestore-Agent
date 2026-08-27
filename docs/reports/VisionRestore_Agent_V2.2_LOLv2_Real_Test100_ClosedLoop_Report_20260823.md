# VisionRestore Agent V2.2：LOLv2 Real Test 100 张闭环评测报告

## 1. 结论摘要

本轮结论为：**闭环 Agent 获得部分且有统计依据的提升，但不是全面提升。**

- FinalScore：79.1494 → 80.3852，平均 +1.2358，95% CI [0.971075, 1.545213]，显著提升。
- PSNR：22.789753 → 22.798507 dB，平均 +0.008754 dB，95% CI [-0.80148, 0.831636]，区间跨 0，不能判定显著提升。
- SSIM：0.838308 → 0.853023，平均 +0.014715，95% CI [0.004613, 0.024577]，显著提升。
- 因此，闭环在结构相似度和内部质量评分上有效；在像素级保真度上与固定基线总体持平。

## 2. 测试设计

- 数据集：`LOLv2 Real_captured Test`。
- 配对：`lowNNNNN.png -> normalNNNNN.png`，有效配对 100 组。
- 抽样：测试目录恰好 100 组，因此覆盖全部样本；执行顺序按固定随机种子 20260823 打乱。
- Agent：`LangGraph V2.2 multi-candidate closed loop`，本地分析模式，不调用外部多模态 API。
- 固定基线：`retinexformer/lol_v2_real`。
- Agent 与基线使用相同 V2.2 CandidateEvaluator；PlanningScore 和 KnowledgeAdjustment 不参与最终质量评分。
- 参考指标：使用配对 Normal 图像计算 PSNR/SSIM。

## 3. 运行完整性

- Agent 任务成功：100/100 （100.0%）。
- 基线成功：100/100。
- 候选失败：4/300（约 1.33%）；均为子进程退出码 `3221226505 / 0xC0000409`，HVI-CIDNet 2 次、DarkIR 2 次。其余候选保证了任务级 100% 成功。
- 总耗时：3609.39 秒；单样本平均 36.0939 秒。

## 4. 模型选择与真实收益

| 最终模型 | 样本数 | FinalScore Δ | PSNR Δ(dB) | SSIM Δ | PSNR胜 | SSIM胜 | 两项同时下降 |
|---|---:|---:|---:|---:|---:|---:|---:|
| darkir | 58 | 1.486034 | 0.664358 | 0.041679 | 31 | 51 | 7 |
| hvi_cidnet | 19 | 1.967895 | -1.981966 | -0.049783 | 4 | 4 | 13 |
| retinexformer | 23 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 |

- 发生模型切换：77 张；切换样本平均 PSNR +0.011369 dB、SSIM +0.019110。
- PSNR/SSIM 同时提升 33 张；同时下降 20 张；方向混合 24 张；与基线相同 23 张。
- 有 20 张出现 FinalScore 上升但 PSNR/SSIM 同时下降，说明无参考评价器与 LOLv2 配对真值存在排序偏差。
- DarkIR 带来主要正收益；HVI-CIDNet 在本测试域中平均退化，当前选择条件需要收紧。

## 5. 后处理与回滚覆盖

- NAFNet 去噪建议：0 次。
- Real-ESRGAN 超分建议：100 次。
- 为保持 Agent 输出与 Normal 真值严格同尺寸，本轮按评测策略跳过 SR。
- 实际后处理：0 次；回滚触发/成功：0/0。
- 因而本轮验证了候选规划—真实执行—评价—选择闭环，但没有覆盖后处理后的重新评价与 rollback 分支。

## 6. 主要问题与下一步建议

1. 优先校准 CandidateEvaluator：降低对 HVI-CIDNet 色彩/亮度收益的过度奖励，并用 LOLv2 参考指标做离线相关性分析。
2. 为 HVI-CIDNet 增加保守门槛：只有颜色偏移证据足够强，且相对 Retinexformer 的 FinalScore 优势超过安全阈值时才允许替换。
3. 增加保守回退：候选差值接近时优先 Retinexformer；同时保留当前按真实结果选择、不让 PlanningScore 进入最终质量评分的边界。
4. 单独建立后处理测试集：使用无需同尺寸比较的无参考指标或先将 SR 输出规范化回原尺寸，验证重新评价与自动 rollback。
5. 排查候选 worker 的 `0xC0000409` 偶发退出；当前任务级容错有效，但不应长期接受 1.33% 的候选失败率。

## 7. 产物

- 逐图 CSV：`E:\codex_project\vision\VisionRestore-Agent - new\data\eval\lolv2_real_test100_v22_seed20260823\per_image_metrics.csv`
- 汇总 JSON：`E:\codex_project\vision\VisionRestore-Agent - new\data\eval\lolv2_real_test100_v22_seed20260823\summary.json`
- 输出目录：`E:\codex_project\vision\VisionRestore-Agent - new\data\eval\lolv2_real_test100_v22_seed20260823`

结论边界：本报告只把真实输出的 FinalScore、PSNR、SSIM 作为质量依据；PlanningScore 与知识调整只用于执行前规划。