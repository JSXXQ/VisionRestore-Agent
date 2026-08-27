# VisionRestore Agent

多模型协同低照度图像增强 Agent。本项目只处理普通 RGB 静态图像，不包含事件相机、视频、连续帧、音频、超分辨率、插帧或训练平台功能。

## 当前已接入的本地模型

- Retinexformer：LOL-v2-real、SDSD-indoor、SDSD-outdoor、NTIRE。
- SCI：easy、medium、difficult。
- Zero-DCE：Epoch99，作为经典轻量基线和最后兜底；上游 README 标明非商业/学术研究用途。

模型源码统一放在 `third_party/<model>/`，运行时权重统一通过 `weights/<model>/` 引用。`config/models.local.yaml` 使用项目相对路径并已加入 `.gitignore`，便携示例为 `config/models.example.yaml`。完整约定见 `docs/model_file_layout.md`。

## 已验证真实推理

使用 `E:\anconda\envs\pytorch\python.exe`，PyTorch 2.5.1，CUDA 12.1，RTX 3060 12GB：

- Retinexformer LOL-v2-real：通过。
- Retinexformer SDSD-outdoor：通过。
- SCI medium：通过。
- Zero-DCE Epoch99：通过。

输出图像经过尺寸断言，保持输入宽高一致。

## Agent 作用

Agent 包含 ImageAnalyzer、IntentParser、HardwareInspector、ModelRegistry、HierarchicalRouter、模型适配器、QualityEvaluator、备用策略和报告生成。它不是固定调用单个模型。

## 分层路由

第一层选择模型架构：Retinexformer、SCI、Zero-DCE。第二层选择权重：Retinexformer 根据用户明确场景和优先级选择 LOL-v2-real/SDSD/NTIRE；SCI 根据亮度分位数、暗像素比例和动态范围选择 easy/medium/difficult。规则在 `config/routing_rules.yaml`。

如果场景无法判断，显示“场景未知”，默认 LOL-v2-real，不用亮度统计伪造室内/室外语义。

## Windows 启动

```powershell
Set-Location -LiteralPath "<VisionRestore-Agent 项目目录>"
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start.ps1
```

前端页面：http://127.0.0.1:5173  
本地 FastAPI：http://127.0.0.1:8000  
API 文档：http://127.0.0.1:8000/docs

默认工作流不需要任何云端 API Key。推理使用 `config/models.local.yaml` 指向的本地 PyTorch 环境、本地模型源码和本地权重；IntentParser 使用本地规则。

多模态 AI 是可选功能，并且不是 OpenAI 专用。后端现在提供 `disabled`、`openai`、`openai_compatible`、`anthropic`、`gemini` 供应商槽位。其中 `openai_compatible` 可以连接任何兼容 OpenAI `/chat/completions` 格式的厂商 Base URL。默认仍是 `disabled`，外部 API 只做语义分析和路由建议，不直接控制本地模型推理。

停止：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/stop.ps1
```

## 模型验证

首次整理或恢复本地模型文件时运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/standardize_model_layout.ps1
```

然后执行注册与真实小图健康检查：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/validate_models.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_smoke_test.ps1
```

## CLI

```powershell
.\.venv\Scripts\python.exe -m visionrestore.cli analyze --input data\cache\test_images\low_light.png
.\.venv\Scripts\python.exe -m visionrestore.cli enhance --input input.jpg --request "自然增强暗部，保护高光，质量优先" --mode auto --output output.png
.\.venv\Scripts\python.exe -m visionrestore.cli enhance --input input.jpg --model retinexformer --weight lol_v2_real --output output.png
.\.venv\Scripts\python.exe -m visionrestore.cli compare --input input.jpg --output output.png --candidates retinexformer:lol_v2_real retinexformer:sdsd_outdoor sci:difficult
```

## API

核心接口位于 `/api/v1/`，包括 health、system、models、model weights、intent parse、images、tasks、history、files、settings。任务创建后立即返回 `task_id`，前端通过轮询或 WebSocket 查看状态。

可选多模态 AI 接口位于 `/api/v1/ai/`：

- `GET /api/v1/ai/providers`
- `POST /api/v1/ai/providers/{provider_id}/health-check`
- `POST /api/v1/ai/analyze`
- `GET /api/v1/ai/settings`
- `PUT /api/v1/ai/settings`
- `POST /api/v1/ai/test`

这些接口不会把完整 API Key 返回给前端；未启用或未配置时会回退到本地规则分析。

也可以在前端“系统设置”页直接配置多模态 API。Key 只写入本机 `.env`，页面刷新或接口响应只显示是否已配置，不回显完整密钥。

## 评价指标

第一版没有 GT 上传，因此不计算 PSNR、SSIM 或 LPIPS。系统只显示无参考指标：亮度变化、暗像素变化、过曝变化、对比度、动态范围、清晰度、噪声、色偏、熵、结构保持估计、耗时、显存和文件大小。

无参考指标只能作为辅助判断，不能完全替代人工主观评价。

## 显存不足和分辨率规则

本项目不是超分辨率项目。用户保存的结果必须与输入图像宽高一致。模型内部如需填充，由 adapter/runner 处理，推理后裁剪回原始尺寸。高分辨率分块推理接口已保留，后续可继续完善加权融合策略。

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest
npm.cmd run build --prefix apps\web
```

当前测试：10 passed；前端构建通过。

## 已知局限

- 健康检查结果当前按请求返回，尚未持久化到每个 checkpoint 状态。
- UI 已按参考图方向重做布局和背景，但拖动分割、同步缩放和平移仍可继续增强。
- 自动化测试已覆盖核心链路，但尚未扩展到说明中列出的全部 20 类测试。
