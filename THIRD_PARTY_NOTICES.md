# Third Party Notices

This project integrates local source trees and local pretrained weights supplied outside Git. Large model files and user images are ignored by Git.

| Model | Local source | Official repository | Local version | License | Weights | Commercial use | Patches |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Retinexformer | `third_party/retinexformer` | https://github.com/caiyuanhao1998/Retinexformer | local source, commit not available from extracted archive unless `.git` exists | MIT License, copyright Yuanhao Cai | `weights/retinexformer/` | MIT permits commercial use subject to license text | No upstream source edits |
| SCI | `third_party/sci/CVPR` | https://github.com/vis-opt-group/SCI | local source, commit not available from extracted archive unless `.git` exists | MIT License, copyright Tengyu Ma | `weights/sci/` | MIT permits commercial use subject to license text | No upstream source edits |
| Zero-DCE | `third_party/zero_dce/Zero-DCE_code` | https://github.com/Li-Chongyi/Zero-DCE | local source, commit not available from extracted archive unless `.git` exists | Attribution-NonCommercial 4.0 / academic research purpose notice in README | `weights/zero_dce/epoch99.pth` | Non-commercial / academic research only per upstream README | No upstream source edits |

Zero-DCE is shown as a classic lightweight baseline and final fallback. The UI and docs must not describe it as state of the art or commercially unrestricted.

## Real-ESRGAN

Real-ESRGAN is referenced from local source under `third_party/realesrgan`, with runtime checkpoints under `weights/realesrgan/`. The project configuration records BSD-3-Clause for the local source; verify dependency licenses before distribution.

## PyIQA

PyIQA is an optional runtime dependency for no-reference IQA metrics. It is not vendored by this project.
