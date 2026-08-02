# APE-Ti WSL2 环境配置与严格推理验证

日期：2026-08-01（Asia/Shanghai）

## 结论

APE-Ti 已在本机 WSL2 + Ubuntu 22.04 独立环境中完成真实 GPU 推理验证，
不再处于“仅下载权重、运行环境阻塞”的状态。

## 验证结果

| 项目 | 结果 |
|---|---|
| APE Python/CUDA 扩展 | 通过 |
| Detectron2 / Detrex 导入 | 通过 |
| Python 依赖一致性 | `No broken requirements found` |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| checkpoint 兼容 | 0 missing / 0 unexpected / 0 incorrect shape |
| 严格等价装载 | 通过 |
| 测试图片 | APE 仓库自带 `Pisa.jpg` |
| 测试文本 | `the leaning tower` |
| 输出候选 | 1 个 bbox |
| 构建耗时 | 7.35 秒 |
| checkpoint 装载耗时 | 15.70 秒 |
| 单图推理耗时 | 12.35 秒 |
| 峰值显存 | 6,809,270,784 bytes（约 6.34 GiB） |

完整机器结果：

```text
outputs/sota_model_smoke/ape_ti/wsl_strict_smoke.json
```

## 诊断与修复记录

1. WSL 默认约 7.3 GiB RAM / 2 GiB swap 导致 Linux OOM kill；已调整为
   10 GB RAM / 12 GB swap。
2. APE 的 EVA02 文本包装器会临时构建随后删除的巨大视觉塔；验证器跳过该
   无用构建后，加载阶段不再使用 swap。
3. cuDNN 需要 `/usr/lib/wsl/lib`；已固化到 `aic-ape` 的
   `LD_LIBRARY_PATH`。
4. 官方 README 允许禁用 xFormers；验证器使用该 PyTorch fallback。
5. PyTorch 1.12 + CUDA 11.6 的 NVRTC 不识别 RTX 4060 的 `sm_89`；Linux
   APE 副本采用数学等价的一行兼容补丁绕开小张量归约 JIT。
6. EVA 文本特征为 FP16；前向使用选择性 autocast，避免 Half/Float 冲突。

## 边界

本次环境验证只证明 APE-Ti 官方权重能在本机完整装载并执行推理，不代表它在
AIC 测试集上优于 Florence、MM-Grounding-DINO-T 或 LLMDet。后续实验决策已
调整为：不使用外部验证分数筛选这三个模型，分别直接完成 AIC 全量零训练推理，
由平台 ACC 决定实际迁移能力；本报告不预判平台结果。
