# Qwen3-VL-30B-A3B-FP8 云端环境配置记录与避坑（2026-08-10）

## 适用场景

下次复跑以下模型前，先看本文件：

```text
Qwen/Qwen3-VL-30B-A3B-Instruct-FP8
Featurize RTX 5090 32GB
vLLM OpenAI-compatible API
AIC 9,555 条全量推理
```

## 已验证硬件

| 项目 | 结果 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 |
| 显存 | 32GB |
| 是否能跑 FP8 | 可以 |
| 是否建议跑 BF16 | 不建议；单卡 32GB 基本放不下 |
| 推理速度 | 全量约 3.5～4 小时量级，受输出长度和拒答重试影响 |

## 已验证启动参数

核心点：禁用 FlashInfer sampler，使用 Triton MoE 和 Cutlass linear。

```bash
unset CUDA_HOME
unset CUDA_PATH
export VLLM_USE_FLASHINFER_SAMPLER=0

python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
  --served-model-name qwen3vl30b_fp8 \
  --host 127.0.0.1 \
  --port 8000 \
  --trust-remote-code \
  --gpu-memory-utilization 0.92 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --cpu-offload-gb 16 \
  --moe-backend triton \
  --linear-backend cutlass
```

## 已踩过的坑

### 1. 中文路径在远端 shell 中变成 `?????`

现象：

```text
FileNotFoundError: ... '/home/featurize/aic_cloud/aic_data/?????/queries/queries.json'
```

解决：

```bash
ln -s /home/featurize/aic_cloud/aic_data/初赛数据集-基于大模型的多模态视觉理解与推理 \
  /home/featurize/aic_cloud/aic_data/aic_round1
```

后续统一使用：

```text
/home/featurize/aic_cloud/aic_data/aic_round1
```

### 2. `--limit-mm-per-prompt image=1` 与 vLLM 版本不兼容

现象：

```text
Value image=1 cannot be converted to loads
```

尝试 JSON 写法时又可能被 shell 引号吃掉，变成 `{image:1}`。

解决：

```text
本任务每条只输入 1 张 visible RGB 图，不需要该参数，直接删除。
```

### 3. FlashInfer sampler JIT 编译失败

现象之一：

```text
RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
```

尝试把 `CUDA_HOME` 指向 conda 内 nvcc 后，又出现：

```text
CUDA compiler and CUDA toolkit headers are incompatible
NVCC compilation failed
```

解决：

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
unset CUDA_HOME
unset CUDA_PATH
```

### 4. 5090 32GB 显存边界较紧

模型可启动，但需要限制并发和上下文：

```text
--max-num-seqs 1
--max-model-len 4096
--cpu-offload-gb 16
--gpu-memory-utilization 0.92
```

启动后显存使用约 28GB/32GB。

### 5. 生成式 bbox 会有极少数拒答或无效框

典型输出：

```text
There is no ...
[]
[0.0, 0.0, 0.0, 0.0]
```

解决策略：

```text
运行阶段允许 center fallback 保证不中断；
最终提交阶段不要使用 center fallback；
用历史最佳 Qwen3-VL-8B-Instruct 0.7582 的 bbox 替换 fallback 样本。
```

本轮最终替换：

```text
13 / 9555
```

## 下次复跑最短流程

确认 vLLM 服务：

```bash
curl -s http://127.0.0.1:8000/v1/models
```

继续或重跑全量：

```bash
cd /home/featurize/aic_cloud/aic-multimodal-visual-grounding
bash tools/cloud/wait_and_finalize_qwen30b_fp8_cascade.sh
```

最终云端平台包：

```text
/home/featurize/aic_cloud/platform_upload_ready/AIC_Qwen3_VL_30B_A3B_FP8_8BInstructFallback_20260809.zip
```

## 不要重复浪费时间的点

- 不要再从 BF16 原权重开始尝试单卡 32GB。
- 不要再使用 `--limit-mm-per-prompt image=1`。
- 不要再启用 FlashInfer sampler。
- 不要用中文路径直接传给远端 shell。
- 不要把 center fallback 直接作为最终提交。
