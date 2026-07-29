# 2026 AIC 多模态视觉定位 Baseline

第一版可完整运行的 RGB-only 基线：

```text
Visible + Query → Florence-2-large-ft → 归一化 bbox → 提交 JSON/ZIP
```

当前版本：`v0.1.2`。正式初赛数据没有 bbox，只用于推理，不用于训练。

## 安装

```powershell
conda activate aic-baseline
python -m pip install -e .
```

已验证环境：

- Python 3.11
- PyTorch 2.5.1+cu121
- Transformers 4.57.6
- RTX 4060 Laptop GPU

## 配置

复制公开模板，本机路径只写入 `.local.yaml`：

```powershell
Copy-Item configs\florence2_rgb_only.example.yaml configs\florence2_rgb_only.local.yaml
Copy-Item configs\official_sample.example.yaml configs\official_sample.local.yaml
```

编辑两个本地配置中的数据集、模型和输出路径。本地配置已被 Git 忽略。

## 验证与推理

```powershell
# 自动化测试
python -m pytest -q

# 检查 9,555 条 Query 及文件路径
python -m aic_baseline.cli validate --config configs\florence2_rgb_only.local.yaml

# 官方单样例 sanity check
python -m aic_baseline.cli sample --config configs\official_sample.local.yaml

# 10 条 smoke test
python -m aic_baseline.cli infer `
  --config configs\florence2_rgb_only.local.yaml `
  --limit 10 `
  --output-dir outputs\smoke_10

# 全量推理、断点续跑并生成提交文件
python -m aic_baseline.cli infer `
  --config configs\florence2_rgb_only.local.yaml `
  --build-submission `
  --resume
```

只有完整处理全部 Query 后才允许生成提交 ZIP。

## 数据审计

```powershell
python tools\audit_dataset.py `
  --dataset-root "D:\path\to\aic-dataset"
```

核心结论见 [数据集审计摘要](reports/dataset_audit_summary.md)。

## 项目结构

```text
configs/   公开配置模板
src/       数据、模型、推理和提交代码
tests/     自动化测试
tools/     数据审计工具
reports/   实验与审计结论
outputs/   本地输出，不上传 Git
```

当前限制：尚未使用 Infrared、Depth、Query 改写或候选框重排序。
