# AIC APE-Ti WSL2 使用说明

## 这是什么

WSL2 让 Windows 在一个轻量 Linux 环境中运行 APE 官方依赖。它不是双系统，
不会改变 Windows 的启动方式。Linux 文件保存在 `D:\WSL\Ubuntu-22.04`，
模型权重继续保存在 Windows 的 `D:\AI_Models`。

当前固定环境：

- WSL 2.7.11.0
- Ubuntu 22.04.5 LTS
- Linux 用户 `aicuser`
- Python 3.10 环境 `aic-ape`
- PyTorch 1.12.1 + CUDA 11.6
- APE commit `8c4920e2014d4818ec3ecafb7b526d896a659d1c`
- Detectron2 commit `017abbf`
- Detrex commit `776058e`

## 最简单的启动方式

双击：

```text
D:\12525\Documents\pytorch\baseline_v0\tools\wsl\start_aic_ape.cmd
```

看到 `AIC APE environment is ready.` 后，终端已经位于：

```text
/home/aicuser/aic/APE
```

## 一键自检

在 Windows PowerShell 中运行：

```powershell
& "D:\12525\Documents\pytorch\baseline_v0\tools\wsl\verify_aic_ape.ps1"
```

自检会检查 APE 扩展、Detectron2、Detrex、CUDA、cuDNN 和 Python 依赖。

## 常用规则

- Windows 的 `D:\AI_Models` 在 Linux 中是 `/mnt/d/AI_Models`。
- 输入 Linux 密码时屏幕不会显示星号，这是正常行为。
- 退出当前 Linux 终端：`exit`。
- 完全停止 WSL 并释放内存：在 Windows PowerShell 运行 `wsl --shutdown`。
- 不要在 WSL 中安装 NVIDIA Linux 显卡驱动；WSL 使用 Windows 已有驱动。
- 不要在 `base` 环境安装 APE 包；始终使用独立的 `aic-ape` 环境。

## 资源配置

`C:\Users\12525\.wslconfig` 当前设置：

```ini
[wsl2]
memory=10GB
swap=12GB
swapFile=D:\\WSL\\wsl-swap.vhdx
localhostForwarding=true

[experimental]
autoMemoryReclaim=gradual
sparseVhd=true
```

修改该文件后必须运行 `wsl --shutdown` 才会生效。

## APE-Ti 兼容处理

基础验证器位于：

```text
tools/wsl/run_ape_ti_smoke.py
```

它只做三项运行兼容处理，不改变 checkpoint 数值：

1. 跳过 APE 构建后立即删除的 EVA-CLIP 图像塔，降低 CPU 内存峰值；
2. 按 APE 官方说明关闭 xFormers，使用 PyTorch 路径；
3. 推理时启用 FP16 autocast，以匹配官方 EVA 文本分支精度。

Linux APE 副本还包含一个 RTX 4060/CUDA 11.6 兼容补丁：将二维尺寸的
`prod()` 改写为完全等价的 `height * width`，避免旧 NVRTC 尝试编译不支持的
`sm_89` 归约内核。Windows 原始 APE 仓库没有被修改。

正式 AIC 全量入口位于：

```text
tools/wsl/run_ape_ti_full_detection.py
```

该入口只输出 AIC 所需的 bbox，因此关闭了不参与比赛提交的 semantic、
panoptic 和 mask 后处理，并跳过未使用的全分辨率 mask 分配。这个处理不改变
bbox logits、候选分数或排序；已在同一 AIC Query 上与原始路径逐值核对 Top-1
bbox 和 score。正式运行使用原始 Query、原生最高分 Top-1、FP16 autocast、
batch size 1，并保存权重、配置、代码和输入的 SHA-256 指纹以支持安全续跑。

补丁的可重放文件保存在：

```text
tools/wsl/ape_ti_rtx4060_cuda116.patch
```
