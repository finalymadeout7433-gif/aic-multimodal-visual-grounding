# 外部训练数据准备报告

## 结论

`D:\AIC赛题一数据集` 已完成下载校验、解压、标注引用核对、统一 JSONL、图像隔离
split 和抽样可视化。数据本体不进入 GitHub。

## 归档校验

| 归档 | 字节数 | ZIP 条目 | SHA-256 | CRC |
|---|---:|---:|---|---|
| COCO train2014 | 13,510,573,713 | 82,784 | `EDE4087E640BDDBA550E090EAE701092534B554B42B05AC33F0300B984B31775` | 通过 |
| SUN RGB-D | 6,885,481,608 | 410,737 | `1A6DBF2A1C9044C4805A35EE648D616EA39A231FD5BD6F77E84CD2B8287FE41C` | 通过 |

解压后文件审计：

- COCO：82,783 个文件，缺失 0、大小错误 0、命名冲突 0。
- SUN RGB-D：285,925 个文件，缺失 0、大小错误 0、命名冲突 0。
- SUN RGB-D 有 24,226 个状态/元数据文件名包含 Windows 非法字符，工具将其确定性
  替换为下划线；RGB/Depth 训练引用不受影响。

## 统一记录

| 数据集 | 可用表达 | RGB 缺失 | Depth 缺失 | 非法 bbox |
|---|---:|---:|---:|---:|
| RefCOCO | 142,210 | 0 | — | 0 |
| RefCOCO+ | 141,564 | 0 | — | 0 |
| RefCOCOg | 95,010 | 0 | — | 0 |
| gRefCOCO | 221,670 | 0 | — | 0 |
| SUN-Spot | 8,010 | 0 | 0 | 0 |

gRefCOCO 的 37,166 条 no-target 表达不适合 AIC 单目标 bbox 监督，已从可训练记录中
排除；多目标表达保留 `target_count`，bbox 使用目标集合的最小外接矩形。

## 第一阶段核心 RGB split

来源：RefCOCO、RefCOCO+、RefCOCOg。跨数据集按 COCO image ID 全局隔离，优先级为
holdout > validation > train。

| split | 表达数 | 图像数 |
|---|---:|---:|
| train | 287,604 | 24,407 |
| validation | 37,128 | 2,567 |
| holdout | 54,052 | 3,982 |

不同 split 的图像交集为 0。训练和调参只使用 train/validation，holdout 只做最终
独立检查。

## 清单格式

本机生成于：

```text
D:\AIC赛题一数据集\manifests\processed
```

核心文件：

```text
rgb_core_v1_train.jsonl
rgb_core_v1_validation.jsonl
rgb_core_v1_holdout.jsonl
```

每条记录包含相对图像路径、Query、像素 bbox、归一化 xyxy bbox、原始 split、
全局 split、数据集名、image ID、annotation ID 和目标数量。SUN-Spot 额外包含
Depth 相对路径。所有生成的 JSONL 大小与 SHA-256 记录在同目录的
`manifest_sha256.json`。

## 可视化核验

预览位于 `D:\AIC赛题一数据集\previews`。已核对：

- RefCOCO：`the lady with the blue shirt`；
- gRefCOCO：`giraffe on left`；
- SUN-Spot：带叶片图案的 shower curtain。

三者的 Query、图像与 bbox 均对应合理。抽样核验不能替代全量语义人工审查，但结合
全量路径、尺寸和 bbox 合法性检查，已经满足开始第一轮训练的工程条件。
