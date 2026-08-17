# RGBT-GroundBench 云端持久化数据清单

本轮 Phase 1 只需要 RGBT-GroundBench，不需要 AIC 正式测试集。建议在租用 GPU 前，将下列 6 个官方归档上传到 Featurize“我的数据集”，总大小约 9.68 GiB（10,390,087,680 bytes）。

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `ann_flir.tar` | 3,307,520 | `7CAEBB8F3E947E6D5440EF0FA5C5CF13FCB492B896EF9D3DF2FC595B9A68E24A` |
| `ann_m3fd.tar` | 4,567,040 | `C69BFB83B5BBAA68312044BBCF6E0B5A3B43E468F795E767D5A134EC03A53FD3` |
| `ann_mfad.tar` | 8,529,920 | `BEEE1CADF991E9823CAF5043BCE8AA33016054E2E6334952B846E8B7CB5654EA` |
| `data_flir.tar` | 935,802,880 | `0B7B81A59ED796BFB2D1BB4258F45EA4B90BE816C9D33C4C1A1F194CFADB470E` |
| `data_m3fd.tar` | 7,246,704,640 | `BDBA0DB59A3FB1BA02E67B5B6CEE19320BE50B11D787115E15F7ED42E1339F7C` |
| `data_mfad.tar` | 2,191,175,680 | `B84155830051A66B601253FDCD734894F1753DF4836C31D2F10AAE54111F73E6` |

本地来源目录：

```text
D:\AIC赛题一数据集\03_RGBT_GroundBench
```

云端展开命令（目标目录可改，但配置必须同步）：

```bash
mkdir -p /home/featurize/data/RGBT_GroundBench/extracted
cd /path/to/persistent_dataset
for f in ann_flir.tar ann_m3fd.tar ann_mfad.tar data_flir.tar data_m3fd.tar data_mfad.tar; do
  tar -xf "$f" -C /home/featurize/data/RGBT_GroundBench/extracted
done
```

展开后必须存在：

```text
extracted/image_data
extracted/rgbtvg_flir
extracted/rgbtvg_m3fd
extracted/rgbtvg_mfad
```
