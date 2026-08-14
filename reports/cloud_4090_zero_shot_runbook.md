# Cloud 4090 Zero-Shot AIC Runbook

Date: 2026-08-05

## Current Decision

Run zero-training full AIC inference on a rented RTX 4090 24GB instance. The current verified control score is MM-Grounding-DINO-T at 0.5011, so each new result is compared against that number.

Model order:

1. LocateAnything-3B
2. Qwen3-VL-8B-Instruct

Do not mix in Tile, Depth, IR, Ranker, Query rewrite, or training in this round.

## Local Assets

Create the upload package on Windows:

```powershell
cd D:\12525\Documents\pytorch\baseline_v0
.\tools\prepare_aic_cloud_assets.ps1
```

Expected output:

```text
D:\12525\Documents\pytorch\aic_cloud_upload_4090_v1\
  aic_testset_20260708.tar
  cloud_upload_manifest.json
```

The archive extracts to `/workspace/aic_data` with `queries/queries.json` and `Images/...` directly under it.

Prepared local archive:

```text
path: D:\12525\Documents\pytorch\aic_cloud_upload_4090_v1\aic_testset_20260708.tar
size: 13,112,779,264 bytes
sha256: C9263079E86EC86582B1D42EE579CFF6841E61238F420287D43D94C18B0088D7
queries_sha256: 2A08CD3A930D9749EDB86A8B420D8259DCE446116AD98D93338EDF8B3E1EF67C
expected_query_count: 9555
```

## Cloud Setup

On the cloud instance:

```bash
cd /workspace
git clone https://github.com/finalymadeout7433-gif/aic-multimodal-visual-grounding.git
cd /workspace/aic-multimodal-visual-grounding
git checkout exp/aic-testset-profile-v1
```

Upload `aic_testset_20260708.tar` to `/workspace/upload/`, then extract:

```bash
mkdir -p /workspace/aic_data
tar -xf /workspace/upload/aic_testset_20260708.tar -C /workspace/aic_data
```

Run preflight:

```bash
bash tools/cloud/run_cloud_preflight.sh
```

## LocateAnything-3B

Install:

```bash
bash tools/cloud/install_locateanything_env.sh
```

Smoke:

```bash
LIMIT_ARG="--limit 10" bash tools/cloud/run_locateanything_aic.sh
LIMIT_ARG="--limit 100" bash tools/cloud/run_locateanything_aic.sh
```

Full run:

```bash
FINALIZE_ARG="--finalize" bash tools/cloud/run_locateanything_aic.sh
```

Output ZIP:

```text
/workspace/outputs/aic_locateanything_3b_zero_shot_v1/submission/predictions_submission.zip
```

## Qwen3-VL-8B-Instruct

Install:

```bash
bash tools/cloud/install_qwen3vl_env.sh
```

Smoke:

```bash
LIMIT_ARG="--limit 10" bash tools/cloud/run_qwen3vl_aic.sh
LIMIT_ARG="--limit 100" bash tools/cloud/run_qwen3vl_aic.sh
```

Full run, only if smoke has zero parse failures:

```bash
FINALIZE_ARG="--finalize" bash tools/cloud/run_qwen3vl_aic.sh
```

Output ZIP:

```text
/workspace/outputs/aic_qwen3_vl_8b_zero_shot_v1/submission/predictions_submission.zip
```

## Release Packaging

After one or both full runs complete:

```bash
bash tools/cloud/package_cloud_results.sh
```

Upload-ready files:

```text
/workspace/outputs/platform_upload_ready/
  AIC_LocateAnything_3B_zero_shot_v1.zip
  AIC_Qwen3_VL_8B_Instruct_zero_shot_v1.zip
  SHA256SUMS.txt
  UPLOAD_GUIDE.md
```

## Stop Rules

- Stop LocateAnything full run if the 100-query smoke returns no candidates or illegal boxes.
- Stop Qwen full run if the 100-query smoke has bbox parse failures.
- Do not manually edit a prediction JSON.
- Do not upload LocateAnything if its NVIDIA research license is deemed incompatible with the contest.
