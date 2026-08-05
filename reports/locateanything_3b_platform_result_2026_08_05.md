# LocateAnything-3B AIC Platform Result and Cloud Storage Notes

Date: 2026-08-05

## Executive conclusion

This round confirms that `LocateAnything-3B` is the current strongest confirmed AIC zero-training model in this project.

Confirmed platform score from the user-provided leaderboard screenshot:

| Item | Value |
|---|---:|
| Model | `nvidia/LocateAnything-3B` |
| Strategy | Zero-shot, RGB visible image + original English query |
| AIC queries | 9,555 |
| AIC platform ACC@0.5 | **0.7210** |
| Platform rank shown | 29 |
| Platform record time | 2026-08-05 21:31:36 |

This score corresponds to `LocateAnything-3B`, not `Qwen3-VL-8B-Instruct`.

## Local submission package

The upload package that produced this result is stored locally outside the Git repository:

```text
D:\12525\Documents\pytorch\aic_cloud_upload_4090_v1\platform_upload_ready\AIC_LocateAnything_3B_zero_shot_v1.zip
```

It is intentionally not committed to GitHub because it is a generated competition submission artifact.

## Cloud run audit

Cloud output directory:

```text
/home/featurize/aic_cloud/outputs/aic_locateanything_3b_zero_shot_v1
```

Audit summary:

| Check | Value |
|---|---:|
| Completed records | 9,555 / 9,555 |
| Invalid bbox count | 0 |
| No-candidate count | 0 |
| Non-bbox field modifications | 0 |
| Query IDs exact | true |
| ZIP entries | `predictions_submission.json` only |
| ZIP SHA-256 | `4763A3627FFE9E11898E9205C967F7FAB41482AD7464C6B07B87ABF55859C45C` |
| JSON SHA-256 | `9119E8F192ABA2F3F11E2A26B832AAF703062805CAC966A1DCC0C7E9B39FB037` |

Runtime environment:

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 |
| Python | 3.10.20 |
| PyTorch | 2.13.0+cu130 |
| CUDA | 13.0 |
| Peak CUDA allocated | 20,691,871,232 bytes |
| Wall time | 8,206.47 seconds |

## Qwen3-VL-8B status

`Qwen3-VL-8B-Instruct` did not produce a completed submission package in this round.

Known evidence:

| Item | Status |
|---|---|
| 10-query smoke | passed |
| 100-query smoke | passed after adding parser fallback |
| Full run | started |
| Last successful watcher state | 1,256 / 9,555 predictions |
| Full submission ZIP | not generated locally |
| Platform score | not confirmed |

Therefore any leaderboard score from 2026-08-05 21:31 should not be attributed to Qwen unless a Qwen submission ZIP is later found and verified.

## Cloud storage clarification

This run uploaded and extracted the working data under:

```text
/home/featurize/aic_cloud
```

That path should be treated as instance-local working storage, not as the platform's persistent netdisk path.

The persistent netdisk path documented by Matpool/Featurize is `/mnt`. Files placed under `/mnt` are reflected in the web netdisk and can persist after instance release. Files outside `/mnt` are part of the running machine environment; after release they normally require either:

1. saving the environment snapshot, or
2. re-uploading / re-downloading the files on the next instance.

For this round, the local evidence shows only the completed LocateAnything submission was downloaded back to Windows. The AIC visible-image working copy, model weights, and partial Qwen full-run outputs were not confirmed as preserved in platform netdisk.

## Practical consequence for the next cloud run

If starting a new cloud instance, assume the following must be prepared again unless the old instance/environment is recoverable:

1. clone the GitHub repository;
2. download model weights;
3. upload or extract AIC visible images and `queries.json`;
4. run Qwen or any next model from scratch or from a confirmed preserved checkpoint.

To avoid repeated upload cost next time, prefer one of these options:

| Option | Use case |
|---|---|
| Store compressed AIC data under `/mnt` | Best for persistent reuse across future instances |
| Use Featurize dataset/netdisk workflow | Best if the platform storage quota is enough |
| Save environment snapshot | Useful for dependencies and weights, but snapshot can be large |
| Keep only GitHub + local upload archive | Cheapest if cloud storage is limited, but requires re-upload |

## Current model ranking

Confirmed AIC platform results so far:

| Rank in project | Model / strategy | ACC@0.5 |
|---:|---|---:|
| 1 | LocateAnything-3B zero-shot | **0.7210** |
| 2 | MM-Grounding-DINO-T zero-shot | 0.5011 |
| 3 | Florence-2-large-ft RGB-only | 0.4980 |
| 4 | Conservative GDINO ranker S04 | 0.4957 |
| 5 | LLMDet-Swin-T zero-shot | 0.4942 |
| 6 | GroundingDINO-Tiny Top-1 | 0.4938 |
| 7 | APE-Ti zero-shot | 0.4424 |
| 8 | RefCOCO-trained LightGBM ranker S03 | 0.3190 |

## Decision

`LocateAnything-3B` should become the new confirmed control baseline.

Next platform experiments should be single-variable changes relative to this baseline, for example:

1. Qwen3-VL-8B full run as an independent model, if cloud compute is available again;
2. LocateAnything prompt / mode ablation;
3. LocateAnything higher input scale or slow/hybrid decoding if supported;
4. controlled ensemble only after each single model score is known.

Do not continue the old RefCOCO LightGBM ranker submission route as a main path.
