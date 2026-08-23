# AIC RGB–TIR module

This package contains the staged RGB–TIR representation work for Qwen3-VL. The
repository deliberately keeps RGB as the frozen control path and treats TIR as a
rejectable residual source. Representation experiments must pass their own gates
before Query conditioning, fusion, bbox generation, or AIC platform claims are made.

The current research status is `STOP_CURRENT_12D_QUALITY_PROXY_ROUTE`. Work after
D2 established a useful but narrower result: D3_SP000 frozen features passed the
locked G1R shared-head dev and confirmation probe, while static Layer16 G2,
Query-conditioned control, and the current 12-dimensional pair-level
quality/registration gate did not generalize through their pre-registered gates.
The strongest quality-proxy upper-bound probe reached OOF Spearman `0.22666`, but
beneficial-vs-harmful AUC was only `0.59024` with a confidence interval crossing
chance. No RGB-TIR Adapter has been released, official validation remains sealed
where required, and no AIC RGB-TIR platform gain is claimed.

## Architectural invariants

- Qwen RGB vision, LLM, and native bbox path stay frozen.
- TIR uses rank-48 LoRA on vision attention `qkv/proj`.
- RGB/TIR share one resize/crop/pad geometry and `grid_thw`.
- Border-connected near-black TIR padding becomes an explicit valid-FOV mask.
- Layer 8/16/24/final pre-merger features are exposed without editing Transformers.
- `gate=0`, missing TIR, invalid TIR, or an all-zero mask degrade to the same model's
  RGB-only path.
- A second-model fallback is outside the module contract.
- Model weights, Teacher Banks, embedding caches, and `outputs/` never belong in Git.

## Module map

| Module | Responsibility |
|---|---|
| `data.py` | Immutable records, official split parsing, pair keys, leakage checks |
| `processing.py` | Shared geometry, bbox mapping, black-border masks, Qwen normalization |
| `modeling.py` | Qwen visual hooks, RGB/TIR paths, rank-48 Adapter, validation triplets |
| `validation.py` | RGB-only equivalence, tensor/hash safety, finite-output checks |
| `phase1.py` | Initial paired alignment training |
| `phase15.py` | Full-val retrieval, collapse, subgroup, cache, hard gates |
| `phase16.py` | InfoNCE, relational distillation, negative sampling, Teacher Bank |
| `phase17a_plus.py` | Absolute drift, RGB-quality and false-negative diagnostics |
| `phase18.py` | Base-relative retention candidates and D1 selection |
| `phase18_full.py` | Full train, checkpoint selection, sealed official-val contract |
| `phase18r_audit.py` | Record/pair-level rank, retrieval and ExcessDrift re-audit |
| `phase19_d2.py` | Base-TIR neighborhood geometry probe |
| `phase19_d2_full.py` | D2_G025 full train, dual-dev gate, Stage A/official boundary |
| `phase19_d3.py` / `phase19_d3a.py` | Same-pair structure probes and strict ROI/layer audits |
| `sidebranch_interface.py` | Causal Layer16 sidebranch and exact RGB/invalid-TIR bypass |
| `phase19_g1.py` / `phase19_g1_protocol.py` | Frozen Query-to-ROI feature probe and locked local evaluation |
| `phase19_g2_layer16.py` / `phase19_g2_evaluation.py` | Layer16-only zero-init residual and formal dual-dev closeout |
| `phase19_query_gate.py` / `phase19_query_control_v2.py` | Query-gate probes with wrong/shuffled-Query controls |
| `phase19_quality_registration_gate.py` | Query-free quality/registration gate and negative controls |
| `query_probe.py` | Frozen Query interface smoke only; not a grounding claim |
| `artifacts.py` | Fingerprints and deterministic artifact handling |

## Experiment sequence

```text
Phase 0       paired-data/model seam and RGB safety
Phase 1       paired alignment warmup
Phase 1.5     full-val retrieval and collapse discovery
Phase 1.6     cross-image InfoNCE + relational distillation
Phase 1.7A+   absolute drift and failure attribution
Phase 1.8     Base-relative retention (D1_L050)
Phase 1.8R    persistent full train + sealed official-val + local audit
Phase 1.9     Base-TIR geometry probe and D2_G025 full dual-dev gate
Phase 1.9-A   four-checkpoint, dual-dev per-layer spectrum audit; no training
Phase 1.9-D3  same-pair structure probe + strict ROI/layer audit
Phase 1.9-G0  causal sidebranch interface trace; no training
Phase 1.9-G1  frozen six-arm Query-to-ROI probe + locked confirmation
Phase 1.9-G2  Layer16-only residual; semantic/multi-query dual-dev NO_GO
Phase 1.9-Q   Query-control probes; wrong Query control NO_GO
Phase 1.9-QR  quality/registration gate + adaptivity/upper-bound audit
next          candidate/ROI-level Rescue-Harm predictability upper bound; local first
```

The full chronology, metrics, failure analysis, and decision boundaries live in
`reports/AIC_RGB_TIR_LIVING_TECHNICAL_REPORT.md`. The self-contained route audit
for web review is
`reports/AIC_RGB_TIR_FULL_ROUTE_OPTIMIZATION_VALIDATION_AND_NEXT_PLAN_2026_08_23.md`.

## Local checks

From the repository root:

```powershell
python -m pytest -q tests/test_rgbtir_*.py
python -m compileall -q src/aic_rgbtir tools
git diff --check
```

Phase-specific launchers use example configuration files under `configs/`. Local
paths and credentials belong in ignored `*.local.yaml` files or environment
variables, never in committed templates.

No source image or official JSON file is modified. Derived manifests, caches,
checkpoints, reports generated by runs, and audit outputs are written below ignored
output directories.
