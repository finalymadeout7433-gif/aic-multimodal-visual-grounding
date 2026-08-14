# 优化触发矩阵

> 证据边界：Query 与图像统计是输入事实；模型框面积、一致性和“小目标”分级是无标签代理；模型优劣与 ACC 必须等待平台验证。


```json
{
  "ape_ti": {
    "status": "triggered",
    "triggered": true
  },
  "depth_late_fusion_preparation": {
    "status": "prepare_only_no_2d_substitution",
    "triggered": true
  },
  "frozen_thresholds": {
    "ape_region_all_disagree_ratio": 0.4,
    "ape_region_ratio": 0.1,
    "depth_relation_ratio": 0.15,
    "depth_rgb_stability_ratio": 0.5,
    "piza_high_plus_medium_ratio": 0.3,
    "piza_high_small_ratio": 0.15,
    "piza_scale_gain_pp": 10.0,
    "rgbt_low_light_ratio": 0.15,
    "rgbt_usable_ir_ratio": 0.8
  },
  "llmdet": {
    "status": "deferred_until_long_query_role_cluster_is_validated"
  },
  "piza": {
    "high_plus_medium_threshold": 0.3,
    "high_small_threshold": 0.15,
    "scale_gain_threshold_pp": 10.0,
    "status": "not_triggered",
    "triggered": false
  },
  "recommendation_rationale": "region/structure prevalence or disagreement crossed the frozen threshold",
  "recommended_next_model": "APE-Ti",
  "rgbt_audit": {
    "status": "not_triggered",
    "triggered": false
  },
  "thresholds_frozen_before_s04": true
}
```

冻结规则优先级：PIZA（须同时满足规模与尺度收益）→ APE-Ti → RGBT 资产审计 → MM-Grounding-DINO-T 多域验证。Depth 只准备 late fusion，禁止用二维位置代替物理深度。


## 本轮冻结判定

- APE-Ti：`triggered`，区域/结构比例为 16.69%。
- PIZA：`not_triggered`；虽然高+中置信小目标代理达到 28.67%，尺度稳定候选增益仅 6.5 pp，未越过 10 pp 门槛。
- RGBT 专项：`not_triggered`；低光图像组比例仅 0.50%。
- Depth：`prepare_only_no_2d_substitution`；只进入数据与后融合准备，不生成本轮提交。
