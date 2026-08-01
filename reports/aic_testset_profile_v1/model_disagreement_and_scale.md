# 模型分歧与尺度代理

> 证据边界：Query 与图像统计是输入事实；模型框面积、一致性和“小目标”分级是无标签代理；模型优劣与 ACC 必须等待平台验证。


- 三基线在 IoU@0.5 下全部分歧代理比例：12.71%。
- 区域/结构簇全部分歧比例：15.05%。
- S03 切换 4,770 条，其中小框到大框代理 1,355 条、跨标签 2,679 条；这些是负迁移风险代理，不是逐条错误真值。
- S04 仅有 164 条切换，最大面积倍率 1.4866505469863338; 任务多标签分布为 `{"action": 21, "attribute": 40, "ordinal": 97, "plural_group": 3, "region_structure": 26, "small_object_lexical_prior": 25, "spatial_relation": 94}`。
- IoU@0.5 一致性分组：`{"all_agree": 3974, "all_disagree": 1214, "florence_gdino_agree_s03_differs": 2383, "florence_s03_agree_gdino_differs": 781, "gdino_s03_agree_florence_differs": 1203}`。
- 尺度探针：`{"aic_accuracy_computed": false, "aic_oracle_computed": false, "enabled": true, "evidence_kind": "model_derived_proxy", "full_miss_tile_only_target_candidate_rate": 0.01, "high_resolution_longest_edge": 1706, "high_resolution_shortest_edge": 1024, "ordinary_control_candidate_inflation": 5.748488095238095, "records": 400, "selected_queries": 400, "small_enhanced_stable_candidate_rate": 0.955, "small_full_stable_candidate_rate": 0.89, "small_records": 200, "small_stable_candidate_gain_pp": 6.5, "stage1_complete_records": 400, "stage2_complete_records": 200, "stage2_selected_records": 200, "status": "complete", "telemetry_is_excluded_from_deterministic_manifest": true, "telemetry_path": "telemetry/scale_probe_telemetry.jsonl", "tile_overlap": 0.2, "tile_processor_size": {"longest_edge": 1333, "shortest_edge": 800}}`。

机器明细位于 `outputs/aic_testset_profile_v1/model_behavior_profile.jsonl` 与 `small_target_proxy.csv`。
