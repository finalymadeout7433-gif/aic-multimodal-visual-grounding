# Florence 候选角色与 Oracle 分析

本报告只使用外部 RefCOCO 系列 validation；未使用 AIC 正式标签或 holdout。

## 核心结果

- 样本数：1500
- first ACC@0.5：0.6893
- candidate oracle ACC@0.5：0.7533
- oracle 差值：0.0640
- first 正确：1034
- first 错但其他候选可救：96
- 所有候选均失败：370

## 候选语义类型

- `multi_distinct_label`：516
- `multi_same_label`：114
- `no_candidate`：2
- `single`：868

## 候选 label 与 Query 的粗粒度角色

- `divergent_or_reference`：1
- `partial_query_phrase`：3
- `query_phrase`：1899
- `whole_query`：635

## 可视化

颜色：GT 绿色、first 红色、oracle 蓝色、其他候选黄色。

- `first_correct` / `refcoco:100157` / `outputs/diagnostics/analysis/visualizations/01_first_correct_refcoco_100157.jpg`
- `first_correct` / `refcoco:100282` / `outputs/diagnostics/analysis/visualizations/02_first_correct_refcoco_100282.jpg`
- `oracle_rescue` / `refcoco:100783` / `outputs/diagnostics/analysis/visualizations/03_oracle_rescue_refcoco_100783.jpg`
- `oracle_rescue` / `refcoco:105303` / `outputs/diagnostics/analysis/visualizations/04_oracle_rescue_refcoco_105303.jpg`
- `multi_distinct_label` / `refcoco:102253` / `outputs/diagnostics/analysis/visualizations/05_multi_distinct_label_refcoco_102253.jpg`
- `multi_same_label` / `refcoco:103702` / `outputs/diagnostics/analysis/visualizations/06_multi_same_label_refcoco_103702.jpg`
- `plural_group` / `refcoco:142122` / `outputs/diagnostics/analysis/visualizations/07_plural_group_refcoco_142122.jpg`
- `plural_group` / `refcoco:73462` / `outputs/diagnostics/analysis/visualizations/08_plural_group_refcoco_73462.jpg`
- `additional` / `refcoco:101009` / `outputs/diagnostics/analysis/visualizations/09_additional_refcoco_101009.jpg`
- `additional` / `refcoco:101375` / `outputs/diagnostics/analysis/visualizations/10_additional_refcoco_101375.jpg`
- `additional` / `refcoco:102348` / `outputs/diagnostics/analysis/visualizations/11_additional_refcoco_102348.jpg`
- `additional` / `refcoco:102480` / `outputs/diagnostics/analysis/visualizations/12_additional_refcoco_102480.jpg`
