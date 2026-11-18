# 约定五：文件与记录

代码位于 `/home/b5ac/freddieyu.b5ac/project_1`。数据与输出位于 `/home/b5ac/freddieyu.b5ac/projects/project_1`，解析后为 `/lus/lfs1aip2/projects/b5ac/project_1`。程序拒绝把研究输出写入代码目录。

## 代码

研究代码在 `src/schema_study/`。命令入口是 `python -m schema_study.cli`。命令包括环境检查、数据准备、分类验证、输入构建、实验清单、训练、预测、生成、审阅材料和 CPU 验证。训练、预测和生成在本轮被硬性阻止。

`local/settings.yaml` 不提交，也不保存密钥。依赖定义在 `environments/`。实际分析环境安装在存储目录的 `environments/analysis`，不含 PyTorch。

## 输出

原始发布文件留在 `data/raw/`。准备结果在 `data/prepared/{corpus}/{data_version}/`，包括 documents、entities、relations、candidates、splits、exclusions、摘要、验证和来源。输入在 `data/inputs/{corpus}/{input_version}/`。

研究目录为 `studies/schema_granularity_v1/`。其中 `frozen_plan/` 本轮只保存开发基线草稿，不是公开预注册。实验清单是计划，不是已运行记录。`runs/` 下的训练、预测和评价目录本轮不写入模型结果。合成验证写入 `tmp/cpu_validation_synthetic/`，并标明 synthetic。

统一预测字段为 run_id、dataset、split、document_id、candidate_id、entity_level、predicate_level、route、model_id、seed、input_id、predicted_labels、output_status、probabilities、source_prediction_id、request_id。金标准从独立候选表关联。空集合与失败用状态区分。

审计收到的原始答案不覆盖。人工、模型辅助和裁决分开保存。审阅包不包含其他人的答案或模型辅助结果。

共享数据和权重按版本保存，不复制到每一次运行。上游只读。重算产生新版本。cache 和 tmp 不作为唯一正式产物。运行完成记录之后不修改正式输出；发现问题时另建更正记录或新版本。

## 状态

以下状态分开记录，不合并成一个完成标志：

- code_complete
- cpu_validation_complete
- real_data_validation_complete
- model_smoke_test_complete
- formal_experiments_started
- human_review_complete
- api_requests_made

后四项在本轮为 false。前三项只按实际通过的语料、输入版本和功能填写。
