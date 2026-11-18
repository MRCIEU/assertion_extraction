# 变更记录

## 2026-11-15

从空的代码工作区建立本项目。存储目录中已有的原始语料、模型目录、缓存和历史说明均保留，没有删除或覆盖。

数据版本 `biored_candidates_v1` 与 `bc8_candidates_v1` 把逗号分隔的复合标识当作一个概念。关系引用的是其中的单个标识，因此大量金标准关系被记为身份缺失。该版本的 `status.json` 标为 superseded。当前版本是 `biored_candidates_v2` 与 `bc8_candidates_v2`：复合标识的每个成分都保留，无身份的 `-` 不合并。

DrugProt 当前版本是 `drugprot_candidates_v1`。候选表按批写入，以避免一次性保存全部未标注配对。

本轮没有模型训练、权重推理、GPU 作业、付费接口请求或人工语义答案。
