# 约定一：分类体系

标签定义、提示中的标签块、归并和评分使用同一份配置：`configs/schemas/biored.yaml`、`configs/schemas/drugprot.yaml`、`configs/schemas/entities.yaml`。显示名称与稳定内部编号分开。每一档完整覆盖最细标签，每个细标签只有一个归属。`No_Relation` 不进入阳性谓词层级。不能把无法映射的标签临时归入 Association。

## BioRED

P1：Any_Relation。

P2：Positive_Correlation、Negative_Correlation、Other_Relation。

P3：Positive_Correlation、Negative_Correlation、Association、Bind、Remaining_Relation。

P4：Positive_Correlation、Negative_Correlation、Association、Bind、Comparison、Cotreatment、Drug_Interaction、Conversion。

归并：

- 正相关与负相关在 P2 和 P3 保留。
- Association 与 Bind 在 P3 保留，在 P2 进入 Other_Relation。
- Comparison、Cotreatment、Drug_Interaction、Conversion 在 P3 进入 Remaining_Relation，在 P2 进入 Other_Relation。
- 全部阳性类别在 P1 进入 Any_Relation。

程序检查 P4→P3→P2 与 P4→P2 相同。

## DrugProt

P4 为官方 13 类。P3 为七组：

- Upregulation_or_activation：ACTIVATOR、INDIRECT-UPREGULATOR。
- Downregulation_or_inhibition：INHIBITOR、INDIRECT-DOWNREGULATOR。
- Direct_regulation_unspecified：DIRECT-REGULATOR。
- Agonism：AGONIST、AGONIST-ACTIVATOR、AGONIST-INHIBITOR。
- Antagonism：ANTAGONIST。
- Metabolic_relation：SUBSTRATE、PRODUCT-OF、SUBSTRATE_PRODUCT-OF。
- Part_of：PART-OF。

P2 只保留明确上调／激活组与明确下调／抑制组，其余进入 Other_Relation。AGONIST 系列不归入上调，ANTAGONIST 不自动当作一般下调。P1 为 Any_Relation。

BioRED 的正负相关与 DrugProt 的明确上调／下调分开报告。

## 实体信息

- E0：名称、上下文和目标角色，不显示显式类型。这不等于模型无法从文字推断类型。
- E1：GeneOrGeneProduct、ChemicalEntity、SequenceVariant 合为 Molecular_Entity，其余原类型保留。
- E2：BioRED 原始六类。

数据库标识不提供给模型。DrugProt 输入把 GENE-Y 与 GENE-N 统一为 GENE，化学物在前。

## 对照

只用于 BioRED、E2、Qwen3-32B。一个 P2 改名版本和一个 P4 改名版本，成员与定义不变。三个随机合并版本，组大小为 1、1、6；种子 42；标签先按稳定编号排序；相同划分不重复。成员和改名映射保存在 `configs/schemas/biored_controls.yaml`。该文件在研究者确认前是草稿。随机组不使用新的医学机制名称。

## E3

本轮只提供映射检查和空模板。禁止把药理作用记录当作药物、把基因数据库类别当作文本分子形式、把变异标识格式当作完整变异类型。可行性评估不在本轮作人工判断，也不自动加入主矩阵。
