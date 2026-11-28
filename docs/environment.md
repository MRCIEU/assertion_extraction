# 环境说明

核查记录在存储目录：

`/home/b5ac/freddieyu.b5ac/projects/project_1/studies/schema_granularity_v1/reports/progress/environment_audit.md`

该报告写于代码完成之前，其中的状态标志只表示当时还没有代码。最终状态以交付报告为准。

## 实际使用的分析环境

路径：`/home/b5ac/freddieyu.b5ac/projects/project_1/environments/analysis`

Python 3.12.12。已安装 numpy 2.2.6、pandas 2.3.3、pyarrow 22.0.0、pyyaml 6.0.3、pytest 9.0.2、matplotlib 3.10.8、transformers 4.57.5、tokenizers 0.22.2。没有安装 PyTorch 或 vLLM。pip 缓存位于存储目录的 `cache/pip`。共享环境 `hf-hpc` 没有修改。

## 测量模型

确定性估计和科学验证使用已有的 `hf-hpc`，不往该环境安装软件包。激活命令：

`source /home/b5ac/freddieyu.b5ac/miniforge3/etc/profile.d/conda.sh && conda activate hf-hpc`

路径：`/home/b5ac/freddieyu.b5ac/miniforge3/envs/hf-hpc`。Python 3.10.19，PyTorch 2.9.1，transformers 4.57.5。当前登录节点上 `torch.cuda.is_available()` 为假。该环境没有 spaCy。生物医学分句比较因此没有装进 `hf-hpc`。

## 训练与生成

`environments/training.yaml` 和 `environments/generation.yaml` 只定义依赖。它们不是已经实测的模型环境。登录节点上导入 PyTorch 曾长时间无响应，因此本轮不用导入方式核查 PyTorch。

## 作业

分区 `workq`，账户 `brics.b5ac`，QoS `normal`。本轮 CPU 上限为同时最多 8 个作业，每个最多 8 CPU、16 GB 内存、30 分钟，不申请 GPU。提交脚本使用 `--export=NONE`，再显式传入代码路径、存储路径和命令。
