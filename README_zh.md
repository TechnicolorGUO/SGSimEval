# SGSimEval: 自动综述生成系统的综合多维度相似性增强基准测试

### 概述

SGSimEval 是一个用于综述生成系统的综合评估框架，结合了传统评估指标和基于相似性的评估方法。它提供常规评估分数和相似性调整分数，使用两种方法：平衡相似性和人类完美相似性。

### 主要功能

- **多维度评估**：大纲、内容、参考文献等
- **基于相似性的评分**：两种相似性方法（平衡和人类完美）
- **批量处理**：支持跨多个系统的大规模评估
- **综合分析**：平均分数、聚合和比较工具
- **灵活输出**：CSV、LaTeX 和 JSON 格式

### 快速开始示例

如果你已经在 `surveys/` 文件夹中准备了包含 CS 类别数据的数据集结构，可以按顺序运行以下命令来获得评估结果：

```bash
# 第一步：评估 CS 类别的综述质量
python main.py batch-evaluate --mode category --categories cs --model "[MODEL_NAME]" --workers 4

# 第二步：计算系统与真实标签之间的相似性分数
python main.py similarity --systems AutoSurvey SurveyForge InteractiveSurvey LLMxMapReduce SurveyX --model [MODEL_NAME]

# 第三步：计算所有分数并生成最终结果 CSV
python main.py calculate-all --categories cs --systems AutoSurvey SurveyForge InteractiveSurvey LLMxMapReduce SurveyX --models [MODEL_NAME] --workers 4
```

这将生成包含综合评估分数的最终结果 CSV 文件。

### 完整使用流程示例

#### 第一步：准备数据集结构

首先，使用 `get_topics.py` 准备数据集结构：

```bash
# 生成数据集结构并创建系统文件夹
python scripts/get_topics.py --systems AutoSurvey SurveyForge InteractiveSurvey LLMxMapReduce SurveyX --numofsurvey 50
```

这将：
- 在 `surveys/` 中创建必要的文件夹结构
- 生成 `surveys/tasks.json` 用于任务映射
- 为每个主题设置系统子文件夹

**预期的文件夹结构：**
```
surveys/
├── tasks.json
└── cs/
    └── 3D Gaussian Splatting Techniques/
        ├── AutoSurvey/
        │   └── 3D Gaussian Splatting Techniques.md
        ├── SurveyForge/
        │   └── 3D Gaussian Splatting Techniques.md
        └── ...
```

#### 第二步：评估综述质量

运行批量评估来评估生成综述的质量：

```bash
# 评估 CS 类别的所有系统
python main.py batch-evaluate --mode system --systems AutoSurvey SurveyForge InteractiveSurvey LLMxMapReduce SurveyX --model [MODEL_NAME] --workers 4
```

这会在每个系统文件夹中生成包含评估分数的 `results_[MODEL_NAME].json` 文件。

#### 第三步：计算相似性分数

**重要**：在运行相似性计算之前，确保你的 `.env` 文件使用嵌入模型：

```bash
# 在你的 .env 文件中，将模型更改为嵌入模型
MODEL=text-embedding-3-large  # 或其他嵌入模型
```

然后计算相似性分数：

```bash
# 计算相似性分数（平衡和人类完美）
python main.py similarity --systems AutoSurvey SurveyForge InteractiveSurvey LLMxMapReduce SurveyX --model [MODEL_NAME]
```

这将：
- 计算大纲、内容和参考文献的嵌入
- 生成系统输出与真实标签之间的相似性分数
- 用相似性调整分数更新 `results_[MODEL_NAME].json`
- 创建包含原始相似性数据的 `similarity.json` 文件

#### 第四步：计算所有分数并生成结果

```bash
# 计算综合分数并生成最终结果
python main.py calculate-all --categories cs --systems AutoSurvey SurveyForge InteractiveSurvey LLMxMapReduce SurveyX --models [MODEL_NAME] --workers 4
```

这生成：
- `surveys/cs_results.csv`：类别级别结果
- `surveys/cs_average.csv`：类别平均值
- `surveys/global_average.csv`：所有类别的全局平均值

### 理解结果

#### 全局平均 CSV 列说明

`surveys/global_average.csv` 文件包含以下列：

| 列名 | 描述 |
|------|------|
| `System` | 系统名称（AutoSurvey、SurveyForge 等） |
| `Model` | 使用的评估模型（[MODEL_NAME]） |
| `Outline` | 原始大纲评估分数（0-5） |
| `Content` | 原始内容评估分数（0-5） |
| `Reference` | 原始参考文献评估分数（0-5） |
| `Outline_sim` | **平衡相似性调整后的大纲分数** |
| `Content_sim` | **平衡相似性调整后的内容分数** |
| `Reference_sim` | **平衡相似性调整后的参考文献分数** |
| `Outline_sim_hp` | **人类完美相似性调整后的大纲分数** |
| `Content_sim_hp` | **人类完美相似性调整后的内容分数** |
| `Reference_sim_hp` | **人类完美相似性调整后的参考文献分数** |
| `Outline_structure` | 原始大纲结构分数（0-100） |
| `Faithfulness` | 原始忠实度分数（0-100） |
| `Reference_quality` | 原始参考文献质量分数（0-100） |
| `Outline_structure_sim` | **平衡相似性调整后的大纲结构分数** |
| `Faithfulness_sim` | **平衡相似性调整后的忠实度分数** |
| `Reference_quality_sim` | **平衡相似性调整后的参考文献质量分数** |
| `Outline_structure_sim_hp` | **人类完美相似性调整后的大纲结构分数** |
| `Faithfulness_sim_hp` | **人类完美相似性调整后的忠实度分数** |
| `Reference_quality_sim_hp` | **人类完美相似性调整后的参考文献质量分数** |

**相似性分数说明：**
- **平衡（`_sim`）**：基于相似性将原始分数与真实标签分数结合
- **人类完美（`_sim_hp`）**：基于相似性将原始分数与完美分数（5.0/100.0）结合

### 命令参考

#### 评估命令

```bash
# 单文件评估
python main.py evaluate --md-path "surveys/cs/topic/system/file.md" --model "[MODEL_NAME]"

# 按类别批量评估
python main.py batch-evaluate --mode category --categories cs econ --model "[MODEL_NAME]" --workers 4

# 按系统批量评估
python main.py batch-evaluate --mode system --systems AutoSurvey SurveyForge --model "[MODEL_NAME]" --workers 4
```

#### 相似性命令

```bash
# 批量相似性计算
python main.py similarity --systems AutoSurvey SurveyForge --model "[MODEL_NAME]"

# 单文件相似性
python main.py similarity --mode single --md-path "path/to/system.md" --gt-path "path/to/ground_truth.md" --model "[MODEL_NAME]"
```

#### 分析命令

```bash
# 计算平均值
python main.py average --mode category --categories cs econ

# 聚合结果
python main.py aggregate --mode category --categories cs econ

# 综合计算
python main.py calculate-all --categories cs --systems AutoSurvey SurveyForge --models [MODEL_NAME] --workers 4
```

#### 工具命令

```bash
# 清理分数
python main.py clear --mode scores --categories cs --systems AutoSurvey --model "[MODEL_NAME]"

# 生成报告
python main.py generate --mode evaluation --systems AutoSurvey SurveyForge --model "[MODEL_NAME]"
```

### 安装

#### 基础安装

```bash
pip install -r requirements.txt
```

#### PDF 处理设置（可选）

如果你的输入数据包含 PDF 文件而不是 Markdown 文件，你需要设置 [MinerU](https://github.com/opendatalab/MinerU) 来进行 PDF 到 Markdown 的转换：

1. **安装 MinerU**：
   ```bash
   # 使用 pip
   pip install --upgrade pip
   pip install uv
   uv pip install -U "mineru[core]"
   
   # 或从源码安装
   git clone https://github.com/opendatalab/MinerU.git
   cd MinerU
   uv pip install -e .[core]
   ```

2. **MinerU 系统要求**：
   - **操作系统**：Linux / Windows / macOS
   - **内存**：最低 16GB+，推荐 32GB+
   - **磁盘空间**：20GB+，推荐 SSD
   - **Python 版本**：3.10-3.13

3. **转换 PDF 到 Markdown**：
   ```bash
   # 基础转换
   mineru -p <输入pdf路径> -o <输出md路径>
   
   # 示例
   mineru -p "surveys/cs/topic/paper.pdf" -o "surveys/cs/topic/paper.md"
   ```

4. **批量 PDF 处理**：
   ```bash
   # 处理目录中的所有 PDF
   for pdf in surveys/cs/topic/*.pdf; do
       mineru -p "$pdf" -o "${pdf%.pdf}.md"
   done
   ```

**注意**：MinerU 是一个高质量的 PDF 到 Markdown 转换器，支持复杂布局、表格和数学公式。详细的安装和使用说明请参考 [MinerU 文档](https://github.com/opendatalab/MinerU)。

### 环境设置

创建 `.env` 文件：

```bash
OPENAI_API_KEY=your_api_key
OPENAI_API_BASE=your_api_base_url
MODEL=[MODEL_NAME]  # 相似性计算时更改为嵌入模型
```
