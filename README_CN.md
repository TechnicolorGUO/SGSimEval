# SGSimEval: 自动综述生成系统的全面多维度相似性增强基准测试

[English](README.md) | [中文](README_CN.md)

## 摘要

SGSimEval 是一个全面的自动综述生成系统评估基准测试，集成了大纲、内容和参考文献的评估，同时结合了基于LLM的评分和定量指标。该框架引入了强调内在质量和与人类相似性的人类偏好指标，提供了多维度评估方法。

## 目录

- [概述](#概述)
- [功能特性](#功能特性)
- [安装](#安装)
- [数据集](#数据集)
- [评估框架](#评估框架)
- [使用指南](#使用指南)
- [结果](#结果)
- [引用](#引用)
- [许可证](#许可证)

## 概述

随着大语言模型（LLMs）、检索增强生成（RAG）和多智能体系统（MASs）的最新进展，自动综述生成（ASG）领域引起了越来越多的关注。SGSimEval 通过提供以下功能解决了现有评估方法的局限性：

1. **全面评估**: 评估综述的大纲质量、内容充分性和参考文献适当性
2. **相似性增强指标**: 引入生成综述与人类撰写综述之间的新颖相似性计算
3. **人类偏好对齐**: 纳入人类偏好指标以获得更好的评估
4. **多维度评估**: 结合基于LLM的评分和定量指标

## 功能特性

### 核心评估组件

- **大纲评估 (SGSimEval-Outline)**
  - 层次结构分析
  - 逻辑连贯性评估
  - 基于LLM的质量评估

- **内容评估 (SGSimEval-Content)**
  - 引用忠实度评估
  - 全面的基于LLM的质量评估
  - 覆盖度、结构、相关性、语言和批判性分析

- **参考文献评估 (SGSimEval-Reference)**
  - 支持性分析
  - 全面的基于LLM的参考文献评估
  - 质量和相关性评估

### 相似性增强评估

SGSimEval 引入了两种相似性增强方法：

1. **以人类为完美的相似性权重**
   - 将人类撰写的内容视为理想参考
   - 强调类人类质量
   - 使用基于嵌入的相似性计算

2. **平衡相似性权重**
   - 同时考虑语义相似性和实际质量分数
   - 提供更细致的评估
   - 纳入实际人类参考质量

## 安装

### 前置要求

- Python 3.8+
- 所需的Python包（见requirements.txt）

### 设置

```bash
# 克隆仓库
git clone https://github.com/TechnicolorGUO
cd SGSimEval

# 安装依赖
pip install -r requirements.txt
```

### 环境配置

复制 `env_example.txt` 为 `.env` 并配置你的API密钥：

```bash
cp env_example.txt .env
# 编辑 .env 文件，添加你的 OpenAI API 密钥
```

## 数据集

SGSimEval 数据集包含来自arXiv的80篇高引用综述论文，涵盖各个领域，均在最近三年内发表。数据集包括：

- 计算机科学领域综述
- 各种学术领域
- 人类撰写的参考综述
- 系统生成的综述，来自：
  - AutoSurvey
  - SurveyForge
  - InteractiveSurvey
  - LLMxMapReduce-V2
  - SurveyX

## 评估框架

### 核心指标

1. **大纲指标**
   - 结构质量 (0-100)
   - 层次组织
   - LLM评估质量

2. **内容指标**
   - 覆盖度 (0-5)
   - 结构 (0-5)
   - 相关性 (0-5)
   - 语言 (0-5)
   - 批判性 (0-5)
   - 忠实度 (0-100)

3. **参考文献指标**
   - 质量 (0-5)
   - 支持性 (0-100)
   - LLM评估质量

## 使用指南

### 数据准备

在 `surveys` 文件夹中按照以下结构组织数据：

```
surveys/
├── cs/                          # 领域分类 (如: cs, econ, physics)
│   ├── Topic Name 1/           # 具体主题
│   │   ├── pdfs/               # 人类撰写的参考论文
│   │   │   ├── paper1.md       # 转换为markdown的论文
│   │   │   ├── paper1.csv      # 引用提取结果
│   │   │   └── references.json # 参考文献列表
│   │   ├── AutoSurvey/         # 系统生成的综述
│   │   │   ├── Topic Name 1.md
│   │   │   ├── outline.json    # 大纲结构
│   │   │   ├── references.json # 参考文献
│   │   │   └── results_*.json  # 评估结果
│   │   ├── InteractiveSurvey/
│   │   ├── SurveyForge/
│   │   ├── LLMxMapReduce/
│   │   └── SurveyX/
│   └── Topic Name 2/
├── econ/
└── ...
```

### 运行评估

#### 方法1: 使用简化的运行脚本 (推荐)

**单个文件评估:**
```bash
python run_evaluation.py --survey_path "surveys/cs/Topic Name/AutoSurvey/Topic Name.md"
```

**单个文件评估（包含相似性计算）:**
```bash
python run_evaluation.py --survey_path "surveys/cs/Topic Name/AutoSurvey/Topic Name.md" \
    --calculate_similarity \
    --ground_truth_path "surveys/cs/Topic Name/pdfs/paper1.md" \
    --similarity_type balanced
```

**批量评估整个类别:**
```bash
python run_evaluation.py --batch_category --category cs --model gpt-4
```

**批量评估特定系统:**
```bash
python run_evaluation.py --batch_system --system AutoSurvey --model gpt-4
```

**批量相似性计算:**
```bash
python run_evaluation.py --batch_similarity \
    --systems AutoSurvey SurveyForge InteractiveSurvey \
    --domains cs econ \
    --model gpt-4
```

**自定义评估选项:**
```bash
python run_evaluation.py --batch_category --category cs \
    --do_outline --do_content --do_reference \
    --criteria_type domain \
    --num_workers 4
```

#### 方法2: 直接使用 evaluate.py

```bash
cd scripts
python evaluate.py
```

在 `evaluate.py` 的 `__main__` 部分修改参数来运行不同的评估任务。

### 评估功能

#### 1. 大纲评估
- **LLM评估**: 基于预定义标准的大纲质量评估
- **覆盖率评估**: 大纲覆盖标准章节的程度
- **结构评估**: 层次结构的逻辑性和组织性
- **密度评估**: 大纲密度与内容长度的关系

#### 2. 内容评估
- **LLM评估**: 覆盖度、结构、相关性、语言、批判性
- **信息密度**: 图像、公式、表格、引用的密度
- **忠实度**: 内容与参考文献的一致性
- **句子数量**: 内容长度统计

#### 3. 参考文献评估
- **LLM评估**: 参考文献质量评估
- **密度评估**: 参考文献密度
- **质量评估**: 参考文献的相关性和支持性
- **数量统计**: 参考文献数量

#### 4. 相似性评估
- **人类偏好**: 与人类撰写内容的相似性
- **平衡权重**: 语义相似性和实际质量的平衡

### 输出结果

#### 单个文件评估结果

评估结果保存在 `results_<model>.json` 文件中，包含：

```json
{
  "Outline": 85.5,
  "Outline_coverage": 78.2,
  "Outline_structure": 82.1,
  "Coverage": 4.2,
  "Structure": 3.8,
  "Relevance": 4.5,
  "Language": 4.1,
  "Criticalness": 3.9,
  "Faithfulness": 76.8,
  "Reference": 4.3,
  "Reference_quality": 81.2
}
```

#### 相似性增强结果

启用相似性计算时，会添加额外的分数：

```json
{
  "Outline_sim": 87.2,
  "Coverage_sim": 4.5,
  "Structure_sim": 4.1,
  "Relevance_sim": 4.6,
  "Language_sim": 4.3,
  "Criticalness_sim": 4.0,
  "Reference_sim": 4.4,
  "Outline_sim_hp": 88.1,
  "Coverage_sim_hp": 4.7,
  "Structure_sim_hp": 4.2,
  "Relevance_sim_hp": 4.8,
  "Language_sim_hp": 4.4,
  "Criticalness_sim_hp": 4.1,
  "Reference_sim_hp": 4.5
}
```

#### 批量评估结果

- **类别平均结果**: `surveys/<category>/average_results.json`
- **全局平均结果**: `surveys/global_average.csv`
- **所有类别结果**: `surveys/all_categories_results.csv`

### 高级功能

#### 1. 计算平均分数
```python
from scripts.evaluate import calculate_average_score_by_cat
calculate_average_score_by_cat("cs")
```

#### 2. 聚合结果到CSV
```python
from scripts.evaluate import aggregate_results_to_csv
aggregate_results_to_csv("cs")
```

#### 3. 转换为LaTeX格式
```python
from scripts.evaluate import convert_to_latex
convert_to_latex()
```

#### 4. 相似性计算
```python
from scripts.cal_similarity import batch_calculate_similarity_scores
batch_calculate_similarity_scores(["AutoSurvey", "SurveyForge"], "gpt-4", ["cs"])
```

### 故障排除

#### 常见问题

1. **导入错误**: 确保在正确的目录中运行脚本
2. **API错误**: 检查 `.env` 文件中的API密钥配置
3. **文件路径错误**: 确保surveys文件夹结构正确
4. **内存不足**: 减少 `num_workers` 参数

#### 日志文件

- 评估日志: `judge.log`
- 相似性日志: `similarity_calculations.log`

### 示例工作流程

1. **准备数据**: 将综述文件放入正确的目录结构
2. **运行评估**: 使用批量评估命令
3. **计算相似性**: 使用地面真值运行相似性计算
4. **查看结果**: 检查生成的JSON和CSV文件
5. **生成报告**: 使用LaTeX转换功能
6. **分析相似性**: 查看相似性增强分数

### 注意事项

- 确保有足够的API配额
- 大型数据集建议使用并行处理
- 定期备份评估结果
- 注意API调用的成本控制

## 结果

我们的评估揭示了几个关键发现：

1. **领域特定性能**
   - CS专用系统在大多数情况下优于通用领域系统
   - SurveyForge在大纲结构和参考文献质量方面表现更好
   - AutoSurvey在内容质量方面表现相当

2. **与人类比较**
   - 大多数ASG系统在大纲生成方面超越人类
   - CS专用系统在内容质量方面超越人类
   - 人类生成的参考文献保持更高质量

3. **相似性分析**
   - 大纲相似性: 0.54-0.63
   - 内容相似性: 0.57-0.67
   - 参考文献相似性: 0.32-0.61

