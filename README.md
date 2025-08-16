# SGSimEval: A Comprehensive Multifaceted and Similarity-Enhanced Benchmark for Automatic Survey Generation Systems

[English](README.md) | [中文](README_CN.md)

## Abstract

SGSimEval is a comprehensive benchmark for evaluating automatic survey generation systems that integrates assessments of outline, content, and references while combining LLM-based scoring with quantitative metrics. The framework introduces human preference metrics that emphasize both inherent quality and similarity to humans, providing a multifaceted evaluation approach.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Evaluation Framework](#evaluation-framework)
- [Usage Guide](#usage-guide)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

## Overview

The growing interest in automatic survey generation (ASG) has been spurred by recent advances in large language models (LLMs), retrieval-augmented generation (RAG), and multi-agent systems (MASs). SGSimEval addresses the limitations of existing evaluation methods by providing:

1. **Comprehensive Evaluation**: Assesses surveys across outline quality, content adequacy, and reference appropriateness
2. **Similarity-Enhanced Metrics**: Introduces novel similarity calculations between generated and human-authored surveys
3. **Human Preference Alignment**: Incorporates human preference metrics for better evaluation
4. **Multi-dimensional Assessment**: Combines LLM-based scoring with quantitative metrics

## Features

### Core Evaluation Components

- **Outline Evaluation (SGSimEval-Outline)**
  - Hierarchical structure analysis
  - Logical coherence assessment
  - LLM-based quality evaluation

- **Content Evaluation (SGSimEval-Content)**
  - Citation faithfulness evaluation
  - Comprehensive LLM-based quality assessment
  - Coverage, Structure, Relevance, Language, and Criticalness analysis

- **Reference Evaluation (SGSimEval-Reference)**
  - Supportiveness analysis
  - Comprehensive LLM-based reference assessment
  - Quality and relevance evaluation

### Similarity-Enhanced Evaluation

SGSimEval introduces two similarity-enhanced approaches:

1. **Human-as-Perfect Similarity Weighting**
   - Treats human-authored content as ideal reference
   - Emphasizes human-like quality
   - Uses embedding-based similarity calculations

2. **Balanced Similarity Weighting**
   - Considers both semantic similarity and actual quality scores
   - Provides more nuanced evaluation
   - Incorporates actual human reference quality

## Installation

### Prerequisites

- Python 3.8+
- Required Python packages (see requirements.txt)
- Qwen-Plus-2025-04-28 for LLM-based evaluation
- Alibaba's Text-Embedding-V3 for vector embeddings

### Setup

```bash
# Clone the repository
git clone https://github.com/your-org/SGSimEval.git
cd SGSimEval

# Install dependencies
pip install -r requirements.txt
```

### Environment Configuration

Copy `env_example.txt` to `.env` and configure your API keys:

```bash
cp env_example.txt .env
# Edit the .env file to add your OpenAI API key
```

## Dataset

The SGSimEval dataset comprises 80 highly-cited survey papers from arXiv spanning various domains, all published within the last three years. The dataset includes:

- Computer Science domain surveys
- Various academic domains
- Human-authored reference surveys
- System-generated surveys from:
  - AutoSurvey
  - SurveyForge
  - InteractiveSurvey
  - LLMxMapReduce-V2
  - SurveyX

## Evaluation Framework

### Core Metrics

1. **Outline Metrics**
   - Structure quality (0-100)
   - Hierarchical organization
   - LLM-assessed quality

2. **Content Metrics**
   - Coverage (0-5)
   - Structure (0-5)
   - Relevance (0-5)
   - Language (0-5)
   - Criticalness (0-5)
   - Faithfulness (0-100)

3. **Reference Metrics**
   - Quality (0-5)
   - Supportiveness (0-100)
   - LLM-assessed quality

## Usage Guide

### Data Preparation

Organize your data in the `surveys` folder with the following structure:

```
surveys/
├── cs/                          # Domain classification (e.g., cs, econ, physics)
│   ├── Topic Name 1/           # Specific topic
│   │   ├── pdfs/               # Human-authored reference papers
│   │   │   ├── paper1.md       # Converted markdown from PDF
│   │   │   ├── paper1.csv      # Extracted citations
│   │   │   └── references.json # Reference list
│   │   ├── AutoSurvey/         # System-generated surveys
│   │   │   ├── Topic Name 1.md
│   │   │   ├── outline.json    # Outline structure
│   │   │   ├── references.json # References
│   │   │   └── results_*.json  # Evaluation results
│   │   ├── InteractiveSurvey/
│   │   ├── SurveyForge/
│   │   ├── LLMxMapReduce/
│   │   └── SurveyX/
│   └── Topic Name 2/
├── econ/
└── ...
```

### Running Evaluations

#### Method 1: Using the Simplified Runner (Recommended)

**Single File Evaluation:**
```bash
python run_evaluation.py --survey_path "surveys/cs/Topic Name/AutoSurvey/Topic Name.md"
```

**Single File Evaluation with Similarity Calculation:**
```bash
python run_evaluation.py --survey_path "surveys/cs/Topic Name/AutoSurvey/Topic Name.md" \
    --calculate_similarity \
    --ground_truth_path "surveys/cs/Topic Name/pdfs/paper1.md" \
    --similarity_type balanced
```

**Batch Evaluation for a Category:**
```bash
python run_evaluation.py --batch_category --category cs --model gpt-4
```

**Batch Evaluation for a Specific System:**
```bash
python run_evaluation.py --batch_system --system AutoSurvey --model gpt-4
```

**Batch Similarity Calculation:**
```bash
python run_evaluation.py --batch_similarity \
    --systems AutoSurvey SurveyForge InteractiveSurvey \
    --domains cs econ \
    --model gpt-4
```

**Custom Evaluation Options:**
```bash
python run_evaluation.py --batch_category --category cs \
    --do_outline --do_content --do_reference \
    --criteria_type domain \
    --num_workers 4
```

#### Method 2: Direct Use of evaluate.py

```bash
cd scripts
python evaluate.py
```

Modify the parameters in the `__main__` section of `evaluate.py` to run different evaluation tasks.

### Evaluation Features

#### 1. Outline Evaluation
- **LLM Assessment**: Quality evaluation based on predefined criteria
- **Coverage Assessment**: Degree of coverage of standard sections
- **Structure Assessment**: Logicality and organization of hierarchical structure
- **Density Assessment**: Relationship between outline density and content length

#### 2. Content Evaluation
- **LLM Assessment**: Coverage, Structure, Relevance, Language, Criticalness
- **Information Density**: Density of images, equations, tables, citations
- **Faithfulness**: Consistency between content and references
- **Sentence Count**: Content length statistics

#### 3. Reference Evaluation
- **LLM Assessment**: Reference quality evaluation
- **Density Assessment**: Reference density
- **Quality Assessment**: Relevance and supportiveness of references
- **Count Statistics**: Reference quantity

#### 4. Similarity Evaluation
- **Human Preference**: Similarity to human-authored content
- **Balanced Weighting**: Balance between semantic similarity and actual quality

### Output Results

#### Single File Evaluation Results

Evaluation results are saved in `results_<model>.json` files, containing:

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

#### Similarity-Enhanced Results

When similarity calculation is enabled, additional scores are added:

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

#### Batch Evaluation Results

- **Category Average Results**: `surveys/<category>/average_results.json`
- **Global Average Results**: `surveys/global_average.csv`
- **All Categories Results**: `surveys/all_categories_results.csv`

### Advanced Features

#### 1. Calculate Average Scores
```python
from scripts.evaluate import calculate_average_score_by_cat
calculate_average_score_by_cat("cs")
```

#### 2. Aggregate Results to CSV
```python
from scripts.evaluate import aggregate_results_to_csv
aggregate_results_to_csv("cs")
```

#### 3. Convert to LaTeX Format
```python
from scripts.evaluate import convert_to_latex
convert_to_latex()
```

#### 4. Similarity Calculation
```python
from scripts.cal_similarity import batch_calculate_similarity_scores
batch_calculate_similarity_scores(["AutoSurvey", "SurveyForge"], "gpt-4", ["cs"])
```

### Troubleshooting

#### Common Issues

1. **Import Errors**: Ensure you're running scripts in the correct directory
2. **API Errors**: Check API key configuration in the `.env` file
3. **File Path Errors**: Ensure correct surveys folder structure
4. **Memory Issues**: Reduce the `num_workers` parameter

#### Log Files

- Evaluation logs: `judge.log`
- Similarity logs: `similarity_calculations.log`

### Example Workflow

1. **Prepare Data**: Place survey files in the correct directory structure
2. **Run Evaluation**: Use batch evaluation commands
3. **Calculate Similarity**: Run similarity calculation with ground truth
4. **View Results**: Check generated JSON and CSV files
5. **Generate Reports**: Use LaTeX conversion functionality
6. **Analyze Similarity**: Review similarity-enhanced scores

### Supported Models

- GPT-4
- GPT-3.5-turbo
- Qwen-Plus
- Other OpenAI API-compatible models

### Important Notes

- Ensure sufficient API quota
- Use parallel processing for large datasets
- Regularly backup evaluation results
- Monitor API call costs

## Results

Our evaluation reveals several key findings:

1. **Domain-Specific Performance**
   - CS-specialized systems consistently outperform general-domain systems
   - SurveyForge achieves better outline structure and reference quality
   - AutoSurvey shows comparable content quality

2. **Human Comparison**
   - Most ASG systems surpass humans in outline generation
   - CS-specific systems outperform humans in content quality
   - Human-generated references maintain higher quality

3. **Similarity Analysis**
   - Outline similarity: 0.54-0.63
   - Content similarity: 0.57-0.67
   - Reference similarity: 0.32-0.61

## Citation

If you use SGSimEval in your research, please cite our work:

```bibtex
@article{sgsimeval2024,
  title={SGSimEval: A Comprehensive Multifaceted and Similarity-Enhanced Benchmark for Automatic Survey Generation Systems},
  author={[Author Names]},
  journal={[Journal Name]},
  year={2024}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

We thank the academic community for providing reference surveys and the survey generation system developers for making their outputs available for evaluation.

---

**Contact**: For questions or collaboration opportunities, please open an issue or contact [contact_email].

**Project Homepage**: [project_url]
