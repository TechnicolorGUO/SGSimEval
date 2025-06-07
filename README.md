# SGSimEval: A Comprehensive Multifaceted and Similarity-Enhanced Benchmark for Automatic Survey Generation Systems

## Abstract

SGSimEval is a comprehensive benchmark for evaluating automatic survey generation systems that integrates assessments of outline, content, and references while combining LLM-based scoring with quantitative metrics. The framework introduces human preference metrics that emphasize both inherent quality and similarity to humans, providing a multifaceted evaluation approach.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Evaluation Framework](#evaluation-framework)
- [Usage](#usage)
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
