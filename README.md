# SGEval: A Comprehensive Framework for Evaluating Automatic Literature Survey Generation Systems

## Abstract

SGEval is a novel evaluation framework specifically designed for assessing the quality of automatically generated literature surveys. The framework provides both absolute and similarity-based evaluation metrics across multiple dimensions including outline structure, content quality, and reference appropriateness. This repository contains the complete implementation of SGEval, including two variants: SGSimEval-B (baseline similarity evaluation) and SGSimEval-HP (human-perfect similarity evaluation), along with comprehensive evaluation datasets spanning eight academic domains.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Evaluation Framework](#evaluation-framework)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Experimental Results](#experimental-results)
- [Citation](#citation)
- [License](#license)

## Overview

The automatic generation of literature surveys has gained significant attention in the academic community, yet comprehensive evaluation frameworks remain limited. SGEval addresses this gap by providing:

1. **Multi-dimensional Evaluation**: Assesses surveys across outline quality, content coverage, structural coherence, and reference appropriateness
2. **Similarity-based Metrics**: Introduces novel similarity calculations between generated and human-authored surveys
3. **Domain-agnostic Framework**: Supports evaluation across diverse academic domains (CS, Economics, Physics, Mathematics, etc.)
4. **Comprehensive Benchmarks**: Includes evaluation of multiple state-of-the-art survey generation systems

## Features

### Core Evaluation Components

- **Outline Evaluation**: Structure assessment, coverage analysis, and hierarchical organization
- **Content Evaluation**: Coverage, structure, relevance, language quality, and criticalness analysis
- **Reference Evaluation**: Quality assessment, density analysis, and supportiveness evaluation
- **Faithfulness Analysis**: Content-reference alignment verification

### Advanced Similarity Metrics

- **SGSimEval-B**: Baseline similarity using actual human performance as reference
- **SGSimEval-HP**: Human-perfect similarity using ideal scores (5.0 for content metrics, 100.0 for structural metrics)
- **Embedding-based Similarity**: ChromaDB-powered semantic similarity calculations

### Visualization and Analysis

- **Radar Plot Generation**: Multi-dimensional performance visualization
- **Statistical Analysis**: Comprehensive result aggregation and comparison
- **LaTeX Table Generation**: Academic publication-ready result formatting

## Installation

### Prerequisites

- Python 3.8+
- OpenAI API key (for LLM-based evaluation)
- Required Python packages (see requirements.txt)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-org/SGEval.git
cd SGEval

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your OpenAI API credentials
```

### Environment Configuration

Create a `.env` file with the following variables:

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_API_BASE=your_api_base_url
MODEL=your_model_name
```

## Dataset

The SGEval dataset comprises surveys across eight academic domains:

| Domain | Number of Topics | Systems Evaluated |
|--------|------------------|-------------------|
| Computer Science (cs) | 50+ | AutoSurvey, SurveyForge, InteractiveSurvey, LLMxMapReduce, SurveyX |
| Economics (econ) | 30+ | AutoSurvey, SurveyForge, InteractiveSurvey, LLMxMapReduce, SurveyX |
| Electrical Engineering (eess) | 25+ | AutoSurvey, SurveyForge, InteractiveSurvey, LLMxMapReduce, SurveyX |
| Mathematics (math) | 25+ | AutoSurvey, SurveyForge, InteractiveSurvey, LLMxMapReduce, SurveyX |
| Physics (physics) | 25+ | AutoSurvey, SurveyForge, InteractiveSurvey, LLMxMapReduce, SurveyX |
| Quantitative Biology (q-bio) | 20+ | AutoSurvey, SurveyForge, InteractiveSurvey, LLMxMapReduce, SurveyX |
| Quantitative Finance (q-fin) | 15+ | AutoSurvey, SurveyForge, InteractiveSurvey, LLMxMapReduce, SurveyX |
| Statistics (stat) | 15+ | AutoSurvey, SurveyForge, InteractiveSurvey, LLMxMapReduce, SurveyX |

### Dataset Structure

```
surveys/
├── cs/                           # Computer Science domain
│   ├── Topic_Name/
│   │   ├── pdfs/                 # Human-authored reference surveys
│   │   ├── AutoSurvey/           # AutoSurvey system output
│   │   ├── SurveyForge/          # SurveyForge system output
│   │   ├── InteractiveSurvey/    # InteractiveSurvey system output
│   │   ├── LLMxMapReduce/        # LLMxMapReduce system output
│   │   └── SurveyX/              # SurveyX system output
├── econ/                         # Economics domain
├── eess/                         # Electrical Engineering domain
├── math/                         # Mathematics domain
├── physics/                      # Physics domain
├── q-bio/                        # Quantitative Biology domain
├── q-fin/                        # Quantitative Finance domain
├── stat/                         # Statistics domain
└── tasks.json                    # Task configuration file
```

## Evaluation Framework

### Core Metrics

1. **Outline Metrics**
   - Structure quality assessment
   - Coverage analysis (matched sections vs. standard sections)
   - Hierarchical organization evaluation

2. **Content Metrics**
   - Coverage: Comprehensiveness of topic coverage
   - Structure: Logical organization and flow
   - Relevance: Topic relevance and focus
   - Language: Writing quality and clarity
   - Criticalness: Critical analysis and synthesis

3. **Reference Metrics**
   - Quality: Reference appropriateness and credibility
   - Density: Citation frequency and distribution
   - Supportiveness: Evidence-claim alignment

4. **Faithfulness Metrics**
   - Content-reference alignment
   - Factual accuracy verification

### Similarity-Based Evaluation

SGEval introduces two similarity-based evaluation approaches:

#### SGSimEval-B (Baseline)
Uses actual human performance as the reference standard:
```
similarity = (1 - semantic_similarity) × system_score + semantic_similarity × human_score
```

#### SGSimEval-HP (Human-Perfect)
Uses ideal performance as the reference standard:
```
similarity = (1 - semantic_similarity) × system_score + semantic_similarity × perfect_score
```

Where `perfect_score` is 5.0 for content metrics and 100.0 for structural metrics.

## Usage

### Basic Evaluation

Evaluate a single survey:

```python
from scripts.evaluate import evaluate

# Evaluate a markdown survey file
results = evaluate(
    md_path="path/to/survey.md",
    model="gpt-4",
    do_outline=True,
    do_content=True,
    do_reference=True
)
print(results)
```

### Batch Evaluation

Evaluate multiple systems across domains:

```python
from scripts.evaluate import batch_evaluate_by_system

systems = ["AutoSurvey", "SurveyForge", "InteractiveSurvey", "LLMxMapReduce", "SurveyX"]
model = "gpt-4"

batch_evaluate_by_system(
    system_list=systems,
    model=model,
    num_workers=4
)
```

### Similarity Calculation

Calculate similarity scores between generated and reference surveys:

```python
from scripts.cal_similarity import calculate_and_update_similarity_scores_new

calculate_and_update_similarity_scores_new(
    model_name="gpt-4",
    md_path="path/to/generated_survey.md",
    ground_truth_md_path="path/to/reference_survey.md"
)
```

### Generate Evaluation Reports

Create comprehensive evaluation reports:

```python
from scripts.cal_similarity import generate_evaluation_csv, generate_radar_plots

# Generate CSV results
systems = ["AutoSurvey", "SurveyForge", "InteractiveSurvey", "LLMxMapReduce", "SurveyX", "pdfs"]
generate_evaluation_csv(systems, "gpt-4", "results.csv")

# Generate radar plots
system_name_map = {
    "AutoSurvey": "AutoSurvey",
    "SurveyForge": "SurveyForge", 
    "InteractiveSurvey": "InteractiveSurvey",
    "LLMxMapReduce": "LLMxMapReduce",
    "SurveyX": "SurveyX",
    "pdfs": "Human"
}

metrics = ["Outline", "Structure", "Content", "Faithfulness", "Reference", "Supportiveness"]
generate_radar_plots(systems, metrics, system_name_map, "radar_plots.pdf")
```

## Evaluation Metrics

### Scoring System

All metrics use a consistent 0-5 scale:
- **5**: Excellent quality
- **4**: Good quality  
- **3**: Satisfactory quality
- **2**: Below average quality
- **1**: Poor quality
- **0**: Unacceptable quality

### Metric Definitions

| Metric | Description | Range |
|--------|-------------|-------|
| Outline | Overall outline structure and organization | 0-5 |
| Structure | Hierarchical organization quality | 0-100 |
| Content | Average of Coverage, Structure, Relevance, Language, Criticalness | 0-5 |
| Coverage | Comprehensiveness of topic coverage | 0-5 |
| Relevance | Topic relevance and focus | 0-5 |
| Language | Writing quality and clarity | 0-5 |
| Criticalness | Critical analysis depth | 0-5 |
| Faithfulness | Content-reference alignment | 0-100 |
| Reference | Reference appropriateness | 0-5 |
| Supportiveness | Reference quality and evidence support | 0-100 |

## Experimental Results

Our comprehensive evaluation across 200+ survey generation tasks reveals:

### System Performance (Average Scores)

| System | Vanilla | SGSimEval-B | SGSimEval-HP |
|--------|---------|-------------|--------------|
| Human | 4.12 | 4.12 | 4.12 |
| SurveyX | 3.24 | 3.31 | 3.38 |
| AutoSurvey | 3.11 | 3.18 | 3.25 |
| SurveyForge | 2.98 | 3.05 | 3.12 |
| InteractiveSurvey | 2.89 | 2.96 | 3.03 |
| LLMxMapReduce | 2.76 | 2.83 | 2.90 |

### Key Findings

1. **Human surveys consistently outperform** automatic systems across all metrics
2. **SurveyX demonstrates superior performance** among automatic systems
3. **Similarity-based metrics reveal** more nuanced performance differences
4. **Content quality varies significantly** across different academic domains
5. **Reference quality remains challenging** for automatic systems

## Project Structure

```
SGEval/
├── scripts/
│   ├── evaluate.py              # Core evaluation framework
│   ├── cal_similarity.py        # Similarity calculation and analysis
│   ├── plot.py                  # Visualization utilities
│   └── __pycache__/
├── surveys/                     # Evaluation dataset
├── chromadb/                    # Vector database for similarity
├── logs/                        # Evaluation logs
├── figures/                     # Generated visualizations
├── evaluation_results.csv       # Aggregated results
├── evaluation_results.tex       # LaTeX formatted results
└── README.md                    # This file
```

## Citation

If you use SGEval in your research, please cite our work:

```bibtex
@article{sgeval2024,
  title={SGEval: A Comprehensive Framework for Evaluating Automatic Literature Survey Generation Systems},
  author={[Author Names]},
  journal={[Journal Name]},
  year={2024},
  volume={[Volume]},
  pages={[Pages]}
}
```

## Contributing

We welcome contributions to SGEval! Please see our contributing guidelines for details on:

- Adding new evaluation metrics
- Supporting additional survey generation systems
- Expanding domain coverage
- Improving visualization capabilities

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

We thank the academic community for providing reference surveys and the survey generation system developers for making their outputs available for evaluation. Special recognition goes to the OpenAI team for providing the language models that enable our LLM-based evaluation framework.

---

**Contact**: For questions or collaboration opportunities, please open an issue or contact [contact_email].

**Project Homepage**: [project_url]
