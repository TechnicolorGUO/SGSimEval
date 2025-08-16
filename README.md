# SGSimEval - Survey Generation Similarity Evaluation System

### Overview

SGSimEval is a comprehensive evaluation framework for survey generation systems that combines traditional evaluation metrics with similarity-based assessment methods. It provides both regular evaluation scores and similarity-adjusted scores using two approaches: balanced similarity and human-perfect similarity.

### Key Features

- **Multi-dimensional Evaluation**: Outline, content, references, and more
- **Similarity-based Scoring**: Two similarity methods (balanced and human-perfect)
- **Batch Processing**: Support for large-scale evaluation across multiple systems
- **Comprehensive Analysis**: Average scores, aggregation, and comparison tools
- **Flexible Output**: CSV, LaTeX, and JSON formats

### Quick Start Example

If you have already prepared the dataset structure with CS category data in the `surveys/` folder, you can run the following commands in sequence to get evaluation results:

```bash
# Step 1: Evaluate survey quality for CS category
python main.py batch-evaluate --mode category --categories cs --model "[MODEL_NAME]" --workers 4

# Step 2: Calculate similarity scores between systems and ground truth
python main.py similarity --systems AutoSurvey SurveyForge InteractiveSurvey LLMxMapReduce SurveyX --model [MODEL_NAME]

# Step 3: Calculate all scores and generate final results CSV
python main.py calculate-all --categories cs --systems AutoSurvey SurveyForge InteractiveSurvey LLMxMapReduce SurveyX --models [MODEL_NAME] --workers 4
```

This will generate the final results CSV file with comprehensive evaluation scores.

### Complete Workflow Example

#### Step 1: Prepare Dataset Structure

First, you need to prepare the dataset structure using `get_topics.py`:

```bash
# Generate dataset structure and create system folders
python scripts/get_topics.py --systems AutoSurvey SurveyForge InteractiveSurvey LLMxMapReduce SurveyX --numofsurvey 50
```

This will:
- Create the necessary folder structure in `surveys/`
- Generate `surveys/tasks.json` for task mapping
- Set up system subfolders for each topic

**Expected folder structure:**
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

#### Step 2: Evaluate Survey Quality

Run batch evaluation to assess the quality of generated surveys:

```bash
# Evaluate all systems for CS category
python main.py batch-evaluate --mode system --systems AutoSurvey SurveyForge InteractiveSurvey LLMxMapReduce SurveyX --model [MODEL_NAME] --workers 4
```

This generates `results_[MODEL_NAME].json` files in each system folder containing evaluation scores.

#### Step 3: Calculate Similarity Scores

**Important**: Before running similarity calculation, ensure your `.env` file uses an embedding model:

```bash
# In your .env file, change the model to an embedding model
MODEL=text-embedding-3-large  # or another embedding model
```

Then calculate similarity scores:

```bash
# Calculate similarity scores (both balanced and human-perfect)
python main.py similarity --systems AutoSurvey SurveyForge InteractiveSurvey LLMxMapReduce SurveyX --model [MODEL_NAME]
```

This will:
- Calculate embeddings for outlines, content, and references
- Generate similarity scores between system outputs and ground truth
- Update `results_[MODEL_NAME].json` with similarity-adjusted scores
- Create `similarity.json` files with raw similarity data

#### Step 4: Calculate All Scores and Generate Results

```bash
# Calculate comprehensive scores and generate final results
python main.py calculate-all --categories cs --systems AutoSurvey SurveyForge InteractiveSurvey LLMxMapReduce SurveyX --models [MODEL_NAME] --workers 4
```

This generates:
- `surveys/cs_results.csv`: Category-level results
- `surveys/cs_average.csv`: Category averages
- `surveys/global_average.csv`: Global averages across all categories

### Understanding Results

#### Global Average CSV Columns

The `surveys/global_average.csv` file contains the following columns:

| Column | Description |
|--------|-------------|
| `System` | System name (AutoSurvey, SurveyForge, etc.) |
| `Model` | Evaluation model used ([MODEL_NAME]) |
| `Outline` | Original outline evaluation score (0-5) |
| `Content` | Original content evaluation score (0-5) |
| `Reference` | Original reference evaluation score (0-5) |
| `Outline_sim` | **Balanced similarity-adjusted outline score** |
| `Content_sim` | **Balanced similarity-adjusted content score** |
| `Reference_sim` | **Balanced similarity-adjusted reference score** |
| `Outline_sim_hp` | **Human-perfect similarity-adjusted outline score** |
| `Content_sim_hp` | **Human-perfect similarity-adjusted content score** |
| `Reference_sim_hp` | **Human-perfect similarity-adjusted reference score** |
| `Outline_structure` | Original outline structure score (0-100) |
| `Faithfulness` | Original faithfulness score (0-100) |
| `Reference_quality` | Original reference quality score (0-100) |
| `Outline_structure_sim` | **Balanced similarity-adjusted outline structure score** |
| `Faithfulness_sim` | **Balanced similarity-adjusted faithfulness score** |
| `Reference_quality_sim` | **Balanced similarity-adjusted reference quality score** |
| `Outline_structure_sim_hp` | **Human-perfect similarity-adjusted outline structure score** |
| `Faithfulness_sim_hp` | **Human-perfect similarity-adjusted faithfulness score** |
| `Reference_quality_sim_hp` | **Human-perfect similarity-adjusted reference quality score** |

**Similarity Score Explanation:**
- **Balanced (`_sim`)**: Combines original score with ground truth score based on similarity
- **Human-Perfect (`_sim_hp`)**: Combines original score with perfect score (5.0/100.0) based on similarity

### Command Reference

#### Evaluation Commands

```bash
# Single file evaluation
python main.py evaluate --md-path "surveys/cs/topic/system/file.md" --model "[MODEL_NAME]"

# Batch evaluation by category
python main.py batch-evaluate --mode category --categories cs econ --model "[MODEL_NAME]" --workers 4

# Batch evaluation by system
python main.py batch-evaluate --mode system --systems AutoSurvey SurveyForge --model "[MODEL_NAME]" --workers 4
```

#### Similarity Commands

```bash
# Batch similarity calculation
python main.py similarity --systems AutoSurvey SurveyForge --model "[MODEL_NAME]"

# Single file similarity
python main.py similarity --mode single --md-path "path/to/system.md" --gt-path "path/to/ground_truth.md" --model "[MODEL_NAME]"
```

#### Analysis Commands

```bash
# Calculate averages
python main.py average --mode category --categories cs econ

# Aggregate results
python main.py aggregate --mode category --categories cs econ

# Comprehensive calculation
python main.py calculate-all --categories cs --systems AutoSurvey SurveyForge --models [MODEL_NAME] --workers 4
```

#### Utility Commands

```bash
# Clear scores
python main.py clear --mode scores --categories cs --systems AutoSurvey --model "[MODEL_NAME]"

# Generate reports
python main.py generate --mode evaluation --systems AutoSurvey SurveyForge --model "[MODEL_NAME]"
```

### Installation

```bash
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file with:

```bash
OPENAI_API_KEY=your_api_key
OPENAI_API_BASE=your_api_base_url
MODEL=[MODEL_NAME]  # Change to embedding model for similarity calculation
```
