"""
Configuration settings for SGSimEval.
Contains constants, default values, and configuration settings used throughout the project.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
RESULTS_DIR = OUTPUT_DIR / "results"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Create directories if they don't exist
for directory in [DATA_DIR, OUTPUT_DIR, RESULTS_DIR, PLOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# File patterns
MD_PATTERN = "*.md"
JSON_PATTERN = "*.json"
CSV_PATTERN = "*.csv"

# Evaluation settings
DEFAULT_MAX_TOKENS = 8192
DEFAULT_TEMPERATURE = 0.5
DEFAULT_MODEL = "gpt-4-turbo-preview"

# Evaluation criteria weights
OUTLINE_WEIGHTS = {
    "structure": 0.4,
    "completeness": 0.3,
    "coherence": 0.3
}

CONTENT_WEIGHTS = {
    "depth": 0.4,
    "clarity": 0.3,
    "originality": 0.3
}

REFERENCE_WEIGHTS = {
    "coverage": 0.4,
    "quality": 0.3,
    "integration": 0.3
}

SIMILARITY_WEIGHTS = {
    "content_overlap": 0.4,
    "structure_similarity": 0.3,
    "reference_overlap": 0.3
}

# Visualization settings
PLOT_STYLE = "seaborn"
FIGURE_SIZE = (12, 8)
DPI = 300
COLOR_PALETTE = "viridis"
FONT_SIZE = 12
TITLE_FONT_SIZE = 14

# Output file names
OUTLINE_EVAL_FILE = "outline_evaluation.json"
CONTENT_EVAL_FILE = "content_evaluation.json"
REFERENCE_EVAL_FILE = "reference_evaluation.json"
SIMILARITY_EVAL_FILE = "similarity_evaluation.json"
COMBINED_EVAL_FILE = "combined_evaluation.json"

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = OUTPUT_DIR / "evaluation.log"

# API settings
API_TIMEOUT = 300  # 5 minutes
API_RETRY_ATTEMPTS = 3
API_RETRY_DELAY = 5  # seconds

# Text processing settings
MAX_SENTENCE_LENGTH = 1000
MIN_SENTENCE_LENGTH = 10
MAX_PARAGRAPH_LENGTH = 5000

# Similarity thresholds
HIGH_SIMILARITY_THRESHOLD = 0.8
MEDIUM_SIMILARITY_THRESHOLD = 0.5
LOW_SIMILARITY_THRESHOLD = 0.3

# Environment variables
REQUIRED_ENV_VARS = [
    "OPENAI_API_KEY",
    "OPENAI_API_BASE",
    "MODEL"
]

def check_environment():
    """Check if all required environment variables are set."""
    missing_vars = [var for var in REQUIRED_ENV_VARS if not os.environ.get(var)]
    if missing_vars:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        ) 