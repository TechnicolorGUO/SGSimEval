"""
Utility modules for SGSimEval.
This package contains helper functions, prompt templates, and configuration settings.
"""

from .helpers import (
    getClient,
    generateResponse,
    robust_json_parse,
    extract_topic_from_path,
    read_md,
    count_sentences,
    count_md_features,
    build_outline_tree_from_levels,
    extract_and_save_outline_from_md,
    refine_outline_if_single_level,
    fill_single_criterion_prompt
)

from .prompts import (
    OUTLINE_EVAL_PROMPT,
    CONTENT_EVAL_PROMPT,
    REFERENCE_EVAL_PROMPT,
    SIMILARITY_EVAL_PROMPT,
    OUTLINE_CRITERIA,
    CONTENT_CRITERIA,
    REFERENCE_CRITERIA,
    SIMILARITY_CRITERIA
)

from .config import (
    PROJECT_ROOT,
    DATA_DIR,
    OUTPUT_DIR,
    RESULTS_DIR,
    PLOTS_DIR,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_MODEL,
    OUTLINE_WEIGHTS,
    CONTENT_WEIGHTS,
    REFERENCE_WEIGHTS,
    SIMILARITY_WEIGHTS,
    check_environment
)

__all__ = [
    # Helper functions
    'getClient',
    'generateResponse',
    'robust_json_parse',
    'extract_topic_from_path',
    'read_md',
    'count_sentences',
    'count_md_features',
    'build_outline_tree_from_levels',
    'extract_and_save_outline_from_md',
    'refine_outline_if_single_level',
    'fill_single_criterion_prompt',
    
    # Prompt templates
    'OUTLINE_EVAL_PROMPT',
    'CONTENT_EVAL_PROMPT',
    'REFERENCE_EVAL_PROMPT',
    'SIMILARITY_EVAL_PROMPT',
    'OUTLINE_CRITERIA',
    'CONTENT_CRITERIA',
    'REFERENCE_CRITERIA',
    'SIMILARITY_CRITERIA',
    
    # Configuration
    'PROJECT_ROOT',
    'DATA_DIR',
    'OUTPUT_DIR',
    'RESULTS_DIR',
    'PLOTS_DIR',
    'DEFAULT_MAX_TOKENS',
    'DEFAULT_TEMPERATURE',
    'DEFAULT_MODEL',
    'OUTLINE_WEIGHTS',
    'CONTENT_WEIGHTS',
    'REFERENCE_WEIGHTS',
    'SIMILARITY_WEIGHTS',
    'check_environment'
] 