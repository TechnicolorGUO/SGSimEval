"""
Core evaluation module for SGSimEval.
Contains the main evaluation classes and functions for outline, content, and reference evaluation.
"""

from typing import Dict, List, Optional, Tuple, Union
import json
import os
import math
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import dotenv
from openai import OpenAI

from ..utils.prompts import (
    CONTENT_EVALUATION_PROMPT, CONTENT_FAITHFULNESS_PROMPT,
    OUTLINE_EVALUATION_PROMPT, CRITERIA, OUTLINE_STRUCTURE_PROMPT,
    REFERENCE_EVALUATION_PROMPT, OUTLINE_COVERAGE_PROMPT,
    REFERENCE_QUALITY_PROMPT, OUTLINE_DOMAIN_CRITERIA,
    REFERENCE_DOMAIN_CRITERIA, COVERAGE_DOMAIN_CRITERIA,
    STRUCTURE_DOMAIN_CRITERIA, RELEVANCE_DOMAIN_CRITERIA,
    LANGUAGE_DOMAIN_CRITERIA, CRITICALNESS_DOMAIN_CRITERIA
)
from ..utils.reference import extract_refs, split_markdown_content_and_refs
from ..utils.helpers import (
    build_outline_tree_from_levels, count_md_features,
    count_sentences, extract_and_save_outline_from_md,
    extract_references_from_md, extract_topic_from_path,
    getClient, generateResponse, pdf2md, refine_outline_if_single_level,
    robust_json_parse, fill_single_criterion_prompt, read_md, refine_ref_lists
)

@dataclass
class EvaluationConfig:
    """Configuration for evaluation settings."""
    do_outline: bool = True
    do_content: bool = True
    do_reference: bool = True
    criteria_type: str = "general"
    do_llm: bool = True
    do_coverage: bool = True
    do_structure: bool = True
    do_info: bool = True
    do_faithfulness: bool = True
    do_density: bool = True
    do_quality: bool = True

class Judge:
    """A class to handle LLM-based evaluation using OpenAI's API."""
    def __init__(self) -> None:
        """Initialize the Judge with OpenAI client and logging configuration."""
        dotenv.load_dotenv()
        with open('judge.log', 'w') as log_file:
            log_file.truncate(0)
        self.client = getClient()
        logging.basicConfig(
            filename='judge.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def judge(self, prompt: str) -> Optional[Dict]:
        """Evaluate a prompt using the LLM and return the parsed response."""
        response = generateResponse(self.client, prompt)
        logging.info(f"Response received: {response}")
        try:
            result = robust_json_parse(response)
            return result
        except Exception as e:
            logging.error(f"Error parsing JSON: {e}")
            print("Error parsing JSON:", e)
            return None

class OutlineEvaluator:
    """Evaluates survey outlines."""
    def __init__(self, judge: Optional[Judge] = None):
        self.judge = judge or Judge()

    def evaluate_outline_llm(self, outline_json_path: str, criteria_type: str = "general") -> Dict:
        """Evaluate outline using LLM-based criteria."""
        criteria_name = "Outline"
        results = {}
        try:
            with open(outline_json_path, "r", encoding="utf-8") as f:
                outline_list = json.load(f)

            outline_str = "\n".join([json.dumps(item, ensure_ascii=False) for item in outline_list])
            topic = extract_topic_from_path(outline_json_path)

            if criteria_type == "domain":
                path_parts = outline_json_path.split(os.sep)
                if len(path_parts) > 1:
                    domain = path_parts[1]
                    criterion = OUTLINE_DOMAIN_CRITERIA.get(domain, CRITERIA[criteria_name])
                else:
                    criterion = CRITERIA[criteria_name]
            else:
                criterion = CRITERIA[criteria_name]

            prompt = fill_single_criterion_prompt(
                prompt_template=OUTLINE_EVALUATION_PROMPT,
                content=outline_str,
                topic=topic,
                criterion=criterion,
                criteria_name=criteria_name,
                type="outline"
            )
            score_dict = self.judge.judge(prompt)
            if not (isinstance(score_dict, dict) and criteria_name in score_dict):
                results[criteria_name] = 0
            else:
                results.update(score_dict)
        except Exception as e:
            results[criteria_name] = 0

        if criteria_type == "domain":
            new_results = {}
            for key, value in results.items():
                if key == "Outline":
                    new_results["Outline_domain"] = value
                else:
                    new_results[key] = value
            return new_results
        return results

    def evaluate_outline_coverage(
        self,
        outline_json_path: str,
        standard_count: int = 10,
        enable_penalty: bool = True
    ) -> float:
        """Evaluate outline coverage score."""
        try:
            with open(outline_json_path, "r", encoding="utf-8") as f:
                outline_list = json.load(f)

            level_1_count = sum(1 for item in outline_list if item[0] == 1)
            level_2_count = sum(1 for item in outline_list if item[0] == 2)
            
            if level_1_count == 1:
                total_section_count = level_2_count
            else:
                total_section_count = level_1_count - 1

            outline_str = "\n".join([json.dumps(item, ensure_ascii=False) for item in outline_list])
            topic = extract_topic_from_path(outline_json_path)
            prompt = OUTLINE_COVERAGE_PROMPT.format(
                outline=outline_str,
                topic=topic,
            )
            response = self.judge.judge(prompt)
            matched_count = response.get("matched_count", 0)

            K = matched_count
            N = standard_count
            M = total_section_count

            if N > 0:
                length_ratio = (M - N) / N
                penalty_term = math.exp(-(length_ratio ** 2))
                coverage = (K / N) * penalty_term
                return round(coverage * 100, 4)
            else:
                return 0.0

        except Exception as e:
            print(f"Error evaluating outline coverage: {e}")
            return 0.0

    def evaluate_outline_structure(self, outline_json_path: str) -> Tuple[float, List[Dict]]:
        """Evaluate the hierarchical structure of the outline."""
        try:
            with open(outline_json_path, "r", encoding="utf-8") as f:
                outline_list = json.load(f)

            outline_str = "\n".join([json.dumps(item, ensure_ascii=False) for item in outline_list])
            topic = extract_topic_from_path(outline_json_path)
            prompt = OUTLINE_STRUCTURE_PROMPT.format(
                outline=outline_str,
                topic=topic,
            )
            response = self.judge.judge(prompt)
            
            if not isinstance(response, dict):
                return 0.0, []

            structure_score = response.get("structure_score", 0)
            structure_details = response.get("structure_details", [])
            
            return structure_score, structure_details

        except Exception as e:
            print(f"Error evaluating outline structure: {e}")
            return 0.0, []

class ContentEvaluator:
    """Evaluates survey content."""
    def __init__(self, judge: Optional[Judge] = None):
        self.judge = judge or Judge()

    def evaluate_content_llm(self, md_path: str, criteria_type: str = "general") -> Dict:
        """Evaluate content using LLM-based criteria."""
        results = {}
        try:
            content = read_md(md_path)
            topic = extract_topic_from_path(md_path)

            if criteria_type == "domain":
                path_parts = md_path.split(os.sep)
                if len(path_parts) > 1:
                    domain = path_parts[1]
                    criteria = {
                        "Coverage": COVERAGE_DOMAIN_CRITERIA.get(domain, CRITERIA["Coverage"]),
                        "Structure": STRUCTURE_DOMAIN_CRITERIA.get(domain, CRITERIA["Structure"]),
                        "Relevance": RELEVANCE_DOMAIN_CRITERIA.get(domain, CRITERIA["Relevance"]),
                        "Language": LANGUAGE_DOMAIN_CRITERIA.get(domain, CRITERIA["Language"]),
                        "Criticalness": CRITICALNESS_DOMAIN_CRITERIA.get(domain, CRITERIA["Criticalness"])
                    }
                else:
                    criteria = CRITERIA
            else:
                criteria = CRITERIA

            for criterion_name, criterion in criteria.items():
                if criterion_name in ["Coverage", "Structure", "Relevance", "Language", "Criticalness"]:
                    prompt = fill_single_criterion_prompt(
                        prompt_template=CONTENT_EVALUATION_PROMPT,
                        content=content,
                        topic=topic,
                        criterion=criterion,
                        criteria_name=criterion_name,
                        type="content"
                    )
                    score_dict = self.judge.judge(prompt)
                    if isinstance(score_dict, dict) and criterion_name in score_dict:
                        results[criterion_name] = score_dict[criterion_name]
                    else:
                        results[criterion_name] = 0

        except Exception as e:
            print(f"Error evaluating content: {e}")
            for criterion_name in ["Coverage", "Structure", "Relevance", "Language", "Criticalness"]:
                results[criterion_name] = 0

        return results

    def evaluate_content_faithfulness(self, md_path: str) -> Dict:
        """Evaluate content faithfulness."""
        try:
            content, refs = split_markdown_content_and_refs(md_path)
            sentences = [s.strip() for s in content.split('.') if s.strip()]
            refs = refine_ref_lists(refs)

            total_sentences = len(sentences)
            supported_sentences = 0

            for sentence in sentences:
                prompt = CONTENT_FAITHFULNESS_PROMPT.format(
                    sentence=sentence,
                    references="\n".join(refs)
                )
                response = self.judge.judge(prompt)
                if isinstance(response, dict) and response.get("is_supported", False):
                    supported_sentences += 1

            faithfulness_score = (supported_sentences / total_sentences) * 100 if total_sentences > 0 else 0
            return {"faithfulness_score": round(faithfulness_score, 2)}

        except Exception as e:
            print(f"Error evaluating content faithfulness: {e}")
            return {"faithfulness_score": 0}

class ReferenceEvaluator:
    """Evaluates survey references."""
    def __init__(self, judge: Optional[Judge] = None):
        self.judge = judge or Judge()

    def evaluate_reference_llm(self, md_path: str, criteria_type: str = "general") -> Dict:
        """Evaluate references using LLM-based criteria."""
        results = {}
        try:
            content, refs = split_markdown_content_and_refs(md_path)
            topic = extract_topic_from_path(md_path)
            refs = refine_ref_lists(refs)

            if criteria_type == "domain":
                path_parts = md_path.split(os.sep)
                if len(path_parts) > 1:
                    domain = path_parts[1]
                    criterion = REFERENCE_DOMAIN_CRITERIA.get(domain, CRITERIA["Reference"])
                else:
                    criterion = CRITERIA["Reference"]
            else:
                criterion = CRITERIA["Reference"]

            prompt = fill_single_criterion_prompt(
                prompt_template=REFERENCE_EVALUATION_PROMPT,
                content=content,
                topic=topic,
                criterion=criterion,
                criteria_name="Reference",
                type="reference",
                references="\n".join(refs)
            )
            score_dict = self.judge.judge(prompt)
            if isinstance(score_dict, dict) and "Reference" in score_dict:
                results["Reference"] = score_dict["Reference"]
            else:
                results["Reference"] = 0

        except Exception as e:
            print(f"Error evaluating references: {e}")
            results["Reference"] = 0

        return results

    def evaluate_reference_quality(self, md_path: str) -> Dict:
        """Evaluate reference quality."""
        try:
            content, refs = split_markdown_content_and_refs(md_path)
            topic = extract_topic_from_path(md_path)
            refs = refine_ref_lists(refs)

            prompt = REFERENCE_QUALITY_PROMPT.format(
                content=content,
                topic=topic,
                references="\n".join(refs)
            )
            response = self.judge.judge(prompt)
            
            if isinstance(response, dict):
                quality_score = response.get("quality_score", 0)
                return {"quality_score": quality_score}
            else:
                return {"quality_score": 0}

        except Exception as e:
            print(f"Error evaluating reference quality: {e}")
            return {"quality_score": 0}

class SurveyEvaluator:
    """Main class for comprehensive survey evaluation."""
    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
        self.judge = Judge()
        self.outline_evaluator = OutlineEvaluator(self.judge)
        self.content_evaluator = ContentEvaluator(self.judge)
        self.reference_evaluator = ReferenceEvaluator(self.judge)

    def evaluate(self, survey_path: str) -> Dict:
        """Evaluate a complete survey."""
        results = {}
        
        if self.config.do_outline:
            outline_results = {}
            if self.config.do_llm:
                outline_results.update(
                    self.outline_evaluator.evaluate_outline_llm(
                        survey_path,
                        self.config.criteria_type
                    )
                )
            if self.config.do_coverage:
                outline_results['coverage_score'] = self.outline_evaluator.evaluate_outline_coverage(
                    survey_path
                )
            if self.config.do_structure:
                structure_score, structure_details = self.outline_evaluator.evaluate_outline_structure(
                    survey_path
                )
                outline_results['structure_score'] = structure_score
                outline_results['structure_details'] = structure_details
            results['outline'] = outline_results
        
        if self.config.do_content:
            content_results = {}
            if self.config.do_llm:
                content_results.update(
                    self.content_evaluator.evaluate_content_llm(
                        survey_path,
                        self.config.criteria_type
                    )
                )
            if self.config.do_faithfulness:
                content_results.update(
                    self.content_evaluator.evaluate_content_faithfulness(survey_path)
                )
            results['content'] = content_results
        
        if self.config.do_reference:
            reference_results = {}
            if self.config.do_llm:
                reference_results.update(
                    self.reference_evaluator.evaluate_reference_llm(
                        survey_path,
                        self.config.criteria_type
                    )
                )
            if self.config.do_quality:
                reference_results.update(
                    self.reference_evaluator.evaluate_reference_quality(survey_path)
                )
            results['reference'] = reference_results
        
        return results

def evaluate_survey(
    survey_path: str,
    config: Optional[EvaluationConfig] = None
) -> Dict:
    """Convenience function to evaluate a survey."""
    evaluator = SurveyEvaluator(config)
    return evaluator.evaluate(survey_path) 