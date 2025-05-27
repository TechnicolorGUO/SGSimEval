from concurrent.futures import ThreadPoolExecutor, as_completed
import glob
import json
import math
import os
from queue import Queue
import threading
import time
import dotenv
import pandas as pd
from prompts import CONTENT_EVALUATION_PROMPT, CONTENT_FAITHFULNESS_PROMPT, OUTLINE_EVALUATION_PROMPT, CRITERIA, OUTLINE_STRUCTURE_PROMPT, REFERENCE_EVALUATION_PROMPT, OUTLINE_COVERAGE_PROMPT, REFERENCE_QUALITY_PROMPT, CONTENT_EVALUATION_SIMULTANEOUS_PROMPT, OUTLINE_DOMAIN_CRITERIA, REFERENCE_DOMAIN_CRITERIA, COVERAGE_DOMAIN_CRITERIA, STRUCTURE_DOMAIN_CRITERIA, RELEVANCE_DOMAIN_CRITERIA, LANGUAGE_DOMAIN_CRITERIA, CRITICALNESS_DOMAIN_CRITERIA, OUTLINE_RANKING_PROMPT, CONTENT_RANKING_PROMPT, REFERENCE_RANKING_PROMPT, OUTLINE_COMPARISON_PROMPT, CONTENT_COMPARISON_PROMPT, REFERENCE_COMPARISON_PROMPT
from reference import extract_refs, split_markdown_content_and_refs
from utils import build_outline_tree_from_levels, count_md_features, count_sentences, extract_and_save_outline_from_md, extract_references_from_md, extract_topic_from_path, getClient, generateResponse, pdf2md, refine_outline_if_single_level, robust_json_parse,fill_single_criterion_prompt, read_md, refine_ref_lists
import logging
from atomic_facts import extract_and_deduplicate_facts, extract_facts_only
import csv
import random
from tqdm import tqdm
import numpy as np

class Judge:
    """
    A class to handle LLM-based evaluation using OpenAI's API.
    """
    def __init__(self) -> None:
        """
        Initialize the Judge with OpenAI client and logging configuration.
        """
        dotenv.load_dotenv()
        with open('judge.log', 'w') as log_file:
            log_file.truncate(0)
        self.client = getClient()
        # Configure logging
        logging.basicConfig(filename='judge.log', level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
    
    def judge(self, prompt: str) -> dict | None:
        """
        Evaluate a prompt using the LLM and return the parsed response.
        
        Args:
            prompt (str): The prompt to evaluate
            
        Returns:
            dict | None: Parsed JSON response or None if parsing fails
        """
        response = generateResponse(self.client, prompt)
        logging.info(f"Response received: {response}")
        try:
            result = robust_json_parse(response)
            return result
        except Exception as e:
            logging.error(f"Error parsing JSON: {e}")
            print("Error parsing JSON:", e)
            return None
        
judge = Judge()


# ------------- Outline Evaluation Functions ------------

def evaluate_outline_llm(outline_json_path: str, criteria_type: str = "general") -> dict:
    """
    Evaluate the outline using LLM-based criteria.
    
    Args:
        outline_json_path (str): Path to the outline JSON file
        criteria_type (str, optional): Type of criteria to use ("general" or "domain"). Defaults to "general".
        
    Returns:
        dict: Dictionary containing evaluation scores
    """
    criteria_name = "Outline"
    results = {}
    try:
        # 1. Read outline.json
        with open(outline_json_path, "r", encoding="utf-8") as f:
            outline_list = json.load(f)

        # 2. Format outline as string
        outline_str = "\n".join([json.dumps(item, ensure_ascii=False) for item in outline_list])

        # 3. Use parent directory name as topic
        topic = extract_topic_from_path(outline_json_path)

        # 4. Build prompt and get score
        if criteria_type == "domain":
            # Extract domain from path (assuming format: surveys/domain/topic/...)
            path_parts = outline_json_path.split(os.sep)
            if len(path_parts) > 1:
                domain = path_parts[1]  # Get domain from path
                if domain in OUTLINE_DOMAIN_CRITERIA:
                    criterion = OUTLINE_DOMAIN_CRITERIA[domain]
                else:
                    criterion = CRITERIA[criteria_name]
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
        score_dict = judge.judge(prompt)
        if not (isinstance(score_dict, dict) and criteria_name in score_dict):
            results[criteria_name] = 0
        else:
            results.update(score_dict)
    except Exception as e:
        results[criteria_name] = 0

    # Add domain suffix to key if using domain-specific criteria
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
    outline_json_path: str,
    standard_count: int = 10,      # Reference section count for coverage calculation
    enable_penalty: bool = True     # Whether to enable length penalty
) -> float:
    """
    Evaluate outline coverage score using the formula:
    R = (K/N) * e^(-(M-N/N)^2)
    where:
    - K is the number of matched sections
    - N is the number of standard section titles
    - M is the number of sections in the evaluated outline
    
    Args:
        outline_json_path (str): Path to the outline JSON file
        standard_count (int, optional): Number of standard section titles (N). Defaults to 10.
        enable_penalty (bool, optional): Whether to enable length penalty. Defaults to True.
        
    Returns:
        float: Coverage score between 0 and 100
    """
    try:
        with open(outline_json_path, "r", encoding="utf-8") as f:
            outline_list = json.load(f)

        # Count level 1 sections based on the two cases
        level_1_count = sum(1 for item in outline_list if item[0] == 1)
        level_2_count = sum(1 for item in outline_list if item[0] == 2)
        
        # Case 1: If there's only one level 1 item (the title), then level 2 items are the main sections
        if level_1_count == 1:
            total_section_count = level_2_count
        # Case 2: If there are multiple level 1 items, count them and subtract 1 (the title)
        else:
            total_section_count = level_1_count - 1

        print(f"Total section count: {total_section_count}")

        outline_str = "\n".join([json.dumps(item, ensure_ascii=False) for item in outline_list])
        topic = extract_topic_from_path(outline_json_path)
        prompt = OUTLINE_COVERAGE_PROMPT.format(
            outline=outline_str,
            topic=topic,
        )
        response = judge.judge(prompt)
        matched_count = response.get("matched_count", 0)

        K = matched_count         # Number of matched sections
        N = standard_count        # Number of standard section titles
        M = total_section_count   # Number of sections in evaluated outline

        # Calculate coverage using the new formula
        if N > 0:
            # Calculate the exponential penalty term
            length_ratio = (M - N) / N
            penalty_term = math.exp(-(length_ratio ** 2))
            
            # Calculate the coverage score
            coverage = (K / N) * penalty_term
            
            # Convert to percentage and round to 4 decimal places
            return round(coverage * 100, 4)
        else:
            return 0.0

    except Exception as e:
        print(f"Error evaluating outline coverage: {e}")
        return 0.0

def evaluate_outline_structure(outline_json_path: str) -> tuple[float, list[dict]]:
    """
    Evaluate the hierarchical structure of the outline.
    Uses depth-based weighted scoring where nodes closer to root have higher weights.
    
    Args:
        outline_json_path (str): Path to the outline JSON file
        
    Returns:
        tuple[float, list[dict]]: (global structure score, list of node scores)
    """
    topic = extract_topic_from_path(outline_json_path)
    with open(outline_json_path, "r", encoding="utf-8") as f:
        outline_list = json.load(f)
    node_objs, _ = build_outline_tree_from_levels(outline_list)
    non_leaf_nodes = [node for node in node_objs if node["children"]]
    node_scores = []
    
    # Calculate max depth for weight normalization
    max_depth = max(node["level"] for node in node_objs) if node_objs else 1
    
    for parent in non_leaf_nodes:
        children_list = "\n".join([
            f'  - Index: {child["index"]}, Title: {child["title"]}'
            for child in parent["children"]
        ])
        prompt = OUTLINE_STRUCTURE_PROMPT.format(
            topic = topic,
            parent_index=parent["index"],
            parent_title=parent["title"],
            children_list=children_list
        )
        response = judge.judge(prompt)
        result = response.get("children", [])
        yes_count = sum(1 for child in result if str(child.get("is_included", "")).lower() == "yes")
        total = len(result)
        node_score = yes_count / total if total > 0 else 1.0  # Full score if no children
        
        # Calculate weight based on depth (inverse of depth)
        # Nodes closer to root (lower depth) get higher weights
        depth = parent["level"]
        weight = (max_depth - depth + 1) / max_depth
        
        node_scores.append({
            "parent_index": parent["index"],
            "parent_title": parent["title"],
            "score": node_score,
            "weight": weight,
            "depth": depth
        })

    # Calculate weighted average
    if node_scores:
        weighted_sum = sum(x["score"] * x["weight"] for x in node_scores)
        total_weight = sum(x["weight"] for x in node_scores)
        global_score = round(weighted_sum / total_weight * 100, 4)
    else:
        global_score = 100.0  # Full score if no nodes to evaluate
        
    return global_score, node_scores

def evaluate_outline_number(outline_json_path: str) -> dict:
    """
    Evaluate the number of sections in the outline.
    
    Args:
        outline_json_path (str): Path to the outline JSON file
        
    Returns:
        dict: Dictionary containing the section count
    """
    results = {}
    try:
        with open(outline_json_path, "r", encoding="utf-8") as f:
            outline_list = json.load(f)
        total_section_count = len(outline_list)
        results["Outline_no"] = total_section_count
    except Exception as e:
        print(f"Error evaluating outline number: {e}")
        results["Outline_no"] = 0
    return results

def evaluate_outline_density(md_path: str) -> dict:
    """
    Calculate the density of sections in the outline.
    
    Args:
        md_path (str): Path to the markdown file
        
    Returns:
        dict: Dictionary containing outline density score
    """
    results = {}

    json_path = os.path.join(os.path.dirname(md_path), "outline.json")

    with open(json_path, "r", encoding="utf-8") as f:
        outline = json.load(f)
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    main_content, _ = split_markdown_content_and_refs(content)
    sentence_count = count_sentences(main_content)
    outline_count = len(outline)

    if sentence_count > 0:
        results["Outline_density"] = round(outline_count / sentence_count, 4) * 100
    else:
        results["Outline_density"] = 0
    return results

def evaluate_outline(
    md_path: str,
    do_llm: bool = False,
    do_coverage: bool = True,
    do_structure: bool = False,
    do_number: bool = False,
    do_density: bool = False,
    criteria_type: str = "general"
) -> dict:
    """
    Evaluate the outline of a markdown file.
    
    Args:
        md_path (str): Path to the markdown file
        do_llm (bool, optional): Whether to perform LLM evaluation. Defaults to True.
        do_coverage (bool, optional): Whether to evaluate coverage. Defaults to True.
        do_structure (bool, optional): Whether to evaluate structure. Defaults to True.
        criteria_type (str, optional): Type of criteria to use ("general" or "domain"). Defaults to "general".
        
    Returns:
        dict: Dictionary containing evaluation results
    """
    results = {}
    outline_json_path = os.path.join(os.path.dirname(md_path), "outline.json")
    
    # 1. Extract outline from md (only if outline.json doesn't exist)
    if not os.path.exists(outline_json_path):
        try:
            extract_and_save_outline_from_md(md_path)
            outline_raw_json_path = os.path.join(os.path.dirname(md_path), "outline_raw.json")
        except Exception as e:
            print("Error extracting outline:", e)
            return results
    else:
        print(f"Found {outline_json_path}, skip extraction.")
        outline_raw_json_path = os.path.join(os.path.dirname(md_path), "outline.json")

    # 2. LLM evaluation
    if do_llm:
        try:
            outline_results = evaluate_outline_llm(outline_raw_json_path, criteria_type)
            results.update(outline_results)
            if criteria_type == "domain":
                return results
        except Exception as e:
            print("Error in evaluating outline llm:", e)
            if criteria_type == "domain":
                results["Outline_domain"] = 0
                return results
            else:
                results["Outline"] = 0
                return results
    else:
        print("Skip evaluate_outline_llm.")

    # 3. Coverage evaluation
    if do_coverage:
        try:
            coverage_results = evaluate_outline_coverage(outline_raw_json_path)
            results["Outline_coverage"] = coverage_results
        except Exception as e:
            print("Error in evaluating outline coverage:", e)
            results["Outline_coverage"] = 0
    else:
        print("Skip evaluate_outline_coverage.")

    # 4. Structure evaluation
    if do_structure:
        outline_json_path = os.path.join(os.path.dirname(md_path), "outline.json")
        refine_outline_if_single_level(outline_raw_json_path, outline_json_path)
        try:
            global_score, node_scores = evaluate_outline_structure(outline_json_path)
            results["Outline_structure"] = global_score
        except Exception as e:
            print("Error in evaluating outline structure:", e)
            results["Outline_structure"] = 0
    else:
        print("Skip evaluate_outline_structure.")

    # 5. Number evaluation
    if do_number:
        try:
            number_results = evaluate_outline_number(outline_json_path)
            results.update(number_results)
        except Exception as e:
            print("Error in evaluating outline number:", e)
            results["Outline_no"] = 0
    else:
        print("Skip evaluate_outline_number.")

    # 6. Density evaluation
    if do_density:
        try:
            density_results = evaluate_outline_density(md_path)
            results.update(density_results)
        except Exception as e:
            print("Error in evaluating outline density:", e)
            results["Outline_density"] = 0
    else:
        print("Skip evaluate_outline_density.")

    return results

# ------------- Content Evaluation Functions ------------

def evaluate_content_llm(md_path: str, criteria_type: str = "general") -> dict:
    """
    Evaluate content using LLM-based criteria.
    
    Args:
        md_path (str): Path to the markdown file
        criteria_type (str, optional): Type of criteria to use ("general" or "domain"). Defaults to "general".
        
    Returns:
        dict: Dictionary containing evaluation scores for each criterion
    """
    content_criteria = ["Coverage", "Structure", "Relevance", "Language", "Criticalness"]
    results = {}

    try:
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()
        content_str, _ = split_markdown_content_and_refs(content)
    except Exception as e:
        for criteria_name in content_criteria:
            if criteria_type == "domain":
                results[f"{criteria_name}_domain"] = 0
            else:
                results[criteria_name] = 0
        print("All content criteria scores:", results)
        return results

    topic = extract_topic_from_path(md_path)

    if criteria_type == "domain":
        # Extract domain from path (assuming format: surveys/domain/topic/...)
        path_parts = md_path.split(os.sep)
        if len(path_parts) > 1:
            domain = path_parts[1]  # Get domain from path
            # Use domain-specific criteria for each criterion
            domain_criteria = {
                "Coverage": COVERAGE_DOMAIN_CRITERIA[domain],
                "Structure": STRUCTURE_DOMAIN_CRITERIA[domain],
                "Relevance": RELEVANCE_DOMAIN_CRITERIA[domain],
                "Language": LANGUAGE_DOMAIN_CRITERIA[domain],
                "Criticalness": CRITICALNESS_DOMAIN_CRITERIA[domain]
            }
        else:
            # Fallback to general criteria if domain not found
            domain_criteria = {name: CRITERIA[name] for name in content_criteria}
    else:
        domain_criteria = {name: CRITERIA[name] for name in content_criteria}

    for criteria_name in content_criteria:
        criterion = domain_criteria[criteria_name]
        prompt = fill_single_criterion_prompt(
            prompt_template=CONTENT_EVALUATION_PROMPT,
            content=content_str,
            topic=topic,
            criterion=criterion,
            criteria_name=criteria_name,
            type="content"
        )
        try:
            score_dict = judge.judge(prompt)
            if not (isinstance(score_dict, dict) and criteria_name in score_dict):
                if criteria_type == "domain":
                    results[f"{criteria_name}_domain"] = 0
                else:
                    results[criteria_name] = 0
            else:
                if criteria_type == "domain":
                    results[f"{criteria_name}_domain"] = score_dict[criteria_name]
                else:
                    results.update(score_dict)
        except Exception as e:
            if criteria_type == "domain":
                results[f"{criteria_name}_domain"] = 0
            else:
                results[criteria_name] = 0

    return results

def evaluate_content_llm_simultaneous(md_path: str, criteria_type: str = "general") -> dict:
    """
    Evaluate content using LLM-based criteria simultaneously for all criteria.
    
    Args:
        md_path (str): Path to the markdown file
        criteria_type (str, optional): Type of criteria to use ("general" or "domain"). Defaults to "general".
        
    Returns:
        dict: Dictionary containing evaluation scores for all criteria
    """
    content_criteria = ["Coverage", "Structure", "Relevance", "Language", "Criticalness"]
    results = {}

    try:
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()
        content_str, _ = split_markdown_content_and_refs(content)
    except Exception as e:
        for criteria_name in content_criteria:
            if criteria_type == "domain":
                results[f"{criteria_name}_domain"] = 0
            else:
                results[criteria_name] = 0
        print("All content criteria scores:", results)
        return results

    topic = extract_topic_from_path(md_path)
    
    # Get domain-specific criteria if needed
    if criteria_type == "domain":
        # Extract domain from path (assuming format: surveys/domain/topic/...)
        path_parts = md_path.split(os.sep)
        if len(path_parts) > 1:
            domain = path_parts[1]  # Get domain from path
            # Use domain-specific criteria for each criterion
            domain_criteria = {
                "Coverage": COVERAGE_DOMAIN_CRITERIA[domain],
                "Structure": STRUCTURE_DOMAIN_CRITERIA[domain],
                "Relevance": RELEVANCE_DOMAIN_CRITERIA[domain],
                "Language": LANGUAGE_DOMAIN_CRITERIA[domain],
                "Criticalness": CRITICALNESS_DOMAIN_CRITERIA[domain]
            }
        else:
            # Fallback to general criteria if domain not found
            domain_criteria = {name: CRITERIA[name] for name in content_criteria}
    else:
        domain_criteria = {name: CRITERIA[name] for name in content_criteria}
    
    # Prepare prompt parameters for all criteria
    prompt_params = {
        "topic": topic,
        "content": content_str
    }
    
    # Add all criteria descriptions and scores to prompt parameters
    for criteria_name in content_criteria:
        criterion = domain_criteria[criteria_name]
        prompt_params[f"{criteria_name.lower()}_description"] = criterion["description"]
        for i in range(1, 6):
            prompt_params[f"{criteria_name.lower()}_score_{i}"] = criterion[f"score {i}"]

    try:
        # Generate the prompt with all criteria
        prompt = CONTENT_EVALUATION_SIMULTANEOUS_PROMPT.format(**prompt_params)
        
        # Get scores for all criteria at once
        score_dict = judge.judge(prompt)
        
        # Validate and update results
        if isinstance(score_dict, dict):
            for criteria_name in content_criteria:
                if criteria_name in score_dict:
                    if criteria_type == "domain":
                        results[f"{criteria_name}_domain"] = score_dict[criteria_name]
                    else:
                        results[criteria_name] = score_dict[criteria_name]
                else:
                    if criteria_type == "domain":
                        results[f"{criteria_name}_domain"] = 0
                    else:
                        results[criteria_name] = 0
        else:
            for criteria_name in content_criteria:
                if criteria_type == "domain":
                    results[f"{criteria_name}_domain"] = 0
                else:
                    results[criteria_name] = 0
    except Exception as e:
        print(f"Error in simultaneous evaluation: {e}")
        for criteria_name in content_criteria:
            if criteria_type == "domain":
                results[f"{criteria_name}_domain"] = 0
            else:
                results[criteria_name] = 0

    return results

def evaluate_content_informativeness(md_path: str) -> dict:
    """
    Evaluate the informativeness of the content by analyzing various features.
    
    Args:
        md_path (str): Path to the markdown file
        
    Returns:
        dict: Dictionary containing density scores for various content features
    """
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()
    content_str, _ = split_markdown_content_and_refs(content)
    counts = count_md_features(content_str)
    sentences = counts.get('sentences', 1)
    results = {}    

    images_density = counts.get('images', 0) / sentences
    equations_density = counts.get('equations', 0) / sentences
    tables_density = counts.get('tables', 0) / sentences
    total_density = (counts.get('images', 0) + counts.get('equations', 0) + counts.get('tables', 0)) / sentences

    results["Images_density"] = round(images_density * 100, 4)
    results["Equations_density"] = round(equations_density * 100, 4)
    results["Tables_density"] = round(tables_density * 100, 4)
    results["Total_density"] = round(total_density * 100, 4)

    # Citation density
    total_citations = 0
    csv_path = os.path.join(os.path.dirname(md_path), os.path.basename(md_path).replace(".md", ".csv"))  
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = pd.read_csv(f)
        for index, row in reader.iterrows():
            sentence = row["sentence"]
            references = row["references"].split(";")
            citations = len(references)
            total_citations += citations
    if total_citations > 0:
        results["Citations_density"] = round(total_citations * 100 / sentences, 4)
    else:
        results["Citations_density"] = 0

    # Claim density
    topic = extract_topic_from_path(md_path)
    results_claims = extract_facts_only(content_str, topic)
    results["Claim_density"] = results_claims.get("claim_density", 0) * 100
    # results["Claim_density_after_deduplication"] = results_claims.get("claim_density_after_dedup", 0) * 100
    return results

def evaluate_content_faithfulness(md_path: str) -> dict:
    """
    Evaluate the faithfulness of content by checking reference support.
    
    Args:
        md_path (str): Path to the markdown file
        
    Returns:
        dict: Dictionary containing reference quality scores
    """
    results = {}
    csv_path = os.path.join(os.path.dirname(md_path), os.path.basename(md_path).replace(".md", ".csv"))  

    # csv columns: [sentence,references]
    refs_mapping = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = pd.read_csv(f)
        for index, row in reader.iterrows():
            sentence = row["sentence"]
            references = row["references"].split(";")
            refs_mapping[sentence] = references

    # Count the number of references that are relevant to the topic
    total_count = 0
    supported_count = 0
    topic = extract_topic_from_path(md_path)
    for sentence, references in refs_mapping.items():
        # Call LLM to evaluate the relevance of each reference
        prompt = CONTENT_FAITHFULNESS_PROMPT.format(
            sentence=sentence,
            references="\n".join(references),
            topic=topic
        )
        try:
            response = judge.judge(prompt)
            total = response.get("total", 0)
            supported = response.get("supported", 0)
            total_count += int(total)
            supported_count += int(supported)
        except Exception as e:
            total_count += 0
            supported_count += 0
            print("Error in evaluating reference quality:", e)
            continue

    if total_count > 0:
        results["Faithfulness"] = round(supported_count / total_count, 4) * 100
    else:
        results["Faithfulness"] = 0
    print("Faithfulness score:", results)
    return results

def evaluate_content_faithfulness_parallel(md_path: str, max_workers: int = 4) -> dict:
    """
    Evaluate content faithfulness using parallel processing.
    
    Args:
        md_path (str): Path to the markdown file
        max_workers (int, optional): Maximum number of worker threads. Defaults to 4.
        
    Returns:
        dict: Dictionary containing reference quality scores
    """
    results = {}
    csv_path = os.path.join(os.path.dirname(md_path), os.path.basename(md_path).replace(".md", ".csv"))  

    # csv columns: [sentence,references]
    refs_mapping = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = pd.read_csv(f)
        for index, row in reader.iterrows():
            sentence = row["sentence"]
            references = row["references"].split(";")
            refs_mapping[sentence] = references

    total_count = 0
    supported_count = 0
    topic = extract_topic_from_path(md_path)

    # Define single evaluation task
    def evaluate_sentence(sentence: str, references: list[str]) -> tuple[int, int]:
        """
        Evaluate a single sentence and its references.
        
        Args:
            sentence (str): The sentence to evaluate
            references (list[str]): List of references to check
            
        Returns:
            tuple[int, int]: (total references, supported references)
        """
        prompt = CONTENT_FAITHFULNESS_PROMPT.format(
            sentence=sentence,
            references="\n".join(references),
            topic=topic
        )
        try:
            response = judge.judge(prompt)
            total = response.get("total", 0)
            supported = response.get("supported", 0)
            return int(total), int(supported)
        except Exception as e:
            print("Error in evaluating reference quality:", e)
            return 0, 0

    # Process all sentences in parallel
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for sentence, references in refs_mapping.items():
            futures.append(executor.submit(evaluate_sentence, sentence, references))

        for future in as_completed(futures):
            total, supported = future.result()
            total_count += total
            supported_count += supported

    if total_count > 0:
        results["Reference_quality"] = round(supported_count / total_count * 100, 2)
    else:
        results["Reference_quality"] = 0.0

    print("Reference quality score:", results)
    return results

def evaluate_content_sentence_number(md_path: str) -> dict:
    """
    Count the number of sentences in the content.
    
    Args:
        md_path (str): Path to the markdown file
        
    Returns:
        dict: Dictionary containing the sentence count
    """
    results = {}
    try:
        with open(md_path, "r", encoding="utf-8") as f:
            content_str = f.read()

        main_content, _ = split_markdown_content_and_refs(content_str)
        sentence_count = count_sentences(main_content)
        results["Sentence_no"] = sentence_count
    except Exception as e:
        print("Error in evaluating sentence number:", e)
        results["Sentence_no"] = 0
    return results

def evaluate_content(
    md_path: str,
    do_llm: bool = True,
    do_info: bool = True,
    do_faithfulness: bool = True,
    criteria_type: str = "general"
) -> dict:
    """
    Evaluate content using multiple criteria.
    
    Args:
        md_path (str): Path to the markdown file
        do_llm (bool, optional): Whether to perform LLM evaluation. Defaults to True.
        do_info (bool, optional): Whether to evaluate informativeness. Defaults to True.
        do_faithfulness (bool, optional): Whether to evaluate faithfulness. Defaults to True.
        criteria_type (str, optional): Type of criteria to use ("general" or "domain"). Defaults to "general".
        
    Returns:
        dict: Dictionary containing all evaluation results
    """
    results = {}
    content_criteria = [
        "Coverage", "Structure", "Relevance", "Language", "Criticalness"
    ]
    info_criteria = [
        "Images_density", "Equations_density", "Tables_density", 
        "Total_density", "Claim_density", 
    ]
    
    with open(md_path, "r", encoding="utf-8") as f:
        content_str = f.read()
    content, _ = split_markdown_content_and_refs(content_str)
    content_path = os.path.join(os.path.dirname(md_path), "content.json")
    with open(content_path, "w", encoding="utf-8") as f:
        json.dump([content], f, ensure_ascii=False, indent=4)
    # 1. LLM evaluation
    if do_llm:
        try:
            # Use the new simultaneous evaluation function
            results.update(evaluate_content_llm_simultaneous(md_path, criteria_type))
            if criteria_type == "domain":
                return results
        except Exception as e:
            print("Error in evaluating content:", e)
            for criteria_name in content_criteria:
                if criteria_type == "domain":
                    results[f"{criteria_name}_domain"] = 0
                else:
                    results[criteria_name] = 0
            return results
    else:
        print("Skip evaluate_content_llm.")

    # 2. Informativeness evaluation
    if do_info:
        try:
            results.update(evaluate_content_informativeness(md_path))
        except Exception as e:
            print("Error in evaluating content informativeness:", e)
            for criteria_name in info_criteria:
                results[criteria_name] = 0
    else:
        print("Skip evaluate_content_informativeness.")

    # 3. Faithfulness evaluation
    if do_faithfulness:
        try:
            results.update(evaluate_content_faithfulness(md_path))
        except Exception as e:
            print("Error in evaluating content faithfulness:", e)
            results['Faithfulness'] = 0
    else:
        print("Skip evaluate_content_faithfulness.")

    # 4. Sentence count evaluation
    try:
        results.update(evaluate_content_sentence_number(md_path))
    except Exception as e:
        print("Error in evaluating content sentence number:", e)
        results["Sentence_no"] = 0

    return results

# ------------- Reference Evaluation Functions -------------

def evaluate_reference_llm(md_path: str, criteria_type: str = "general") -> dict:
    """
    Evaluate references using LLM-based criteria.
    
    Args:
        md_path (str): Path to the markdown file
        criteria_type (str, optional): Type of criteria to use ("general" or "domain"). Defaults to "general".
        
    Returns:
        dict: Dictionary containing reference evaluation scores
    """
    results = {}
    criteria_name = "Reference"
    try:
        references = extract_references_from_md(md_path)
        if not references:
            results[criteria_name] = 0
            print("No references found.")
            return results

        topic = extract_topic_from_path(md_path)
        
        if criteria_type == "domain":
            # Extract domain from path (assuming format: surveys/domain/topic/...)
            path_parts = md_path.split(os.sep)
            if len(path_parts) > 1:
                domain = path_parts[1]  # Get domain from path
                if domain in REFERENCE_DOMAIN_CRITERIA:
                    criterion = REFERENCE_DOMAIN_CRITERIA[domain]
                else:
                    criterion = CRITERIA[criteria_name]
            else:
                criterion = CRITERIA[criteria_name]
        else:
            criterion = CRITERIA[criteria_name]

        references_str = "\n".join(references)
        prompt = fill_single_criterion_prompt(
            prompt_template=REFERENCE_EVALUATION_PROMPT,
            content=references_str,
            topic=topic,
            criterion=criterion,
            criteria_name=criteria_name,
            type="reference"
        )
        try:
            score_dict = judge.judge(prompt)
            if not (isinstance(score_dict, dict) and criteria_name in score_dict):
                results[criteria_name] = 0
            else:
                results.update(score_dict)
        except Exception:
            print("Error in scoring references.")
            results[criteria_name] = 0
    except Exception:
        print("Error in extracting references.")
        results[criteria_name] = 0

    # Add domain suffix to key if using domain-specific criteria
    if criteria_type == "domain":
        new_results = {}
        for key, value in results.items():
            if key == "Reference":
                new_results["Reference_domain"] = value
            else:
                new_results[key] = value
        return new_results
    return results

def evaluate_reference_density(md_path: str) -> dict:
    """
    Calculate the density of references in the content.
    
    Args:
        md_path (str): Path to the markdown file
        
    Returns:
        dict: Dictionary containing reference density score
    """
    results = {}

    json_path = os.path.join(os.path.dirname(md_path), "references.json")

    with open(json_path, "r", encoding="utf-8") as f:
        references = json.load(f)
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    main_content, _ = split_markdown_content_and_refs(content)
    sentence_count = count_sentences(main_content)
    references_count = len(references)

    if sentence_count > 0:
        results["Reference_density"] = round(references_count / sentence_count, 4) * 100
    else:
        results["Reference_density"] = 0
    return results

def evaluate_reference_quality(md_path: str) -> dict:
    """
    Evaluate the quality of references using LLM.
    
    Args:
        md_path (str): Path to the markdown file
        
    Returns:
        dict: Dictionary containing reference quality score
    """
    results = {}
    batch_size = 20
    ref_path = os.path.join(os.path.dirname(md_path), "references.json")
    topic = extract_topic_from_path(md_path)
    total_count = 0
    supported_count = 0

    with open(ref_path, "r", encoding="utf-8") as f:
        refs = json.load(f)
    
    for i in range(0, len(refs), batch_size):
        batch_refs = refs[i:i + batch_size]
        prompt = REFERENCE_QUALITY_PROMPT.format(
            references="\n".join(batch_refs),
            topic=topic
        )
        try:
            response = judge.judge(prompt)
            total = response.get("total", 0)
            supported = response.get("supported", 0)
            total_count += int(total)
            supported_count += int(supported)
        except Exception as e:
            print("Error in evaluating reference quality:", e)
            continue

    if total_count > 0:
        results["Reference_quality"] = round(supported_count / total_count * 100, 2)
    else:
        results["Reference_quality"] = 0.0

    return results

# def evaluate_reference_coverage(md_path: str) -> dict:

def evaluate_reference_number(md_path: str) -> dict:
    """
    Count the number of references in the content.
    
    Args:
        md_path (str): Path to the markdown file
        
    Returns:
        dict: Dictionary containing the reference count
    """
    results = {}
    json_path = os.path.join(os.path.dirname(md_path), "references.json")

    with open(json_path, "r", encoding="utf-8") as f:
        references = json.load(f)
    
    total_count = len(references)
    results["Reference_no"] = total_count
    return results

def evaluate_reference(
    md_path: str,
    do_llm: bool = True,
    do_number: bool = True,
    do_density: bool = True,
    do_quality: bool = True,
    criteria_type: str = "general"
) -> dict:
    """
    Evaluate references using multiple criteria.
    
    Args:
        md_path (str): Path to the markdown file
        do_llm (bool, optional): Whether to perform LLM evaluation. Defaults to True.
        do_density (bool, optional): Whether to evaluate density. Defaults to True.
        do_quality (bool, optional): Whether to evaluate quality. Defaults to True.
        criteria_type (str, optional): Type of criteria to use ("general" or "domain"). Defaults to "general".
        
    Returns:
        dict: Dictionary containing all reference evaluation results
    """
    results = {}

    # 0. Extract references if not exists
    reference_path = os.path.join(os.path.dirname(md_path), "references.json")
    if not os.path.exists(reference_path):
        extract_refs(input_file=md_path, output_folder=os.path.dirname(md_path))

    # 1. LLM evaluation
    if do_llm:
        try:
            results.update(evaluate_reference_llm(md_path, criteria_type))
            if criteria_type == "domain":
                return results
        except Exception as e:
            print("Error in evaluating reference:", e)
            if criteria_type == "domain":
                results["Reference_domain"] = 0
                return results
            else:
                results["Reference"] = 0
                return results
    else:
        print("Skip evaluate_reference_llm.")

    # 2. Density evaluation
    if do_density:
        try:
            results.update(evaluate_reference_density(md_path))
        except Exception as e:
            print("Error in evaluating reference density:", e)
            results["Reference_density"] = 0
    else:
        print("Skip evaluate_reference_density.")

    # 3. Quality evaluation
    if do_quality:
        try:
            results.update(evaluate_reference_quality(md_path))
        except Exception as e:
            print("Error in evaluating reference quality:", e)
            results["Reference_quality"] = 0
    else:
        print("Skip evaluate_reference_quality.")

    # 4. Number evaluation
    if do_number:
        try:
            results.update(evaluate_reference_number(md_path))
        except Exception as e:
            print("Error in evaluating reference number:", e)
            results["Reference_no"] = 0
    else:
        print("Skip evaluate_reference_number.")
    return results

# ------------- Relative Comparison Functions -------------
def compare_with_pdfs(topic_dir: str, system: str, metrics: list[str], modelname: str) -> dict:
    """
    Compare a system's output with PDFs for a given topic.
    
    Args:
        topic_dir (str): Path to the topic directory
        system (str): Name of the system to compare with PDFs
        metrics (list[str]): List of metrics to evaluate, can include 'outline', 'content', 'reference'
        modelname (str): Name of the evaluation model
        
    Returns:
        dict: Dictionary containing comparison results
    """
    results = {}
    
    # Get pdfs results
    pdfs_dir = os.path.join(topic_dir, "pdfs")
    if not os.path.exists(pdfs_dir):
        print(f"Error: pdfs directory not found at {pdfs_dir}")
        return results
    
    # Get system results
    system_dir = os.path.join(topic_dir, system)
    if not os.path.exists(system_dir):
        print(f"Error: system directory not found at {system_dir}")
        return results
    
    # Extract topic name from path
    topic = extract_topic_from_path(topic_dir)
    
    # Process each metric
    for metric in metrics:
        if metric.lower() == "outline":
            # Compare outlines
            pdfs_outline_path = os.path.join(pdfs_dir, "outline.json")
            system_outline_path = os.path.join(system_dir, "outline.json")
            
            if os.path.exists(pdfs_outline_path) and os.path.exists(system_outline_path):
                try:
                    with open(pdfs_outline_path, "r", encoding="utf-8") as f:
                        pdfs_outline = json.load(f)
                    with open(system_outline_path, "r", encoding="utf-8") as f:
                        system_outline = json.load(f)
                    
                    # Format outlines as strings
                    pdfs_outline_str = "\n".join([json.dumps(item, ensure_ascii=False) for item in pdfs_outline])
                    system_outline_str = "\n".join([json.dumps(item, ensure_ascii=False) for item in system_outline])
                    
                    # Randomly assign system and PDF to outline_1 and outline_2
                    import random
                    is_system_first = random.choice([True, False])
                    outline_1 = system_outline_str if is_system_first else pdfs_outline_str
                    outline_2 = pdfs_outline_str if is_system_first else system_outline_str
                    
                    # Generate prompt and get comparison
                    prompt = OUTLINE_COMPARISON_PROMPT.format(
                        topic=topic,
                        outline_1=outline_1,
                        outline_2=outline_2
                    )
                    comparison = judge.judge(prompt)
                    if isinstance(comparison, dict) and "is_better" in comparison:
                        # If system was outline_1, use the result directly; otherwise, invert it
                        is_better = comparison["is_better"] if is_system_first else not comparison["is_better"]
                        results["outline_is_better"] = is_better
                        results["outline_reason"] = comparison.get("reason", "")
                except Exception as e:
                    print(f"Error comparing outlines: {e}")
        
        elif metric.lower() == "content":
            # Compare contents
            pdfs_md = [f for f in os.listdir(pdfs_dir) if f.lower().endswith(".md")]
            system_md = [f for f in os.listdir(system_dir) if f.lower().endswith(".md")]
            
            if pdfs_md and system_md:
                try:
                    with open(os.path.join(pdfs_dir, pdfs_md[0]), "r", encoding="utf-8") as f:
                        pdfs_content = f.read()
                    with open(os.path.join(system_dir, system_md[0]), "r", encoding="utf-8") as f:
                        system_content = f.read()
                    
                    # Split content and references
                    pdfs_content_str, _ = split_markdown_content_and_refs(pdfs_content)
                    system_content_str, _ = split_markdown_content_and_refs(system_content)
                    
                    # Randomly assign system and PDF to content_1 and content_2
                    is_system_first = random.choice([True, False])
                    content_1 = system_content_str if is_system_first else pdfs_content_str
                    content_2 = pdfs_content_str if is_system_first else system_content_str
                    
                    # Generate prompt and get comparison
                    prompt = CONTENT_COMPARISON_PROMPT.format(
                        topic=topic,
                        content_1=content_1,
                        content_2=content_2
                    )
                    comparison = judge.judge(prompt)
                    if isinstance(comparison, dict) and "is_better" in comparison:
                        # If system was content_1, use the result directly; otherwise, invert it
                        is_better = comparison["is_better"] if is_system_first else not comparison["is_better"]
                        results["content_is_better"] = is_better
                        results["content_reason"] = comparison.get("reason", "")
                except Exception as e:
                    print(f"Error comparing contents: {e}")
        
        elif metric.lower() == "reference":
            # Compare references
            pdfs_ref_path = os.path.join(pdfs_dir, "references.json")
            system_ref_path = os.path.join(system_dir, "references.json")
            
            if os.path.exists(pdfs_ref_path) and os.path.exists(system_ref_path):
                try:
                    with open(pdfs_ref_path, "r", encoding="utf-8") as f:
                        pdfs_refs = json.load(f)
                    with open(system_ref_path, "r", encoding="utf-8") as f:
                        system_refs = json.load(f)
                    
                    # Format references as strings
                    pdfs_refs_str = "\n".join(pdfs_refs)
                    system_refs_str = "\n".join(system_refs)
                    
                    # Randomly assign system and PDF to references_1 and references_2
                    is_system_first = random.choice([True, False])
                    references_1 = system_refs_str if is_system_first else pdfs_refs_str
                    references_2 = pdfs_refs_str if is_system_first else system_refs_str
                    
                    # Generate prompt and get comparison
                    prompt = REFERENCE_COMPARISON_PROMPT.format(
                        topic=topic,
                        references_1=references_1,
                        references_2=references_2
                    )
                    comparison = judge.judge(prompt)
                    if isinstance(comparison, dict) and "is_better" in comparison:
                        # If system was references_1, use the result directly; otherwise, invert it
                        is_better = comparison["is_better"] if is_system_first else not comparison["is_better"]
                        results["reference_is_better"] = is_better
                        results["reference_reason"] = comparison.get("reason", "")
                except Exception as e:
                    print(f"Error comparing references: {e}")
    
    # Save results to system directory
    results_path = os.path.join(system_dir, f"compare_{modelname}.json")
    try:
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error saving comparison results: {e}")
    
    return results

def aggregate_comparison_results(root_dir: str, systems: list[str], metrics: list[str], modelname: str) -> None:
    """
    Aggregate comparison results across all categories and topics.
    
    Args:
        root_dir (str): Root directory containing categories
        systems (list[str]): List of system names to compare
        metrics (list[str]): List of metrics to evaluate
        modelname (str): Name of the evaluation model
    """
    from tqdm import tqdm
    
    # Initialize results storage
    category_results = {}  # Results per category
    global_results = {     # Global results across all categories
        system: {
            metric: {"wins": 0, "losses": 0, "total": 0}
            for metric in metrics
        }
        for system in systems
    }
    
    # Get all categories
    categories = [cat for cat in os.listdir(root_dir) 
                 if os.path.isdir(os.path.join(root_dir, cat)) and cat != "pdfs"]
    
    # Process each category with progress bar
    for category in tqdm(categories, desc="Processing categories"):
        category_dir = os.path.join(root_dir, category)
        
        # Initialize category results
        category_results[category] = {
            system: {
                metric: {"wins": 0, "losses": 0, "total": 0}
                for metric in metrics
            }
            for system in systems
        }
        
        # Get all topics in this category
        topics = [topic for topic in os.listdir(category_dir) 
                 if os.path.isdir(os.path.join(category_dir, topic)) and topic != "pdfs"]
        
        # Process each topic in the category with nested progress bar
        for topic in tqdm(topics, desc=f"Processing {category} topics", leave=False):
            topic_dir = os.path.join(category_dir, topic)
            
            # Process each system
            for system in systems:
                system_dir = os.path.join(topic_dir, system)
                if not os.path.exists(system_dir):
                    continue
                
                # Compare with PDFs
                compare_path = os.path.join(system_dir, f"compare_{modelname}.json")
                if not os.path.exists(compare_path):
                    results = compare_with_pdfs(topic_dir, system, metrics, modelname)
                else:
                    with open(compare_path, "r", encoding="utf-8") as f:
                        if f.read() == "":
                            results = compare_with_pdfs(topic_dir, system, metrics, modelname)
                        else:
                            with open(compare_path, "r", encoding="utf-8") as f2:
                                results = json.load(f2)
                
                # Update category and global results
                for metric in metrics:
                    metric_key = f"{metric}_is_better"
                    if metric_key in results:
                        is_better = results[metric_key]
                        
                        # Update category results
                        category_results[category][system][metric]["total"] += 1
                        if is_better:
                            category_results[category][system][metric]["wins"] += 1
                        else:
                            category_results[category][system][metric]["losses"] += 1
                        
                        # Update global results
                        global_results[system][metric]["total"] += 1
                        if is_better:
                            global_results[system][metric]["wins"] += 1
                        else:
                            global_results[system][metric]["losses"] += 1
        print(category_results)
        print(global_results)
        # Save category results to CSV
        category_csv_path = os.path.join(category_dir, f"comparison_results_{modelname}.csv")
        try:
            with open(category_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                # Write header
                header = ["System", "Metric", "Wins", "Losses", "Total", "Win Rate"]
                writer.writerow(header)
                
                # Write data
                for system in systems:
                    for metric in metrics:
                        stats = category_results[category][system][metric]
                        if stats["total"] > 0:
                            win_rate = stats["wins"] / stats["total"]
                            writer.writerow([
                                system,
                                metric,
                                stats["wins"],
                                stats["losses"],
                                stats["total"],
                                f"{win_rate:.2%}"
                            ])
        except Exception as e:
            print(f"Error saving category results for {category}: {e}")
    
    # Save global results to CSV
    global_csv_path = os.path.join(root_dir, f"global_comparison_results_{modelname}.csv")
    try:
        with open(global_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Write header
            header = ["System", "Metric", "Wins", "Losses", "Total", "Win Rate"]
            writer.writerow(header)
            
            # Write data
            for system in systems:
                for metric in metrics:
                    stats = global_results[system][metric]
                    if stats["total"] > 0:
                        win_rate = stats["wins"] / stats["total"]
                        writer.writerow([
                            system,
                            metric,
                            stats["wins"],
                            stats["losses"],
                            stats["total"],
                            f"{win_rate:.2%}"
                        ])
        print(f"\nSaved global results to {global_csv_path}")
    except Exception as e:
        print(f"Error saving global results: {e}")

def aggregate_comparison_results_parallel(root_dir: str, systems: list[str], metrics: list[str], modelname: str, num_workers: int = 4) -> None:
    """
    Parallel version of aggregate_comparison_results that uses ThreadPoolExecutor.
    
    Args:
        root_dir (str): Root directory containing categories
        systems (list[str]): List of system names to compare
        metrics (list[str]): List of metrics to evaluate
        modelname (str): Name of the evaluation model
        num_workers (int, optional): Number of worker threads. Defaults to 4.
    """
    # Initialize results storage
    category_results = {}  # Results per category
    global_results = {     # Global results across all categories
        system: {
            metric: {"wins": 0, "losses": 0, "total": 0}
            for metric in metrics
        }
        for system in systems
    }
    
    # Create a lock for thread-safe updates to global results
    global_lock = threading.Lock()
    
    def process_topic(topic_dir: str, system: str) -> dict:
        """Process a single topic for a system."""
        results = compare_with_pdfs(topic_dir, system, metrics, modelname)
        return results
    
    def process_category(category: str) -> None:
        """Process a single category."""
        category_dir = os.path.join(root_dir, category)
        if not os.path.isdir(category_dir) or category == "pdfs":
            return
        
        # Initialize category results
        category_results[category] = {
            system: {
                metric: {"wins": 0, "losses": 0, "total": 0}
                for metric in metrics
            }
            for system in systems
        }
        
        # Create tasks for each topic-system combination
        tasks = []
        for topic in os.listdir(category_dir):
            topic_dir = os.path.join(category_dir, topic)
            if not os.path.isdir(topic_dir) or topic == "pdfs":
                continue
            
            for system in systems:
                system_dir = os.path.join(topic_dir, system)
                if os.path.exists(system_dir):
                    tasks.append((topic_dir, system))
        
        # Process tasks in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for topic_dir, system in tasks:
                futures.append(executor.submit(process_topic, topic_dir, system))
            
            # Process results as they complete
            for (topic_dir, system), future in zip(tasks, futures):
                try:
                    results = future.result()
                    
                    # Update category and global results
                    with global_lock:
                        for metric in metrics:
                            metric_key = f"{metric}_is_better"
                            if metric_key in results:
                                is_better = results[metric_key]
                                
                                # Update category results
                                category_results[category][system][metric]["total"] += 1
                                if is_better:
                                    category_results[category][system][metric]["wins"] += 1
                                else:
                                    category_results[category][system][metric]["losses"] += 1
                                
                                # Update global results
                                global_results[system][metric]["total"] += 1
                                if is_better:
                                    global_results[system][metric]["wins"] += 1
                                else:
                                    global_results[system][metric]["losses"] += 1
                except Exception as e:
                    print(f"Error processing {topic_dir}/{system}: {e}")
        
        # Save category results to CSV
        category_csv_path = os.path.join(category_dir, f"comparison_results_{modelname}.csv")
        try:
            with open(category_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                # Write header
                header = ["System", "Metric", "Wins", "Losses", "Total", "Win Rate"]
                writer.writerow(header)
                
                # Write data
                for system in systems:
                    for metric in metrics:
                        stats = category_results[category][system][metric]
                        if stats["total"] > 0:
                            win_rate = stats["wins"] / stats["total"]
                            writer.writerow([
                                system,
                                metric,
                                stats["wins"],
                                stats["losses"],
                                stats["total"],
                                f"{win_rate:.2%}"
                            ])
        except Exception as e:
            print(f"Error saving category results for {category}: {e}")
    
    # Process categories in parallel
    categories = [cat for cat in os.listdir(root_dir) 
                 if os.path.isdir(os.path.join(root_dir, cat)) and cat != "pdfs"]
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_category, category) for category in categories]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing category: {e}")
    
    # Save global results to CSV
    global_csv_path = os.path.join(root_dir, f"global_comparison_results_{modelname}.csv")
    try:
        with open(global_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Write header
            header = ["System", "Metric", "Wins", "Losses", "Total", "Win Rate"]
            writer.writerow(header)
            
            # Write data
            for system in systems:
                for metric in metrics:
                    stats = global_results[system][metric]
                    if stats["total"] > 0:
                        win_rate = stats["wins"] / stats["total"]
                        writer.writerow([
                            system,
                            metric,
                            stats["wins"],
                            stats["losses"],
                            stats["total"],
                            f"{win_rate:.2%}"
                        ])
    except Exception as e:
        print(f"Error saving global results: {e}")

# ------------- Main Evaluation Functions -------------

def evaluate(
    md_path: str, 
    model: str = "default",
    do_outline: bool = True, 
    do_content: bool = True, 
    do_reference: bool = True,
    criteria_type: str = "general"
) -> dict:
    """
    Evaluate a markdown file using specified criteria and model.
    
    Args:
        md_path (str): Path to the markdown file
        model (str, optional): Model name for evaluation. Defaults to "default".
        do_outline (bool, optional): Whether to evaluate outline. Defaults to True.
        do_content (bool, optional): Whether to evaluate content. Defaults to True.
        do_reference (bool, optional): Whether to evaluate references. Defaults to True.
        criteria_type (str, optional): Type of criteria to use ("general" or "domain"). Defaults to "general".
        
    Returns:
        dict: Dictionary containing all evaluation results
    """
    start_time = time.time()
    results = {}
    results_path = os.path.join(
        os.path.dirname(md_path), 
        f"results_{model}.json"
    )
    print("Start evaluating:", md_path)
    print("Using model:", model)
    with open(md_path, "r", encoding="utf-8") as f:
        content_str = f.read()
    content, _ = split_markdown_content_and_refs(content_str)
    content_path = os.path.join(os.path.dirname(md_path), "content.json")
    with open(content_path, "w", encoding="utf-8") as f:
        json.dump([content], f, ensure_ascii=False, indent=4)

    # Define required keys for each evaluation section
    if criteria_type == "general":
        outline_keys = ["Outline", "Outline_coverage", "Outline_structure", "Outline_no", "Outline_density"]
        content_keys = [
            "Coverage", "Structure", "Relevance", "Language", "Criticalness",
            "Faithfulness",
            "Images_density", "Equations_density", "Tables_density", "Citations_density", "Sentence_no",
            "Claim_density"
        ]
        reference_keys = ["Reference", "Reference_density", "Reference_quality", "Reference_no"]
    elif criteria_type == "domain":
        outline_keys = ["Outline_domain"]
        content_keys = [
            "Coverage_domain", "Structure_domain", "Relevance_domain", "Language_domain", "Criticalness_domain"
        ]
        reference_keys = [
            "Reference_domain"
        ]
    else:
        raise ValueError(f"Invalid criteria type: {criteria_type}")

    # Load existing results if available
    if os.path.exists(results_path):
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                results = json.load(f)
        except Exception as e:
            print("Error in loading existing results:", e)

    # Evaluate outline if requested and not already complete
    if do_outline:
        print("Evaluating outline...")
        if not all(k in results for k in outline_keys):
            try:
                results.update(evaluate_outline(md_path, criteria_type=criteria_type))
            except Exception as e:
                print("Error in evaluating outline:", e)
        else:
            print("Outline already complete, skip.")
    else:
        print("Skip evaluating outline.")

    # Evaluate content if requested and not already complete
    if do_content:
        print("Evaluating content...")
        if not all(k in results for k in content_keys):
            try:
                results.update(evaluate_content(md_path, criteria_type=criteria_type))
            except Exception as e:
                print("Error in evaluating content:", e)
        else:
            print("Content already complete, skip.")
    else:
        print("Skip evaluating content.")

    # Evaluate references if requested and not already complete
    if do_reference:
        print("Evaluating reference...")
        if not all(k in results for k in reference_keys):
            try:
                results.update(evaluate_reference(md_path, criteria_type=criteria_type))
            except Exception as e:
                print("Error in evaluating reference:", e)
        else:
            print("Reference already complete, skip.")
    else:
        print("Skip evaluating reference.")

    # Save results
    try:
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print("Error in saving results:", e)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Evaluation completed in {elapsed_time:.2f} seconds.")
    return results

def evaluate_pairs(topic_dir: str, system: str, metrics: list[str]) -> dict:
    """
    Compare evaluation scores between pdfs and another system for a given topic directory.
    
    Args:
        topic_dir (str): Path to the topic directory
        system (str): Name of the system to compare with pdfs
        metrics (list[str]): List of metrics to evaluate, can include 'outline', 'reference', 'content'
        
    Returns:
        dict: Dictionary containing score differences (system - pdfs) for each metric
    """
    results = {}
    
    # Get pdfs results
    pdfs_dir = os.path.join(topic_dir, "pdfs")
    if not os.path.exists(pdfs_dir):
        print(f"Error: pdfs directory not found at {pdfs_dir}")
        return results
    
    # Get system results
    system_dir = os.path.join(topic_dir, system)
    if not os.path.exists(system_dir):
        print(f"Error: system directory not found at {system_dir}")
        return results
    
    # Find markdown files
    pdfs_md = [f for f in os.listdir(pdfs_dir) if f.lower().endswith(".md")]
    system_md = [f for f in os.listdir(system_dir) if f.lower().endswith(".md")]
    
    if not pdfs_md:
        print(f"Error: No markdown file found in pdfs directory")
        return results
    if not system_md:
        print(f"Error: No markdown file found in system directory")
        return results
    
    pdfs_md_path = os.path.join(pdfs_dir, pdfs_md[0])
    system_md_path = os.path.join(system_dir, system_md[0])
    
    # Evaluate each requested metric
    for metric in metrics:
        if metric.lower() == "outline":
            print("\nEvaluating outline...")
            pdfs_outline = evaluate_outline(pdfs_md_path)
            system_outline = evaluate_outline(system_md_path)
            
            # Calculate differences
            for key in pdfs_outline:
                if key in system_outline:
                    diff = system_outline[key] - pdfs_outline[key]
                    results[f"Outline_{key}"] = diff
                    print(f"{key}: {pdfs_outline[key]:+.4f} | {system_outline[key]:+.4f} | {diff:+.4f}")
        
        elif metric.lower() == "reference":
            print("\nEvaluating reference...")
            pdfs_reference = evaluate_reference(pdfs_md_path)
            system_reference = evaluate_reference(system_md_path)
            
            # Calculate differences
            for key in pdfs_reference:
                if key in system_reference:
                    diff = system_reference[key] - pdfs_reference[key]
                    results[f"Reference_{key}"] = diff
                    print(f"{key}: {pdfs_reference[key]:+.4f} | {system_reference[key]:+.4f} | {diff:+.4f}")
        
        elif metric.lower() == "content":
            print("\nEvaluating content...")
            pdfs_content = evaluate_content(pdfs_md_path)
            system_content = evaluate_content(system_md_path)
            
            # Calculate differences
            for key in pdfs_content:
                if key in system_content:
                    diff = system_content[key] - pdfs_content[key]
                    results[f"Content_{key}"] = diff
                    print(f"{key}: {pdfs_content[key]:+.4f} | {system_content[key]:+.4f} | {diff:+.4f}")
    
    return results

def evaluate_topic_ranking(topic_dir: str, metrics: list[str]) -> dict:
    """
    Evaluate and rank different systems' outputs for a given topic.
    
    Args:
        topic_dir (str): Path to the topic directory
        metrics (list[str]): List of metrics to evaluate, can include 'outline', 'content', 'reference'
        
    Returns:
        dict: Dictionary containing rankings for each metric
    """
    results = {}
    
    # Get all systems in the topic directory
    systems = [d for d in os.listdir(topic_dir) if os.path.isdir(os.path.join(topic_dir, d))]
    if not systems:
        print(f"No systems found in {topic_dir}")
        return results
    
    # Create system to index mapping
    system_to_index = {system: str(i+1) for i, system in enumerate(systems)}
    index_to_system = {str(i+1): system for i, system in enumerate(systems)}
    
    # Extract topic name from path
    topic = extract_topic_from_path(topic_dir)
    
    # Process each metric
    for metric in metrics:
        if metric.lower() == "outline":
            # Collect outlines from each system
            outlines = {}
            for system in systems:
                outline_path = os.path.join(topic_dir, system, "outline.json")
                if os.path.exists(outline_path):
                    try:
                        with open(outline_path, "r", encoding="utf-8") as f:
                            outline = json.load(f)
                            outlines[system_to_index[system]] = "\n".join([json.dumps(item, ensure_ascii=False) for item in outline])
                    except Exception as e:
                        print(f"Error reading outline for {system}: {e}")
            
            if outlines:
                # Generate prompt and get ranking
                prompt = OUTLINE_RANKING_PROMPT.format(
                    topic=topic,
                    outlines="\n\n".join([f"Index {idx}:\n{outline}" for idx, outline in outlines.items()])
                )
                try:
                    ranking = judge.judge(prompt)
                    if isinstance(ranking, dict):
                        # Convert index-based ranking back to system-based ranking
                        system_ranking = {index_to_system[idx]: rank for idx, rank in ranking.items()}
                        results["outline_ranking"] = system_ranking
                except Exception as e:
                    print(f"Error getting outline ranking: {e}")
        
        elif metric.lower() == "content":
            # Collect content from each system
            contents = {}
            for system in systems:
                md_files = [f for f in os.listdir(os.path.join(topic_dir, system)) if f.lower().endswith(".md")]
                if md_files:
                    try:
                        with open(os.path.join(topic_dir, system, md_files[0]), "r", encoding="utf-8") as f:
                            content = f.read()
                            content_str, _ = split_markdown_content_and_refs(content)
                            contents[system_to_index[system]] = content_str
                    except Exception as e:
                        print(f"Error reading content for {system}: {e}")
            
            if contents:
                # Generate prompt and get ranking
                prompt = CONTENT_RANKING_PROMPT.format(
                    topic=topic,
                    contents="\n\n".join([f"Index {idx}:\n{content}" for idx, content in contents.items()])
                )
                try:
                    ranking = judge.judge(prompt)
                    if isinstance(ranking, dict):
                        # Convert index-based ranking back to system-based ranking
                        system_ranking = {index_to_system[idx]: rank for idx, rank in ranking.items()}
                        results["content_ranking"] = system_ranking
                except Exception as e:
                    print(f"Error getting content ranking: {e}")
        
        elif metric.lower() == "reference":
            # Collect references from each system
            references = {}
            for system in systems:
                ref_path = os.path.join(topic_dir, system, "references.json")
                if os.path.exists(ref_path):
                    try:
                        with open(ref_path, "r", encoding="utf-8") as f:
                            refs = json.load(f)
                            references[system_to_index[system]] = "\n".join(refs)
                    except Exception as e:
                        print(f"Error reading references for {system}: {e}")
            
            if references:
                # Generate prompt and get ranking
                prompt = REFERENCE_RANKING_PROMPT.format(
                    topic=topic,
                    references="\n\n".join([f"Index {idx}:\n{refs}" for idx, refs in references.items()])
                )
                try:
                    ranking = judge.judge(prompt)
                    if isinstance(ranking, dict):
                        # Convert index-based ranking back to system-based ranking
                        system_ranking = {index_to_system[idx]: rank for idx, rank in ranking.items()}
                        results["reference_ranking"] = system_ranking
                except Exception as e:
                    print(f"Error getting reference ranking: {e}")
    
    # Save results to topic directory
    results_path = os.path.join(topic_dir, "ranking_results.json")
    try:
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error saving ranking results: {e}")
    
    return results

def process_system(
    md_path: str,
    model: str,
    results_path: str,
    topic: str,
    system: str,
    do_outline: bool,
    do_content: bool,
    do_reference: bool,
    criteria_type: str
) -> None:
    """
    Process a single system's evaluation.
    
    Args:
        md_path (str): Path to the markdown file
        model (str): Model name for evaluation
        results_path (str): Path to save results
        topic (str): Topic name
        system (str): System name
        do_outline (bool): Whether to evaluate outline
        do_content (bool): Whether to evaluate content
        do_reference (bool): Whether to evaluate references
        criteria_type (str): Type of criteria to use ("general" or "domain"). Defaults to "general".
    """
    print(f"[{topic}/{system}] Evaluating: {md_path}")
    evaluate(md_path, 
             model=model,
             do_outline=do_outline, 
             do_content=do_content, 
             do_reference=do_reference,
             criteria_type=criteria_type)

def batch_evaluate_by_cat(
    cats: list[str],
    model: str,
    do_outline: bool = True,
    do_content: bool = True,
    do_reference: bool = True,
    num_workers: int = 1,
    criteria_type: str = "general"
) -> None:
    """
    Batch evaluate all markdown files in specified categories.
    
    Args:
        cats (list[str]): List of category names to evaluate
        model (str): Model name for evaluation
        do_outline (bool, optional): Whether to evaluate outline. Defaults to True.
        do_content (bool, optional): Whether to evaluate content. Defaults to True.
        do_reference (bool, optional): Whether to evaluate references. Defaults to True.
        num_workers (int, optional): Number of worker threads. Defaults to 1.
        criteria_type (str, optional): Type of criteria to use ("general" or "domain"). Defaults to "general".
    """
    for cat in cats:
        base_dir = os.path.join("surveys", cat)
        topics = [d for d in os.listdir(base_dir) 
                if os.path.isdir(os.path.join(base_dir, d))]
        print(f"Found topics: {topics}")

        tasks = []

        for topic in topics:
            topic_path = os.path.join(base_dir, topic)
            systems = [d for d in os.listdir(topic_path)
                    if os.path.isdir(os.path.join(topic_path, d))]
            print(f"Topic: {topic} | Systems: {systems}")
            for system in systems:
                sys_path = os.path.join(topic_path, system)
                # Find markdown files
                md_files = [f for f in os.listdir(sys_path) 
                            if f.lower().endswith(".md")]
                pdf_files = [f for f in os.listdir(sys_path) 
                            if f.lower().endswith(".pdf")]
                results_path = os.path.join(
                    sys_path, 
                    f"results_{model}.json"
                )
                # Convert PDF to markdown if no markdown exists
                if not md_files:
                    if pdf_files:
                        pdf_path = os.path.join(sys_path, pdf_files[0])
                        print(f"[{topic}/{system}] No md found, converting pdf: {pdf_path}")
                        md_path, _ = pdf2md(pdf_path, sys_path)
                        if not md_path:
                            print(f"[{topic}/{system}] PDF to md failed, skip.")
                            continue
                    else:
                        print(f"[{topic}/{system}] No md or pdf found, skip.")
                        continue
                else:
                    md_path = os.path.join(sys_path, md_files[0])

                tasks.append((md_path, model, results_path, topic, system, do_outline, do_content, do_reference, criteria_type))
        
        if num_workers == 1:
            # Sequential execution
            for args in tasks:
                process_system(*args)
        else:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_system = {executor.submit(process_system, *args): args for args in tasks}
                for future in as_completed(future_to_system):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Exception in thread: {e}")

def batch_evaluate_by_system(
    system_list: list[str],
    model: str,
    tasks_json_path: str = "surveys/tasks.json",
    do_outline: bool = True,
    do_content: bool = True,
    do_reference: bool = True,
    num_workers: int = 1,
    criteria_type: str = "general"
) -> None:
    """
    Batch evaluate all tasks for specified systems.
    
    Args:
        system_list (list[str]): List of system names to evaluate
        model (str): Model name for evaluation
        tasks_json_path (str, optional): Path to tasks mapping JSON. Defaults to "surveys/tasks.json".
        do_outline (bool, optional): Whether to evaluate outline. Defaults to True.
        do_content (bool, optional): Whether to evaluate content. Defaults to True.
        do_reference (bool, optional): Whether to evaluate references. Defaults to True.
        num_workers (int, optional): Number of worker threads. Defaults to 1.
        criteria_type (str, optional): Type of criteria to use ("general" or "domain"). Defaults to "general".
    """
    # Read tasks.json
    with open(tasks_json_path, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    
    tasks_to_run = []
    for system in system_list:
        if system not in tasks:
            print(f"System {system} not found in {tasks_json_path}, skip.")
            continue
        for topic_map in tasks[system]:  # topic_map: {topic: path}
            for topic, rel_path in topic_map.items():
                sys_path = os.path.join("surveys", rel_path)
                # Find markdown or PDF files
                md_files = [f for f in os.listdir(sys_path) if f.lower().endswith(".md")]
                pdf_files = [f for f in os.listdir(sys_path) if f.lower().endswith(".pdf")]
                results_path = os.path.join(
                    sys_path, 
                    f"results_{model}.json"
                )
                if not md_files:
                    if pdf_files:
                        pdf_path = os.path.join(sys_path, pdf_files[0])
                        print(f"[{topic}/{system}] No md found, converting pdf: {pdf_path}")
                        md_path, _ = pdf2md(pdf_path, sys_path)
                        if not md_path:
                            print(f"[{topic}/{system}] PDF to md failed, skip.")
                            continue
                    else:
                        print(f"[{topic}/{system}] No md or pdf found, skip.")
                        continue
                else:
                    md_path = os.path.join(sys_path, md_files[0])
                tasks_to_run.append((md_path, model, results_path, topic, system, do_outline, do_content, do_reference, criteria_type))
    
    if num_workers == 1:
        for args in tasks_to_run:
            process_system(*args)
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_system = {executor.submit(process_system, *args): args for args in tasks_to_run}
            for future in as_completed(future_to_system):
                try:
                    future.result()
                except Exception as e:
                    print(f"Exception in thread: {e}")

# ------------- Average Score Calculation -------------

def calculate_average_score(cat: str, system: str, model: str) -> dict:
    """
    Calculate average scores for a specific category, system, and model.
    
    Args:
        cat (str): Category name (e.g., "cs")
        system (str): System name (e.g., "InteractiveSurvey")
        model (str): Model name (e.g., "qwen-plus")
        
    Returns:
        dict: Dictionary containing average scores
    """
    base_dir = os.path.join("surveys", cat)
    topics = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    total_results = {}
    count = 0

    for topic in topics:
        topic_path = os.path.join(base_dir, topic)
        sys_path = os.path.join(topic_path, system)
        results_path = os.path.join(
            sys_path, 
            f"results_{model}.json"
        )
        if not os.path.exists(results_path):
            continue
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        for key, value in results.items():
            # Handle both general and domain-specific keys
            if key not in total_results:
                total_results[key] = 0
            total_results[key] += value
        count += 1

    if count == 0:
        average_scores = {}
    else:
        average_scores = {key: round(value / count, 4) for key, value in total_results.items()}

    # Write to average_results.json
    avg_results_path = os.path.join("surveys", cat, "average_results.json")
    if os.path.exists(avg_results_path):
        with open(avg_results_path, "r", encoding="utf-8") as f:
            avg_results_data = json.load(f)
    else:
        avg_results_data = {}

    # Structure: system_name -> model_name -> averages
    if system not in avg_results_data:
        avg_results_data[system] = {}
    avg_results_data[system][model] = average_scores

    with open(avg_results_path, "w", encoding="utf-8") as f:
        json.dump(avg_results_data, f, ensure_ascii=False, indent=4)

    return average_scores

def calculate_average_score_by_cat(cat: str) -> dict:
    """
    Calculate average scores for all systems in a specific category.
    
    Args:
        cat (str): Category name (e.g., "cs")
        
    Returns:
        dict: Dictionary containing average scores for all systems in the category
    """
    base_dir = os.path.join("surveys", cat)
    topics = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    # Dictionary to store results for each system and model
    all_results = {}
    
    for topic in topics:
        topic_path = os.path.join(base_dir, topic)
        systems = [d for d in os.listdir(topic_path) if os.path.isdir(os.path.join(topic_path, d))]
        
        for system in systems:
            sys_path = os.path.join(topic_path, system)
            # Find all results files for this system
            results_files = glob.glob(os.path.join(sys_path, "results_*.json"))
            
            for results_file in results_files:
                # Extract model name from filename (e.g., "results_qwen-plus.json" -> "qwen-plus")
                model = os.path.basename(results_file).replace("results_", "").replace(".json", "")
                
                if system not in all_results:
                    all_results[system] = {}
                if model not in all_results[system]:
                    all_results[system][model] = {"total": {}, "count": 0}
                
                try:
                    with open(results_file, "r", encoding="utf-8") as f:
                        results = json.load(f)
                    
                    # Add scores to total
                    for key, value in results.items():
                        if key not in all_results[system][model]["total"]:
                            all_results[system][model]["total"][key] = 0
                        all_results[system][model]["total"][key] += value
                    
                    all_results[system][model]["count"] += 1
                except Exception as e:
                    print(f"Error processing {results_file}: {e}")
                    continue
    
    # Calculate averages for each system and model
    average_scores = {}
    for system, models in all_results.items():
        average_scores[system] = {}
        for model, data in models.items():
            if data["count"] > 0:
                average_scores[system][model] = {
                    key: round(value / data["count"], 4)
                    for key, value in data["total"].items()
                }
    
    # Write to average_results.json
    avg_results_path = os.path.join("surveys", cat, "average_results.json")
    with open(avg_results_path, "w", encoding="utf-8") as f:
        json.dump(average_scores, f, ensure_ascii=False, indent=4)
    
    return average_scores

def calculate_average_score_by_system(system: str, model: str) -> dict:
    """
    Calculate average scores for a specific system and model across all categories.
    
    Args:
        system (str): System name (e.g., "AutoSurvey")
        model (str): Model name (e.g., "gpt-4")
        
    Returns:
        dict: Dictionary containing average scores for the system-model combination
    """
    base_dir = "surveys"
    all_scores = {}
    count = 0
    
    # Get all category directories
    cats = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for cat in cats:
        cat_path = os.path.join(base_dir, cat)
        avg_results_path = os.path.join(cat_path, "average_results.json")
        
        if os.path.exists(avg_results_path):
            try:
                with open(avg_results_path, "r", encoding="utf-8") as f:
                    cat_results = json.load(f)
                
                if system in cat_results and model in cat_results[system]:
                    system_model_scores = cat_results[system][model]
                    for metric, value in system_model_scores.items():
                        if metric not in all_scores:
                            all_scores[metric] = 0
                        all_scores[metric] += value
                    count += 1
            except Exception as e:
                print(f"Error processing {avg_results_path}: {e}")
                continue
    
    # Calculate averages if we have data
    if count > 0:
        average_scores = {
            metric: round(value / count, 4)
            for metric, value in all_scores.items()
        }
    else:
        average_scores = {}
    
    return average_scores

def calculate_all_cats_average_scores() -> dict:
    """
    Calculate average scores for all categories and store them in their respective average_results.json files.
    
    Returns:
        dict: Dictionary containing average scores for all categories
    """
    base_dir = "surveys"
    all_cats_results = {}
    
    # Get all category directories
    cats = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for cat in cats:
        print(f"Calculating average scores for category: {cat}")
        cat_results = calculate_average_score_by_cat(cat)
        all_cats_results[cat] = cat_results
    
    return all_cats_results

# ------------- Average Score Clearing -------------
def clear_average_score_by_cat(cat: str) -> None:
    """
    Clear average results for a specific category.
    
    Args:
        cat (str): Category name (e.g., "cs")
    """
    avg_results_path = os.path.join("surveys", cat, "average_results.json")
    if os.path.exists(avg_results_path):
        try:
            os.remove(avg_results_path)
            print(f"Removed average results for category: {cat}")
        except Exception as e:
            print(f"Failed to remove {avg_results_path}: {e}")

def clear_average_score_by_system(system: str, model: str) -> None:
    """
    Clear average results for a specific system and model from all category average results.
    
    Args:
        system (str): System name (e.g., "AutoSurvey")
        model (str): Model name (e.g., "gpt-4")
    """
    base_dir = "surveys"
    cats = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for cat in cats:
        avg_results_path = os.path.join(base_dir, cat, "average_results.json")
        if os.path.exists(avg_results_path):
            try:
                with open(avg_results_path, "r", encoding="utf-8") as f:
                    avg_results = json.load(f)
                
                if system in avg_results:
                    if model in avg_results[system]:
                        del avg_results[system][model]
                        # Remove system if no models remain
                        if not avg_results[system]:
                            del avg_results[system]
                
                with open(avg_results_path, "w", encoding="utf-8") as f:
                    json.dump(avg_results, f, ensure_ascii=False, indent=4)
                
                print(f"Cleared {system}/{model} from {cat} average results")
            except Exception as e:
                print(f"Error processing {avg_results_path}: {e}")

def clear_all_average_scores() -> None:
    """
    Clear all average results files (average_results.json and global_average_results.json).
    """
    base_dir = "surveys"
    
    # Clear category average results
    cats = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    for cat in cats:
        avg_results_path = os.path.join(base_dir, cat, "average_results.json")
        if os.path.exists(avg_results_path):
            try:
                os.remove(avg_results_path)
                print(f"Removed average results for category: {cat}")
            except Exception as e:
                print(f"Failed to remove {avg_results_path}: {e}")
    
    # Clear global average results
    global_results_path = os.path.join(base_dir, "global_average_results.json")
    if os.path.exists(global_results_path):
        try:
            os.remove(global_results_path)
            print("Removed global average results")
        except Exception as e:
            print(f"Failed to remove {global_results_path}: {e}")

def clear_scores(cat: str, system: str, model: str, target: str | list[str] = "All") -> None:
    """
    Clear evaluation results for a specific category, system, and model.
    
    Args:
        cat (str): Category name (e.g., "cs")
        system (str): System name (e.g., "InteractiveSurvey")
        model (str): Model name (e.g., "qwen-plus")
        target (str | list[str]): Target metric(s) to clear. Can be:
            - "All": clear all metrics
            - A single metric name (e.g., "Outline", "Coverage", "Reference_density")
            - A list of metric names (e.g., ["Outline", "Coverage", "Reference_density"])
    """
    base_dir = os.path.join("surveys", cat)
    topics = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for topic in topics:
        topic_path = os.path.join(base_dir, topic)
        sys_path = os.path.join(topic_path, system)
        results_path = os.path.join(sys_path, f"results_{model}.json")
        
        if os.path.exists(results_path):
            try:
                with open(results_path, "r", encoding="utf-8") as f:
                    results = json.load(f)
                
                if target == "All":
                    # Remove the entire file
                    os.remove(results_path)
                else:
                    # Convert single metric to list for uniform handling
                    metrics_to_remove = [target] if isinstance(target, str) else target
                    # Remove specified metrics
                    for metric in metrics_to_remove:
                        if metric in results:
                            del results[metric]
                    
                    # Save updated results
                    with open(results_path, "w", encoding="utf-8") as f:
                        json.dump(results, f, ensure_ascii=False, indent=4)
            except Exception as e:
                print(f"Error processing {results_path}: {e}")
    
    # Update average_results.json
    avg_results_path = os.path.join("surveys", cat, "average_results.json")
    if os.path.exists(avg_results_path):
        try:
            with open(avg_results_path, "r", encoding="utf-8") as f:
                avg_results_data = json.load(f)
            
            if system in avg_results_data and model in avg_results_data[system]:
                if target == "All":
                    # Remove the entire system-model entry
                    del avg_results_data[system][model]
                    if not avg_results_data[system]:
                        del avg_results_data[system]
                else:
                    # Convert single metric to list for uniform handling
                    metrics_to_remove = [target] if isinstance(target, str) else target
                    # Remove specified metrics
                    for metric in metrics_to_remove:
                        if metric in avg_results_data[system][model]:
                            del avg_results_data[system][model][metric]
            
            with open(avg_results_path, "w", encoding="utf-8") as f:
                json.dump(avg_results_data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"Error processing {avg_results_path}: {e}")

def clear_all_scores(model: str = None, target: str | list[str] = None) -> None:
    """
    Clear all scores for all categories, systems, and models with clear_scores function.
    
    Args:
        model (str, optional): Model name to clear scores for. If None, clear for all models.
        target (str | list[str], optional): Target metric(s) to clear. Can be:
            - None: clear all metrics
            - A single metric name (e.g., "Outline", "Coverage", "Reference_density")
            - A list of metric names (e.g., ["Outline", "Coverage", "Reference_density"])
    """
    base_dir = "surveys"
    for cat in os.listdir(base_dir):
        cat_path = os.path.join(base_dir, cat)
        # Check if it is a directory
        if os.path.isdir(cat_path):
            # Get all topics in this category
            for topic in os.listdir(cat_path):
                topic_path = os.path.join(cat_path, topic)
                if os.path.isdir(topic_path):
                    # Get all systems in this topic
                    for system in os.listdir(topic_path):
                        system_path = os.path.join(topic_path, system)
                        if os.path.isdir(system_path):
                            # Check if there are any result files
                            if model is None:
                                # Clear all result files for this system
                                results_files = [f for f in os.listdir(system_path) if f.startswith("results_") and f.endswith(".json")]
                                for results_file in results_files:
                                    current_model = results_file.replace("results_", "").replace(".json", "")
                                    try:
                                        if target is None:
                                            clear_scores(cat, system, current_model, "All")
                                        elif isinstance(target, str):
                                            clear_scores(cat, system, current_model, target)
                                        else:  # target is a list
                                            for t in target:
                                                clear_scores(cat, system, current_model, t)
                                    except Exception as e:
                                        print(f"Error clearing scores for {cat}/{topic}/{system}/{current_model}: {e}")
                            else:
                                # Clear specific model's result file
                                results_file = f"results_{model}.json"
                                if os.path.exists(os.path.join(system_path, results_file)):
                                    try:
                                        if target is None:
                                            clear_scores(cat, system, model, "All")
                                        elif isinstance(target, str):
                                            clear_scores(cat, system, model, target)
                                        else:  # target is a list
                                            for t in target:
                                                clear_scores(cat, system, model, t)
                                    except Exception as e:
                                        print(f"Error clearing scores for {cat}/{topic}/{system}/{model}: {e}")

def delete_system(systems: list[str]) -> None:
    """
    Delete specified system folders under all topics in all categories.
    
    Args:
        systems (list[str]): List of system names to delete
    """
    base_dir = "surveys"
    for cat in os.listdir(base_dir):
        cat_path = os.path.join(base_dir, cat)
        # Check if it is a directory
        if os.path.isdir(cat_path):
            # Get all topics in this category
            for topic in os.listdir(cat_path):
                topic_path = os.path.join(cat_path, topic)
                if os.path.isdir(topic_path):
                    # Check each system
                    for system in systems:
                        system_path = os.path.join(topic_path, system)
                        if os.path.exists(system_path):
                            try:
                                import shutil
                                shutil.rmtree(system_path)
                                print(f"Deleted {cat}/{topic}/{system}")
                            except Exception as e:
                                print(f"Error deleting {cat}/{topic}/{system}: {e}")

# ------------- Data Post-processing -------------

def aggregate_results_to_csv(cat: str) -> None:
    """
    Aggregate all results from a category into a CSV file.
    For specified metrics, if value is 0, fill with average of the same model.
    The metrics that need to be filled with average if 0 are:
    - Outline, Outline_coverage, Outline_structure, Outline_no
    - Reference, Reference_density, Reference_quality, Reference_no
    - Coverage, Structure, Relevance, Language, Criticalness
    - Images_density, Equations_density, Tables_density, Total_density
    - Citations_density, Sentence_no, Claim_density
    
    Args:
        cat (str): Category name (e.g., "cs")
    """
    base_dir = os.path.join("surveys", cat)
    all_results = []
    
    metrics_to_fill = [
        "Outline",  "Outline_coverage", "Outline_structure", "Outline_no", "Outline_density",
        "Reference", "Reference_density", "Reference_quality", "Reference_no",
        "Coverage", "Structure", "Relevance", "Language", "Criticalness", "Faithfulness",
        "Images_density", "Equations_density", "Tables_density", 
        "Citations_density", "Sentence_no", "Claim_density",
        "Outline_domain", "Reference_domain",
        "Coverage_domain", "Structure_domain", "Relevance_domain", "Language_domain", "Criticalness_domain"
    ]
    
    # Get all topics
    topics = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    # Collect all results
    for topic in topics:
        topic_path = os.path.join(base_dir, topic)
        systems = [d for d in os.listdir(topic_path) if os.path.isdir(os.path.join(topic_path, d))]
        
        for system in systems:
            sys_path = os.path.join(topic_path, system)
            # Find all results files
            results_files = glob.glob(os.path.join(sys_path, "results_*.json"))
            
            for results_file in results_files:
                model = os.path.basename(results_file).replace("results_", "").replace(".json", "")
                try:
                    with open(results_file, "r", encoding="utf-8") as f:
                        results = json.load(f)
                    
                    # Add basic info
                    entry = {
                        "topic": topic,
                        "system": system,
                        "model": model,
                        "category": cat  # Add category column
                    }
                    # Add all metrics
                    entry.update(results)
                    all_results.append(entry)
                except Exception as e:
                    print(f"Error processing {results_file}: {e}")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Fill 0 values with system averages for specified metrics
    for metric in metrics_to_fill:
        if metric in df.columns:
            # Calculate system averages for this metric
            system_avgs = df[df[metric] != 0].groupby('system')[metric].mean()
            
            # Fill 0 values with corresponding system average
            for system, avg in system_avgs.items():
                mask = (df['system'] == system) & (df[metric] == 0)
                df.loc[mask, metric] = avg
    
    # Save to CSV
    output_path = os.path.join(base_dir, f"{cat}_results.csv")
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

def supplement_missing_scores(cat: str = None, model: str = None, system: str = None) -> None:
    """
    Check and supplement missing scores for specific metrics.
    If any parameter is None, process all items for that parameter.
    
    Args:
        cat (str, optional): Category name (e.g., "cs"). If None, process all categories.
        model (str, optional): Model name (e.g., "gpt-4"). If None, process all models.
        system (str, optional): System name. If None, process all systems.
    """
    # Define metrics to check and their corresponding evaluation functions
    metric_functions = {
        "Outline": evaluate_outline_llm,
        "Reference": evaluate_reference_llm,
        "Content": evaluate_content_llm_simultaneous,  # Changed to handle all content metrics at once
        "Outline_coverage": evaluate_outline_coverage,
        "Outline_structure": evaluate_outline_structure,
        "Reference_quality": evaluate_reference_quality
    }
    
    # Define content metrics
    content_metrics = ["Coverage", "Structure", "Relevance", "Language", "Criticalness"]
    
    # Get categories to process
    base_dir = "surveys"
    if cat is not None:
        cats = [cat]
    else:
        cats = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for current_cat in cats:
        cat_dir = os.path.join(base_dir, current_cat)
        topics = [d for d in os.listdir(cat_dir) if os.path.isdir(os.path.join(cat_dir, d))]
        
        for topic in topics:
            topic_path = os.path.join(cat_dir, topic)
            systems = [d for d in os.listdir(topic_path) if os.path.isdir(os.path.join(topic_path, d))]
            
            # Filter systems if specified
            if system is not None:
                systems = [s for s in systems if s == system]
                if not systems:
                    print(f"System {system} not found in {topic}")
                    continue
            
            for current_system in systems:
                sys_path = os.path.join(topic_path, current_system)
                
                # Get all results files
                if model is not None:
                    results_files = [f"results_{model}.json"]
                else:
                    results_files = [f for f in os.listdir(sys_path) if f.startswith("results_") and f.endswith(".json")]
                
                for results_file in results_files:
                    results_path = os.path.join(sys_path, results_file)
                    current_model = results_file.replace("results_", "").replace(".json", "")
                    
                    if not os.path.exists(results_path):
                        continue
                        
                    try:
                        # Read current results
                        with open(results_path, "r", encoding="utf-8") as f:
                            results = json.load(f)
                        
                        needs_update = False
                        
                        # Check each metric group
                        for metric_group, eval_func in metric_functions.items():
                            if metric_group == "Content":
                                # Check if any content metric is missing
                                if any(results.get(metric, 0) == 0 for metric in content_metrics):
                                    print(f"Found missing content scores in {current_cat}/{topic}/{current_system}/{current_model}")
                                    
                                    # Find the markdown file
                                    md_files = [f for f in os.listdir(sys_path) if f.lower().endswith(".md")]
                                    if not md_files:
                                        print(f"No markdown file found in {sys_path}")
                                        continue
                                        
                                    md_path = os.path.join(sys_path, md_files[0])
                                    
                                    # Re-evaluate all content metrics at once
                                    try:
                                        # Evaluate general metrics
                                        new_scores = eval_func(md_path, criteria_type="general")
                                        if isinstance(new_scores, dict):
                                            for metric in content_metrics:
                                                if metric in new_scores:
                                                    results[metric] = new_scores[metric]
                                                    needs_update = True
                                                    print(f"Updated {metric} score to {new_scores[metric]}")
                                    except Exception as e:
                                        print(f"Error evaluating content metrics for {current_cat}/{topic}/{current_system}/{current_model}: {e}")
                            elif metric_group in ["Outline_coverage", "Outline_structure"]:
                                # Handle outline-specific metrics
                                if metric_group in results and results[metric_group] == 0:
                                    print(f"Found missing score for {metric_group} in {current_cat}/{topic}/{current_system}/{current_model}")
                                    
                                    # Find the outline.json file
                                    outline_json_path = os.path.join(sys_path, "outline.json")
                                    if not os.path.exists(outline_json_path):
                                        print(f"No outline.json found in {sys_path}")
                                        continue
                                    
                                    # Re-evaluate the metric
                                    try:
                                        if metric_group == "Outline_coverage":
                                            new_score = eval_func(outline_json_path)
                                            results[metric_group] = new_score
                                        else:  # Outline_structure
                                            new_score, _ = eval_func(outline_json_path)
                                            results[metric_group] = new_score
                                        needs_update = True
                                        print(f"Updated {metric_group} score to {new_score}")
                                    except Exception as e:
                                        print(f"Error evaluating {metric_group} for {current_cat}/{topic}/{current_system}/{current_model}: {e}")
                            elif metric_group == "Reference_quality":
                                # Handle reference quality metric
                                if metric_group in results and results[metric_group] == 0:
                                    print(f"Found missing score for {metric_group} in {current_cat}/{topic}/{current_system}/{current_model}")
                                    
                                    # Find the markdown file
                                    md_files = [f for f in os.listdir(sys_path) if f.lower().endswith(".md")]
                                    if not md_files:
                                        print(f"No markdown file found in {sys_path}")
                                        continue
                                        
                                    md_path = os.path.join(sys_path, md_files[0])
                                    
                                    # Re-evaluate the metric
                                    try:
                                        new_scores = eval_func(md_path)
                                        if isinstance(new_scores, dict) and metric_group in new_scores:
                                            results[metric_group] = new_scores[metric_group]
                                            needs_update = True
                                            print(f"Updated {metric_group} score to {new_scores[metric_group]}")
                                    except Exception as e:
                                        print(f"Error evaluating {metric_group} for {current_cat}/{topic}/{current_system}/{current_model}: {e}")
                            else:
                                # Handle other metrics (Outline and Reference)
                                if metric_group in results and results[metric_group] == 0:
                                    print(f"Found missing score for {metric_group} in {current_cat}/{topic}/{current_system}/{current_model}")
                                    
                                    # Find the markdown file
                                    md_files = [f for f in os.listdir(sys_path) if f.lower().endswith(".md")]
                                    if not md_files:
                                        print(f"No markdown file found in {sys_path}")
                                        continue
                                        
                                    md_path = os.path.join(sys_path, md_files[0])
                                    
                                    # Re-evaluate the metric
                                    try:
                                        if metric_group == "Outline":
                                            # For outline, we need to use the outline.json path
                                            outline_json_path = os.path.join(sys_path, "outline.json")
                                            new_scores = eval_func(outline_json_path)
                                        else:
                                            new_scores = eval_func(md_path)
                                        
                                        # Update results
                                        if isinstance(new_scores, dict):
                                            if metric_group in new_scores:
                                                results[metric_group] = new_scores[metric_group]
                                                needs_update = True
                                                print(f"Updated {metric_group} score to {new_scores[metric_group]}")
                                        else:
                                            print(f"Unexpected result format for {metric_group}")
                                            
                                    except Exception as e:
                                        print(f"Error evaluating {metric_group} for {current_cat}/{topic}/{current_system}/{current_model}: {e}")
                        
                        # Save updated results if any changes were made
                        if needs_update:
                            with open(results_path, "w", encoding="utf-8") as f:
                                json.dump(results, f, ensure_ascii=False, indent=4)
                            print(f"Updated results saved to {results_path}")
                        
                    except Exception as e:
                        print(f"Error processing {results_path}: {e}")

def supplement_domain_specific_scores(cat: str = None, model: str = None, system: str = None) -> None:
    """
    Check and supplement missing domain-specific scores for specific metrics.
    If any parameter is None, process all items for that parameter.
    
    Args:
        cat (str, optional): Category name (e.g., "cs"). If None, process all categories.
        model (str, optional): Model name (e.g., "gpt-4"). If None, process all models.
        system (str, optional): System name. If None, process all systems.
    """
    # Define metrics to check and their corresponding evaluation functions
    metric_functions = {
        "Outline": evaluate_outline_llm,
        "Reference": evaluate_reference_llm,
        "Content": evaluate_content_llm_simultaneous,  # Added content evaluation
    }
    
    # Define content metrics
    content_metrics = ["Coverage", "Structure", "Relevance", "Language", "Criticalness"]
    
    # Get categories to process
    base_dir = "surveys"
    if cat is not None:
        cats = [cat]
    else:
        cats = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for current_cat in cats:
        cat_dir = os.path.join(base_dir, current_cat)
        topics = [d for d in os.listdir(cat_dir) if os.path.isdir(os.path.join(cat_dir, d))]
        
        for topic in topics:
            topic_path = os.path.join(cat_dir, topic)
            systems = [d for d in os.listdir(topic_path) if os.path.isdir(os.path.join(topic_path, d))]
            
            # Filter systems if specified
            if system is not None:
                systems = [s for s in systems if s == system]
                if not systems:
                    print(f"System {system} not found in {topic}")
                    continue
            
            for current_system in systems:
                sys_path = os.path.join(topic_path, current_system)
                
                # Get all results files
                if model is not None:
                    results_files = [f"results_{model}.json"]
                else:
                    results_files = [f for f in os.listdir(sys_path) if f.startswith("results_") and f.endswith(".json")]
                
                for results_file in results_files:
                    results_path = os.path.join(sys_path, results_file)
                    current_model = results_file.replace("results_", "").replace(".json", "")
                    
                    if not os.path.exists(results_path):
                        continue
                        
                    try:
                        # Read current results
                        with open(results_path, "r", encoding="utf-8") as f:
                            results = json.load(f)
                        
                        needs_update = False
                        
                        # Check each metric group
                        for metric_group, eval_func in metric_functions.items():
                            if metric_group == "Content":
                                # Check if any content metric is missing
                                missing_metrics = [f"{metric}_domain" for metric in content_metrics 
                                                 if f"{metric}_domain" in results and results[f"{metric}_domain"] == 0]
                                if missing_metrics:
                                    print(f"Found missing domain-specific scores for {', '.join(missing_metrics)} in {current_cat}/{topic}/{current_system}/{current_model}")
                                    
                                    # Find the markdown file
                                    md_files = [f for f in os.listdir(sys_path) if f.lower().endswith(".md")]
                                    if not md_files:
                                        print(f"No markdown file found in {sys_path}")
                                        continue
                                        
                                    md_path = os.path.join(sys_path, md_files[0])
                                    
                                    # Re-evaluate all content metrics at once
                                    try:
                                        new_scores = eval_func(md_path, criteria_type="domain")
                                        if isinstance(new_scores, dict):
                                            for metric in content_metrics:
                                                domain_key = f"{metric}_domain"
                                                if domain_key in new_scores:
                                                    results[domain_key] = new_scores[domain_key]
                                                    needs_update = True
                                                    print(f"Updated {domain_key} score to {new_scores[domain_key]}")
                                    except Exception as e:
                                        print(f"Error evaluating content metrics for {current_cat}/{topic}/{current_system}/{current_model}: {e}")
                            else:
                                domain_key = f"{metric_group}_domain"
                                if domain_key in results and results[domain_key] == 0:
                                    print(f"Found missing domain-specific score for {domain_key} in {current_cat}/{topic}/{current_system}/{current_model}")
                                    
                                    # Find the markdown file
                                    md_files = [f for f in os.listdir(sys_path) if f.lower().endswith(".md")]
                                    if not md_files:
                                        print(f"No markdown file found in {sys_path}")
                                        continue
                                        
                                    md_path = os.path.join(sys_path, md_files[0])
                                    
                                    # Re-evaluate the metric with domain-specific criteria
                                    try:
                                        if metric_group == "Outline":
                                            # For outline, we need to use the outline.json path
                                            outline_json_path = os.path.join(sys_path, "outline.json")
                                            new_scores = eval_func(outline_json_path, criteria_type="domain")
                                        else:
                                            new_scores = eval_func(md_path, criteria_type="domain")
                                        
                                        # Update results
                                        if isinstance(new_scores, dict):
                                            if domain_key in new_scores:
                                                results[domain_key] = new_scores[domain_key]
                                                needs_update = True
                                                print(f"Updated {domain_key} score to {new_scores[domain_key]}")
                                        else:
                                            print(f"Unexpected result format for {domain_key}")
                                            
                                    except Exception as e:
                                        print(f"Error evaluating {domain_key} for {current_cat}/{topic}/{current_system}/{current_model}: {e}")
                        
                        # Save updated results if any changes were made
                        if needs_update:
                            with open(results_path, "w", encoding="utf-8") as f:
                                json.dump(results, f, ensure_ascii=False, indent=4)
                            print(f"Updated results saved to {results_path}")
                        
                    except Exception as e:
                        print(f"Error processing {results_path}: {e}")

def calculate_category_average_from_csv(cat: str) -> None:
    """
    Calculate average scores from category results CSV and save as a new CSV.
    Uses system+model as the primary key.
    
    Args:
        cat (str): Category name (e.g., "cs")
    """
    base_dir = os.path.join("surveys", cat)
    input_csv = os.path.join(base_dir, f"{cat}_results.csv")
    
    if not os.path.exists(input_csv):
        print(f"Input CSV file not found: {input_csv}")
        return
    
    try:
        # Read the results CSV
        df = pd.read_csv(input_csv)
        
        # Group by system and model, calculate mean for all numeric columns
        avg_df = df.groupby(['system', 'model']).mean(numeric_only=True).reset_index()
        
        # Round numeric columns to 4 decimal places
        numeric_cols = avg_df.select_dtypes(include=['float64', 'int64']).columns
        avg_df[numeric_cols] = avg_df[numeric_cols].round(4)
        
        # Save to new CSV
        output_csv = os.path.join(base_dir, f"{cat}_average.csv")
        avg_df.to_csv(output_csv, index=False)
        print(f"Category averages saved to {output_csv}")
        
    except Exception as e:
        print(f"Error processing CSV for category {cat}: {e}")

def aggregate_all_categories_average() -> None:
    """
    Aggregate all category average CSVs and calculate global averages.
    Creates two files in the surveys directory:
    1. all_categories_results.csv - Combined results from all categories
    2. global_average.csv - Global averages across all categories
    All numeric values are rounded to 2 decimal places.
    """
    base_dir = "surveys"
    all_cats_data = []
    
    # Get all category directories
    cats = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    # Collect data from each category
    for cat in cats:
        cat_avg_csv = os.path.join(base_dir, cat, f"{cat}_average.csv")
        if os.path.exists(cat_avg_csv):
            try:
                df = pd.read_csv(cat_avg_csv)
                # Round all numeric columns to 2 decimal places
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                df[numeric_cols] = df[numeric_cols].round(2)
                df['category'] = cat  # Add category column
                all_cats_data.append(df)
            except Exception as e:
                print(f"Error reading {cat_avg_csv}: {e}")
    
    if not all_cats_data:
        print("No category average data found")
        return
    
    try:
        # Combine all category data
        combined_df = pd.concat(all_cats_data, ignore_index=True)
        
        # Save combined results
        combined_csv = os.path.join(base_dir, "all_categories_results.csv")
        combined_df.to_csv(combined_csv, index=False)
        print(f"Combined results saved to {combined_csv}")
        
        # Calculate global averages
        avg_df = combined_df.groupby(['system', 'model']).mean(numeric_only=True).reset_index()
        
        # Round all numeric columns to 2 decimal places
        numeric_cols = avg_df.select_dtypes(include=['float64', 'int64']).columns
        avg_df[numeric_cols] = avg_df[numeric_cols].round(2)
        
        # Save global averages
        global_avg_csv = os.path.join(base_dir, "global_average.csv")
        avg_df.to_csv(global_avg_csv, index=False)
        print(f"Global averages saved to {global_avg_csv}")
        
    except Exception as e:
        print(f"Error processing global averages: {e}")

def calculate_all_scores(
    cats: list[str] = None,
    systems: list[str] = None,
    models: list[str] = None,
    num_workers: int = 1
) -> None:
    """
    Comprehensive function to calculate and aggregate all scores.
    This function performs the following steps in sequence:
    1. Calculate average scores for all specified categories/systems/models
    2. Supplement any missing scores
    3. Supplement any missing domain-specific scores
    4. Aggregate results to CSV files
    5. Calculate category averages
    6. Aggregate all categories into global results
    7. Reorganize results columns
    8. Convert to LaTeX format
    
    Args:
        cats (list[str], optional): List of categories to process. If None, process all categories.
        systems (list[str], optional): List of systems to process. If None, process all systems.
        models (list[str], optional): List of models to process. If None, process all models.
        num_workers (int, optional): Number of worker threads for parallel processing. Defaults to 1.
    """
    print("Starting comprehensive score calculation process...")
    
    # Step 1: Calculate average scores
    print("\nStep 1: Calculating average scores...")
    if cats is None:
        cats = [d for d in os.listdir("surveys") if os.path.isdir(os.path.join("surveys", d))]
    
    # Prepare tasks for parallel processing
    tasks = []
    for cat in cats:
        print(f"\nPreparing tasks for category: {cat}")
        if systems is None:
            # Get all systems from the category
            base_dir = os.path.join("surveys", cat)
            topics = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
            systems = set()
            for topic in topics:
                topic_path = os.path.join(base_dir, topic)
                systems.update([d for d in os.listdir(topic_path) if os.path.isdir(os.path.join(topic_path, d))])
            systems = list(systems)
        
        for system in systems:
            if models is None:
                # Get all models from the system's results files
                base_dir = os.path.join("surveys", cat)
                topics = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
                models = set()
                for topic in topics:
                    topic_path = os.path.join(base_dir, topic)
                    sys_path = os.path.join(topic_path, system)
                    if os.path.exists(sys_path):
                        results_files = [f for f in os.listdir(sys_path) if f.startswith("results_") and f.endswith(".json")]
                        models.update([f.replace("results_", "").replace(".json", "") for f in results_files])
                models = list(models)
            
            for model in models:
                tasks.append((cat, system, model))
    
    # Process tasks in parallel
    if num_workers > 1:
        print(f"\nProcessing {len(tasks)} tasks with {num_workers} workers...")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for cat, system, model in tasks:
                print(f"Submitting task for {cat}/{system}/{model}")
                futures.append(executor.submit(calculate_average_score, cat, system, model))
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in task: {e}")
    else:
        # Sequential processing
        for cat, system, model in tasks:
            print(f"\nProcessing {cat}/{system}/{model}")
            try:
                calculate_average_score(cat, system, model)
            except Exception as e:
                print(f"Error calculating average score for {cat}/{system}/{model}: {e}")
    
    # Step 2: Supplement missing scores
    print("\nStep 2: Supplementing missing scores...")
    if num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for cat, system, model in tasks:
                print(f"Submitting supplement task for {cat}/{system}/{model}")
                futures.append(executor.submit(supplement_missing_scores, cat, model, system))
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in supplement task: {e}")
    else:
        for cat, system, model in tasks:
            print(f"\nProcessing {cat}/{system}/{model}")
            try:
                supplement_missing_scores(cat, model, system)
            except Exception as e:
                print(f"Error supplementing scores for {cat}/{system}/{model}: {e}")

    # Step 3: Supplement missing domain-specific scores
    print("\nStep 3: Supplementing missing domain-specific scores...")
    if num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for cat, system, model in tasks:
                print(f"Submitting domain-specific supplement task for {cat}/{system}/{model}")
                futures.append(executor.submit(supplement_domain_specific_scores, cat, model, system))
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in domain-specific supplement task: {e}")
    else:
        for cat, system, model in tasks:
            print(f"\nProcessing {cat}/{system}/{model}")
            try:
                supplement_domain_specific_scores(cat, model, system)
            except Exception as e:
                print(f"Error supplementing domain-specific scores for {cat}/{system}/{model}: {e}")
    
    # Step 4: Aggregate results to CSV
    print("\nStep 4: Aggregating results to CSV...")
    if num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for cat in cats:
                print(f"Submitting aggregation task for {cat}")
                futures.append(executor.submit(aggregate_results_to_csv, cat))
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in aggregation task: {e}")
    else:
        for cat in cats:
            print(f"\nProcessing category: {cat}")
            try:
                aggregate_results_to_csv(cat)
            except Exception as e:
                print(f"Error aggregating results for {cat}: {e}")
    
    # Step 5: Calculate category averages
    print("\nStep 5: Calculating category averages...")
    if num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for cat in cats:
                print(f"Submitting average calculation task for {cat}")
                futures.append(executor.submit(calculate_category_average_from_csv, cat))
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in average calculation task: {e}")
    else:
        for cat in cats:
            print(f"\nProcessing category: {cat}")
            try:
                calculate_category_average_from_csv(cat)
            except Exception as e:
                print(f"Error calculating category average for {cat}: {e}")
    
    # Step 6: Aggregate all categories
    print("\nStep 6: Aggregating all categories...")
    try:
        aggregate_all_categories_average()
    except Exception as e:
        print(f"Error aggregating all categories: {e}")
    
    # Step 7: Reorganize results columns
    print("\nStep 7: Reorganizing results columns...")
    try:
        reorganize_results_columns(systems=systems, models=models)
    except Exception as e:
        print(f"Error reorganizing results columns: {e}")
    
    # Step 8: Convert to LaTeX format
    print("\nStep 8: Converting to LaTeX format...")
    try:
        convert_to_latex()
    except Exception as e:
        print(f"Error converting to LaTeX format: {e}")
    
    print("\nComprehensive score calculation completed!")

def reorganize_results_columns(systems: list[str] = None, models: list[str] = None) -> None:
    """
    Reorganize the columns in global_average.csv and all_categories_results.csv
    according to specified order and save to new files.
    Replace original values with ratio values for quantitative columns.
    
    Args:
        systems (list[str], optional): List of system names to process. If None, process all systems.
        models (list[str], optional): List of model names to process. If None, process all models.
    """
    base_dir = "surveys"
    
    # Define column orders
    global_columns = [
        "system", "model",
        "Outline", 
        "Outline_no", "Outline_density", 
        "Outline_coverage", "Outline_structure", 
        "Coverage", "Structure", "Relevance", "Language", "Criticalness",
        "Sentence_no", "Images_density", "Equations_density", "Tables_density", "Citations_density","Claim_density", 
        "Faithfulness",
        "Reference", 
        "Reference_no", "Reference_density", 
        "Reference_quality", 
    ]
    
    category_columns = [
        "system", "model", "category",
        "Outline_domain", 
        "Outline_no", "Outline_density", 
        "Outline_coverage", "Outline_structure", 
        "Coverage_domain", "Structure_domain", "Relevance_domain", "Language_domain", "Criticalness_domain",
        "Sentence_no", "Images_density", "Equations_density", "Tables_density", "Citations_density","Claim_density", 
        "Faithfulness",
        "Reference_domain", 
        "Reference_no", "Reference_density", 
        "Reference_quality", 
    ]

    relative_quantitative_columns = [
        "Outline_no","Outline_density", 
        "Sentence_no", "Images_density", "Equations_density", "Tables_density", "Citations_density", "Claim_density",
        "Reference_no", "Reference_density",
    ]
    
    # Process global_average.csv
    global_avg_path = os.path.join(base_dir, "global_average.csv")
    if os.path.exists(global_avg_path):
        try:
            df = pd.read_csv(global_avg_path)
            
            # Filter by systems and models if specified
            if systems is not None:
                df = df[df['system'].isin(systems)]
            if models is not None:
                df = df[df['model'].isin(models)]
            
            if not df.empty:
                # Create new DataFrame with only required columns
                new_df = pd.DataFrame()
                for col in global_columns:
                    if col in df.columns:
                        new_df[col] = df[col]
                    else:
                        new_df[col] = ""
                
                # Calculate and replace with relative ratios for global average
                for col in relative_quantitative_columns:
                    if col in new_df.columns:
                        # Get pdfs values for each model
                        pdfs_values = {}
                        for model in new_df['model'].unique():
                            pdfs_row = new_df[(new_df['system'] == 'pdfs') & (new_df['model'] == model)]
                            if not pdfs_row.empty:
                                pdfs_values[model] = pdfs_row[col].iloc[0]
                        
                        # Replace values with ratios
                        new_df[col] = new_df.apply(
                            lambda row: 1.00 if row['system'] == 'pdfs' 
                            else round(float(row[col]) / float(pdfs_values[row['model']]), 2) 
                            if row[col] != "" and pdfs_values.get(row['model']) != "" 
                            else "", 
                            axis=1
                        )
                
                # Format numeric columns to 2 decimal places
                numeric_cols = new_df.select_dtypes(include=['float64', 'int64']).columns
                new_df[numeric_cols] = new_df[numeric_cols].round(2).applymap(lambda x: f"{x:.2f}" if pd.notnull(x) else "")
                
                # Save to new file
                output_path = os.path.join(base_dir, "global_average_reorganized.csv")
                new_df.to_csv(output_path, index=False)
                print(f"Reorganized global averages saved to {output_path}")
        except Exception as e:
            print(f"Error processing global_average.csv: {e}")
    
    # Process all_categories_results.csv
    all_cats_path = os.path.join(base_dir, "all_categories_results.csv")
    if os.path.exists(all_cats_path):
        try:
            df = pd.read_csv(all_cats_path)
            
            # Filter by systems and models if specified
            if systems is not None:
                df = df[df['system'].isin(systems)]
            if models is not None:
                df = df[df['model'].isin(models)]
            
            if not df.empty:
                # Create new DataFrame with only required columns
                new_df = pd.DataFrame()
                for col in category_columns:
                    if col in df.columns:
                        new_df[col] = df[col]
                    else:
                        new_df[col] = ""
                
                # Calculate and replace with relative ratios for category results
                for col in relative_quantitative_columns:
                    if col in new_df.columns:
                        # Get pdfs values for each model and category
                        pdfs_values = {}
                        for model in new_df['model'].unique():
                            for category in new_df['category'].unique():
                                pdfs_row = new_df[(new_df['system'] == 'pdfs') & 
                                                (new_df['model'] == model) & 
                                                (new_df['category'] == category)]
                                if not pdfs_row.empty:
                                    pdfs_values[(model, category)] = pdfs_row[col].iloc[0]
                        
                        # Replace values with ratios
                        new_df[col] = new_df.apply(
                            lambda row: 1.00 if row['system'] == 'pdfs'
                            else round(float(row[col]) / float(pdfs_values.get((row['model'], row['category']), "")), 2)
                            if row[col] != "" and pdfs_values.get((row['model'], row['category'])) != ""
                            else "",
                            axis=1
                        )
                
                # Format numeric columns to 2 decimal places
                numeric_cols = new_df.select_dtypes(include=['float64', 'int64']).columns
                new_df[numeric_cols] = new_df[numeric_cols].round(2).applymap(lambda x: f"{x:.2f}" if pd.notnull(x) else "")
                
                # Save to new file
                output_path = os.path.join(base_dir, "all_categories_results_reorganized.csv")
                new_df.to_csv(output_path, index=False)
                print(f"Reorganized category results saved to {output_path}")
        except Exception as e:
            print(f"Error processing all_categories_results.csv: {e}")

def convert_to_latex() -> None:
    """
    Convert reorganized CSV files to LaTeX format.
    Creates two files:
    1. global_average.tex - LaTeX format for global averages
    2. category_average.tex - LaTeX format for category averages
    Uses the same column order as reorganize_results_columns.
    """
    base_dir = "surveys"
    
    # Define column orders (same as in reorganize_results_columns)
    global_columns = [
        "system", "model",
        "Outline", 
        # "Outline_no", "Outline_density", 
        "Outline_coverage", "Outline_structure", 
        "Coverage", "Structure", "Relevance", "Language", "Criticalness",
        # "Sentence_no", "Images_density", "Equations_density", "Tables_density", "Citations_density","Claim_density", 
        "Faithfulness",
        "Reference", 
        # "Reference_no", "Reference_density", 
        "Reference_quality", 
    ]
    
    category_columns = [
        "system", "model", "category",
        "Outline_domain", 
        # "Outline_no", "Outline_density", 
        "Outline_coverage", "Outline_structure", 
        "Coverage_domain", "Structure_domain", "Relevance_domain", "Language_domain", "Criticalness_domain",
        # "Sentence_no", "Images_density", "Equations_density", "Tables_density", "Citations_density","Claim_density", 
        "Faithfulness",
        "Reference_domain", 
        # "Reference_no", "Reference_density", 
        "Reference_quality", 
    ]
    
    # Define non-numeric columns
    non_numeric_columns = {'system', 'model', 'category'}
    
    # Process global average
    global_avg_path = os.path.join(base_dir, "global_average_reorganized.csv")
    if os.path.exists(global_avg_path):
        try:
            df = pd.read_csv(global_avg_path)
            latex_lines = []
            current_system = None
            
            for _, row in df.iterrows():
                system = row['system']
                model = row['model']
                
                # Start new system group if system changes
                if system != current_system:
                    if current_system is not None:
                        latex_lines.append("")  # Add empty line between systems
                    latex_lines.append(f"\\textbf{{{system}}}")
                    current_system = system
                
                # Format the line using global_columns order
                values = []
                for col in global_columns[2:]:  # Skip system and model columns
                    val = row[col]
                    if pd.isna(val) or val == "":
                        values.append("-")  # Replace empty values with dash
                    else:
                        # Skip formatting for non-numeric columns
                        if col in non_numeric_columns:
                            values.append(str(val))
                        else:
                            values.append(f"{float(val):.2f}")
                
                # Create the line
                line = f"& \\textit{{{model}}} & {' & '.join(values)}\\\\"
                latex_lines.append(line)
            
            # Save to file
            output_path = os.path.join(base_dir, "global_average.tex")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(latex_lines))
            print(f"Global average LaTeX saved to {output_path}")
            
        except Exception as e:
            print(f"Error processing global average: {e}")
    
    # Process category average
    category_avg_path = os.path.join(base_dir, "all_categories_results_reorganized.csv")
    if os.path.exists(category_avg_path):
        try:
            df = pd.read_csv(category_avg_path)
            latex_lines = []
            current_system = None
            
            # Group by system and sort by category
            for system in sorted(df['system'].unique()):
                system_df = df[df['system'] == system]
                
                # Add system header and hline if not first system
                if current_system is not None:
                    latex_lines.append("\\hline")  # Add hline between systems
                latex_lines.append(f"\\textbf{{{system}}}")
                current_system = system
                
                # Process each category for this system
                for _, row in system_df.iterrows():
                    category = row['category']  # Using category column instead of model
                    
                    # Format the line using category_columns order
                    values = []
                    for col in category_columns[3:]:  # Skip system, model, and category columns
                        val = row[col]
                        if pd.isna(val) or val == "":
                            values.append("-")  # Replace empty values with dash
                        else:
                            # Skip formatting for non-numeric columns
                            if col in non_numeric_columns:
                                values.append(str(val))
                            else:
                                values.append(f"{float(val):.2f}")
                    
                    # Create the line
                    line = f"& \\textit{{{category}}} & {' & '.join(values)}\\\\"
                    latex_lines.append(line)
                
                # Add Area-aware ASG-Bench row for this system
                avg_values = []
                for col in category_columns[3:]:  # Skip system, model, and category columns
                    # Calculate average for numeric columns
                    if col not in non_numeric_columns:
                        valid_values = [float(val) for val in system_df[col] if val != "" and not pd.isna(val)]
                        if valid_values:
                            avg = sum(valid_values) / len(valid_values)
                            avg_values.append(f"{avg:.2f}")
                        else:
                            avg_values.append("-")  # Replace empty values with dash
                    else:
                        avg_values.append("-")  # Replace empty values with dash
                
                # Add the Area-aware ASG-Bench line
                line = f"& Area-aware ASG-Bench & {' & '.join(avg_values)}\\\\"
                latex_lines.append(line)
            
            # Save to file
            output_path = os.path.join(base_dir, "category_average.tex")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(latex_lines))
            print(f"Category average LaTeX saved to {output_path}")
            
        except Exception as e:
            print(f"Error processing category average: {e}")

    # Generate new comparison table
    print("\nGenerating new comparison table...")
    new_latex_lines = []
    
    # Read global average data
    global_avg_path = os.path.join(base_dir, "global_average_reorganized.csv")
    if os.path.exists(global_avg_path):
        global_df = pd.read_csv(global_avg_path)
        
        # Read category average data
        cat_avg_path = os.path.join(base_dir, "all_categories_results_reorganized.csv")
        if os.path.exists(cat_avg_path):
            cat_df = pd.read_csv(cat_avg_path)
            
            # Process each system
            for system in global_df['system'].unique():
                # Get global average row
                global_row = global_df[global_df['system'] == system].iloc[0]
                
                # Calculate domain averages for this system
                system_cat_df = cat_df[cat_df['system'] == system]
                domain_avgs = {}
                for col in ['Outline_domain', 'Coverage_domain', 'Structure_domain', 
                          'Relevance_domain', 'Language_domain', 'Criticalness_domain', 
                          'Reference_domain']:
                    valid_values = [float(val) for val in system_cat_df[col] if val != "" and not pd.isna(val)]
                    if valid_values:
                        domain_avgs[col] = sum(valid_values) / len(valid_values)
                    else:
                        domain_avgs[col] = 0.0
                
                # Format global average row
                global_values = [
                    f"{global_row['Outline']:.2f}",
                    f"{global_row['Coverage']:.2f}",
                    f"{global_row['Structure']:.2f}",
                    f"{global_row['Relevance']:.2f}",
                    f"{global_row['Language']:.2f}",
                    f"{global_row['Criticalness']:.2f}",
                    f"{global_row['Reference']:.2f}"
                ]
                
                # Format domain average row
                cat_values = [
                    f"{domain_avgs['Outline_domain']:.2f}",
                    f"{domain_avgs['Coverage_domain']:.2f}",
                    f"{domain_avgs['Structure_domain']:.2f}",
                    f"{domain_avgs['Relevance_domain']:.2f}",
                    f"{domain_avgs['Language_domain']:.2f}",
                    f"{domain_avgs['Criticalness_domain']:.2f}",
                    f"{domain_avgs['Reference_domain']:.2f}"
                ]
                
                # Add rows to LaTeX table
                new_latex_lines.extend([
                    f"\\textbf{{{system}}} & \\textit{{ASGBench}} & {global_values[0]} & {global_values[1]} & {global_values[2]} & {global_values[3]} & {global_values[4]} & {global_values[5]} & {global_values[6]} \\\\",
                    f" & \\textit{{AA-ASGBench}} & {cat_values[0]} & {cat_values[1]} & {cat_values[2]} & {cat_values[3]} & {cat_values[4]} & {cat_values[5]} & {cat_values[6]} \\\\",
                    "\\hline"
                ])
    
    # Save new LaTeX table
    new_latex_path = os.path.join(base_dir, "comparison_table.tex")
    with open(new_latex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(new_latex_lines))
    
    print(f"New comparison table saved to {new_latex_path}")

def convert_to_latex_similarity() -> None:
    """
    Convert reorganized CSV files to LaTeX format with similarity scores.
    Generates data rows for the LaTeX table with density metrics and similarity scores.
    """
    base_dir = "surveys"
    input_path = os.path.join(base_dir, "global_average_reorganized.csv")
    output_path = os.path.join(base_dir, "experiment_similarity.tex")
    
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        return
        
    try:
        df = pd.read_csv(input_path)
        density_cols = [col for col in df.columns if col.endswith('_density')]
        
        # 计算similarity score
        pdfs_rows = df[df['system'] == 'pdfs'].set_index('model')
        
        def compute_l2(row):
            if row['system'] == 'pdfs':
                return 1.0
            model = row['model']
            if model not in pdfs_rows.index:
                return np.nan
            pdfs_density = pdfs_rows.loc[model, density_cols].astype(float)
            rel = row[density_cols].astype(float)
            l2 = np.sqrt(np.sum((rel - 1) ** 2))
            score = 1 / (1 + l2)
            return score
        
        df['Similarity_score'] = df.apply(compute_l2, axis=1)
        
        # 准备LaTeX表格内容
        systems = ['AutoSurvey', 'InteractiveSurvey', 'LLMxMapReduce', 'SurveyForge', 'SurveyX', 'pdfs']
        latex_lines = []
        
        # 为每个系统添加数据行
        for system in systems:
            system_data = df[df['system'] == system]
            if not system_data.empty:
                row_data = system_data.iloc[0]
                values = []
                for col in density_cols:
                    values.append(f"{row_data[col]:.2f}")
                similarity = f"{row_data['Similarity_score']:.2f}"
                
                latex_row = f"    \\textbf{{{system}}} & {' & '.join(values)} & {similarity}\\\\"
                latex_lines.append(latex_row)
                latex_lines.append("    \\hline")
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(latex_lines))
        print(f"Similarity LaTeX saved to {output_path}")
        
    except Exception as e:
        print(f"Error processing similarity LaTeX: {e}")

def convert_to_latex_comparison(csv_paths: list[str]) -> None:
    """
    Convert multiple CSV files to LaTeX format with mean and standard deviation.
    Each CSV file should have the same format as global_comparison_results_*.csv.
    
    Parameters:
    -----------
    csv_paths : list[str]
        List of paths to CSV files containing comparison results
    """
    # Read all CSV files
    dfs = []
    for path in csv_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Convert Win Rate from percentage string to float
            df['Win Rate'] = df['Win Rate'].str.rstrip('%').astype('float') / 100
            dfs.append(df)
        else:
            print(f"Warning: File not found: {path}")
    
    if not dfs:
        print("No valid CSV files found")
        return
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Group by System and Metric to calculate mean and std
    grouped = combined_df.groupby(['System', 'Metric'])['Win Rate'].agg(['mean', 'std'])
    
    # Create a dictionary to store results
    results = {}
    for (system, metric), stats in grouped.iterrows():
        if system not in results:
            results[system] = {}
        results[system][metric] = {
            'mean': stats['mean'],
            'std': stats['std']
        }
    
    # Generate LaTeX table rows
    latex_rows = []
    systems = ['AutoSurvey', 'InteractiveSurvey', 'LLMxMapReduce', 'SurveyForge', 'SurveyX']
    metrics = ['outline', 'content', 'reference']
    
    for system in systems:
        if system in results:
            row = [f"\\textbf{{{system}}}"]
            for metric in metrics:
                if metric in results[system]:
                    mean = results[system][metric]['mean']
                    std = results[system][metric]['std']
                    # Format as mean ± std with 2 decimal places
                    cell = f"${mean:.2f}_{{\\pm {std:.2f}}}$"
                else:
                    cell = "-"
                row.append(cell)
            latex_rows.append(" & ".join(row) + " \\\\")
            latex_rows.append("\\hline")
    
    # Print the LaTeX table rows
    print("\n".join(latex_rows))
    
    # Optionally save to file
    output_path = os.path.join("surveys", "comparison_table_rows.tex")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_rows))
    print(f"\nLaTeX table rows saved to {output_path}")

if __name__ == "__main__":
    # 测试代码
    # md_path = "surveys\cs\Optimization Techniques for Transformer Inference\pdfs/2307.07982.md"  # 替换为实际的文件路径
    # md_path = "surveys\cs\Agent-based Modeling and Simulation using Large Language Models\AutoSurvey\Agent-based Modeling and Simulation using Large Language Models.md"
    # json_path = os.path.join(os.path.dirname(md_path), "outline.json")
    # evaluate_outline(md_path)
    # evaluate_content_informativeness(md_path)
    # evaluate_content(md_path)
    # evaluate_reference(md_path)
    # print(evaluate_outline_coverage(json_path))
    # batch_evaluate_by_cat(["cs"])
    # calculate_average_score("cs", "vanilla_outline", "qwen-plus-2025-04-28")
    # calculate_average_score_by_cat("econ")
    # clear_scores("cs", "AutoSurvey")
    # batch_evaluate_by_system(["vanilla"], "qwen-plus-2025-04-28", num_workers=4)
    # clear_all_scores()
    # evaluate("surveys/cs/3D Gaussian Splatting Techniques/vanilla_outline/3D Gaussian Splatting Techniques.md")
    # evaluate("surveys/cs/3D Gaussian Splatting Techniques/vanilla/3D Gaussian Splatting Techniques.md")
    # print(evaluate_outline_coverage("surveys/cs/3D Gaussian Splatting Techniques/vanilla/outline.json"))
    # evaluate_reference("surveys\cs/3D Gaussian Splatting Techniques\AutoSurvey/3D Gaussian Splatting Techniques.md")
    # evaluate("surveys/cs/3D Gaussian Splatting Techniques/AutoSurvey/3D Gaussian Splatting Techniques.md")
    # surveys\cs\3D Gaussian Splatting Techniques\InteractiveSurvey
    # evaluate("surveys/cs/3D Gaussian Splatting Techniques/InteractiveSurvey/survey_3D Gaussian Splatting Techniques.md")
    # batch_evaluate_by_system(["AutoSurvey", "InteractiveSurvey", "LLMxMapReduce", "SurveyForge", "SurveyX", "pdfs"], "qwen-plus-latest", num_workers=4)
    # batch_evaluate_by_system(["AutoSurvey", "InteractiveSurvey", "LLMxMapReduce", "SurveyForge", "SurveyX", "pdfs"], "qwen-plus-latest", num_workers=4, criteria_type="domain")
    # evaluate("surveys/cs/3D Gaussian Splatting Techniques/AutoSurvey/3D Gaussian Splatting Techniques.md")
    # print(evaluate_content_informativeness("surveys/cs/3D Gaussian Splatting Techniques/AutoSurvey/3D Gaussian Splatting Techniques.md"))
    # print(evaluate_content_llm_simultaneous("surveys/cs/3D Gaussian Splatting Techniques/AutoSurvey/3D Gaussian Splatting Techniques.md"))
    # calculate_all_cats_average_scores()
    # aggregate_results_to_csv("cs")
    # calculate_category_average_from_csv("cs")
    # aggregate_all_categories_average()
    # calculate_all_scores(models=["deepseek-r1"])
    # batch_evaluate_by_system(["AutoSurvey", "InteractiveSurvey", "LLMxMapReduce", "pdfs", "SurveyForge", "SurveyX"], do_reference=False, do_content=False, model="qwen-plus-latest", num_workers=4)
    # calculate_all_scores(models=["qwen-plus-latest"])
    # aggregate_comparison_results(root_dir="surveys", systems=["AutoSurvey", "InteractiveSurvey", "LLMxMapReduce", "SurveyForge", "SurveyX"], metrics=["outline", "content", "reference"], modelname="gpt-4o")
    # evaluate_pairs("surveys/cs/3D Gaussian Splatting Techniques", "AutoSurvey", ["Reference"])
    # print(evaluate_outline_density("surveys/physics/Modeling Thermodynamic Properties of Deep Eutectic Solvents/pdfs/2303.17159.md"))
    # delete_system(["vanilla", "vanilla_outline"])
    # evaluate_topic_ranking("surveys/cs/3D Gaussian Splatting Techniques", ["Outline", "Content", "Reference"])
    # print(evaluate_outline_coverage("surveys/cs/3D Gaussian Splatting Techniques/InteractiveSurvey/outline.json"))
    # print(refine_ref_lists("surveys/cs/3D Gaussian Splatting Techniques/InteractiveSurvey/survey_3D Gaussian Splatting Techniques.md", judge=judge))
    # clear_all_scores(model="qwen-plus-latest", target="Outline_coverage")
    # convert_to_latex_similarity()
    csv_paths = [
    "surveys\global_comparison_results_qwen-plus-latest.csv",
    "surveys\global_comparison_results_qwen2_5.csv",
    "surveys\global_comparison_results.csv"
    ]
    convert_to_latex_comparison(csv_paths)  


