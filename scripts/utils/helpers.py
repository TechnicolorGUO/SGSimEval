"""
Helper functions for SGSimEval.
Contains utility functions for file operations, text processing, and API interactions.
"""

# Standard library imports
import ast
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Third-party imports
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from openai import OpenAI
from dotenv import load_dotenv

# Local imports
from .prompts import OUTLINE_REFINE_PROMPT
from .config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    LOG_FILE,
    LOG_FORMAT,
    LOG_LEVEL
)

# Initialize environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    filename=LOG_FILE,
    level=LOG_LEVEL,
    format=LOG_FORMAT
)

def getClient() -> OpenAI:
    """
    Initialize and return an OpenAI client using environment variables.
    
    Returns:
        OpenAI: Configured OpenAI client instance
        
    Raises:
        EnvironmentError: If required environment variables are not set
    """
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    openai_api_base = os.environ.get("OPENAI_API_BASE")
    
    if not openai_api_key or not openai_api_base:
        raise EnvironmentError("Missing required environment variables: OPENAI_API_KEY or OPENAI_API_BASE")

    return OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base
    )

def generateResponse(
    client: OpenAI,
    prompt: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE
) -> str:
    """
    Generate a response from the OpenAI model using streaming.
    
    Args:
        client (OpenAI): OpenAI client instance
        prompt (str): Input prompt for the model
        max_tokens (int, optional): Maximum tokens in response. Defaults to DEFAULT_MAX_TOKENS.
        temperature (float, optional): Response randomness. Defaults to DEFAULT_TEMPERATURE.
    
    Returns:
        str: Generated response text
        
    Raises:
        Exception: If API call fails
    """
    try:
        # Log request information
        request_info = {
            'timestamp': datetime.now().isoformat(),
            'model': os.environ.get("MODEL"),
            'max_tokens': max_tokens,
            'temperature': temperature,
            'prompt_length': len(prompt)
        }
        logging.info(f"Request: {json.dumps(request_info)}")
        
        chat_response = client.chat.completions.create(
            model=os.environ.get("MODEL"),
            max_tokens=max_tokens,
            temperature=temperature,
            stop="<|im_end|>",
            stream=True,
            messages=[{"role": "user", "content": prompt}]
        )
        
        text = ""
        for chunk in chat_response:
            if chunk.choices[0].delta.content:
                text += chunk.choices[0].delta.content
        
        # Log response information
        response_info = {
            'timestamp': datetime.now().isoformat(),
            'response_length': len(text),
            'response_preview': text[:100] + '...' if len(text) > 100 else text
        }
        logging.info(f"Response: {json.dumps(response_info)}")
        
        return text
        
    except Exception as e:
        logging.error(f"Error generating response: {str(e)}")
        raise

def robust_json_parse(text: str) -> Dict:
    """
    Parse JSON from text, handling common formatting issues.
    
    Args:
        text (str): Text containing JSON to parse
        
    Returns:
        Dict: Parsed JSON object or empty dict if parsing fails
        
    Raises:
        ValueError: If no JSON object can be found in text
    """
    try:
        # Try direct parsing first
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            # Try to extract JSON from text
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                raise ValueError("No JSON object found in text")
        except Exception as e:
            raise ValueError(f"Failed to parse JSON: {e}")

def robust_list_parse(text: str) -> List:
    """
    Parse a string into a Python list, handling various formats.
    
    Args:
        text (str): Text to parse as a list
        
    Returns:
        List: Parsed list
        
    Raises:
        ValueError: If parsing fails
    """
    def extract_most_likely_list(text: str) -> str:
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1 and end > start:
            return text[start:end+1]
        return text

    cleaned = text.strip()
    cleaned = extract_most_likely_list(cleaned)

    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return result
    except Exception:
        pass

    try:
        result = ast.literal_eval(cleaned)
        if isinstance(result, list):
            return result
    except Exception:
        pass

    raise ValueError("Could not parse response as a list")

def read_md(file_path: str) -> str:
    """
    Read markdown file content.
    
    Args:
        file_path (str): Path to markdown file
        
    Returns:
        str: File contents
        
    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If file can't be read
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logging.error(f"Error reading markdown file {file_path}: {str(e)}")
        raise

def count_sentences(text: str) -> int:
    """
    Count number of sentences in text.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        int: Number of sentences
    """
    return len([s for s in text.split('.') if s.strip()])

def count_md_features(text: str) -> Dict[str, int]:
    """
    Count markdown features in text.
    
    Args:
        text (str): Markdown text to analyze
        
    Returns:
        Dict[str, int]: Counts of different markdown features
    """
    return {
        'headings': len(re.findall(r'^#+\s', text, re.MULTILINE)),
        'lists': len(re.findall(r'^[\*\-]\s', text, re.MULTILINE)),
        'code_blocks': len(re.findall(r'```', text)),
        'links': len(re.findall(r'\[.*?\]\(.*?\)', text)),
        'images': len(re.findall(r'!\[.*?\]\(.*?\)', text))
    }

def extract_topic_from_path(path: str) -> str:
    """
    Extract topic name from file path.
    
    Args:
        path (str): File path
        
    Returns:
        str: Extracted topic name
    """
    parts = path.split(os.sep)
    if len(parts) > 2:
        return parts[-2]  # Assuming format: .../topic/filename
    return os.path.splitext(os.path.basename(path))[0]

def build_outline_tree_from_levels(outline_list: List[List[Union[int, str]]]) -> List[List[str]]:
    """
    Build outline tree from level-based structure.
    
    Args:
        outline_list (List[List[Union[int, str]]]): List of [level, title] pairs
        
    Returns:
        List[List[str]]: Tree structure of outline
    """
    tree = []
    current_path = []
    
    for level, title in outline_list:
        while len(current_path) >= level:
            current_path.pop()
        current_path.append(title)
        tree.append(current_path.copy())
    
    return tree

def extract_and_save_outline_from_md(md_path: str, output_path: Optional[str] = None) -> List[List]:
    """
    Extract outline from markdown and save to JSON.
    
    Args:
        md_path (str): Path to markdown file
        output_path (Optional[str]): Path to save JSON output
        
    Returns:
        List[List]: Extracted outline structure
        
    Raises:
        FileNotFoundError: If markdown file doesn't exist
        IOError: If file operations fail
    """
    content = read_md(md_path)
    outline = []
    
    for line in content.split('\n'):
        if line.startswith('#'):
            level = len(re.match(r'^#+', line).group())
            title = line[level:].strip()
            outline.append([level, title])
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(outline, f, ensure_ascii=False, indent=2)
    
    return outline

def refine_outline_if_single_level(outline: List[List]) -> List[List]:
    """
    Refine outline if it has only one level.
    
    Args:
        outline (List[List]): Original outline structure
        
    Returns:
        List[List]: Refined outline structure
    """
    if not outline:
        return outline
    
    levels = [item[0] for item in outline]
    if len(set(levels)) == 1:
        # Add a title level
        return [[1, "Title"]] + [[2, item[1]] for item in outline]
    return outline

def fill_single_criterion_prompt(
    prompt_template: str,
    content: str,
    topic: str,
    criterion: str,
    criteria_name: str,
    type: str,
    references: Optional[str] = None
) -> str:
    """
    Fill a prompt template with content and criteria.
    
    Args:
        prompt_template (str): Template string
        content (str): Content to evaluate
        topic (str): Topic of the content
        criterion (str): Evaluation criterion
        criteria_name (str): Name of criteria set
        type (str): Type of evaluation
        references (Optional[str]): Reference content if needed
        
    Returns:
        str: Filled prompt
    """
    prompt = prompt_template.format(
        content=content,
        topic=topic,
        criterion=criterion,
        criteria_name=criteria_name,
        type=type
    )
    if references:
        prompt = prompt.format(references=references)
    return prompt

# --------------File I/O and Content Processing-------------

def count_md_files(surveys_root: str = "surveys") -> dict[str, int]:
    """
    Count the number of markdown files in the surveys directory for each system.
    
    Args:
        surveys_root (str, optional): Root directory of surveys. Defaults to "surveys".
        
    Returns:
        dict[str, int]: Dictionary mapping system names to their markdown file counts
    """
    sys_dict: dict[str, int] = {}
    for cat in os.listdir(surveys_root):
        cat_path = os.path.join(surveys_root, cat)
        if not os.path.isdir(cat_path):
            continue
        for topic in os.listdir(cat_path):
            topic_path = os.path.join(cat_path, topic)
            if not os.path.isdir(topic_path):
                continue
            for system in os.listdir(topic_path):
                md_files = os.listdir(os.path.join(topic_path, system))
                for file in md_files:
                    if file.lower().endswith(".md"):
                        sys_dict[system] = sys_dict.get(system, 0) + 1
    for system, count in sys_dict.items():
        print(f"{system}: {count}")
    return sys_dict

# --------------PDF Processing-------------

def download_arxiv_pdf(arxiv_id: str, save_dir: str) -> str:
    """
    Download an arXiv paper PDF to the specified directory.
    
    Args:
        arxiv_id (str): arXiv paper ID (e.g., '2301.00001')
        save_dir (str): Target directory path (e.g., './pdfs')
        
    Returns:
        str: Path to the saved PDF file
        
    Raises:
        Exception: If download fails
    """
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{arxiv_id}.pdf")

    if os.path.exists(file_path):
        print(f"PDF already exists: {file_path}")
        return file_path

    print(f"Downloading {pdf_url} ...")
    try:
        resp = requests.get(pdf_url, stream=True, timeout=20)
        resp.raise_for_status()
        with open(file_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Saved: {file_path}")
        return file_path
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        print(f"Failed to download {arxiv_id}: {e}")
        raise

def pdf2md(pdf_path: str, output_dir: str) -> tuple[str | None, str | None]:
    """
    Convert a PDF file to markdown using magic-pdf.
    Moves the generated markdown to the output directory root.
    Removes the intermediate directory after conversion.
    
    Args:
        pdf_path (str): Path to the PDF file
        output_dir (str): Output directory path
        
    Returns:
        tuple[str | None, str | None]: (path to new markdown file, markdown content) or (None, None) if conversion fails
    """
    cmd = ["magic-pdf"]
    if pdf_path:
        cmd += ["-p", pdf_path]
    cmd += ["-o", output_dir]

    try:
        print("Running command:", " ".join(shlex.quote(str(x)) for x in cmd))
        subprocess.run(cmd, check=True)
        print("Conversion finished!")
    except subprocess.CalledProcessError as e:
        print("Error running magic-pdf:", e)
        return None, None

    pdf_filename = os.path.basename(pdf_path)
    pdf_stem = pdf_filename.replace(".pdf", "")
    md_orig_path = os.path.join(output_dir, pdf_stem, "auto", pdf_stem + ".md")
    md_new_path = os.path.join(output_dir, pdf_stem + ".md")
    pdf_stem_dir = os.path.join(output_dir, pdf_stem)

    if not os.path.exists(md_orig_path):
        print(f"Cannot find md file at {md_orig_path}")
        return None, None

    shutil.move(md_orig_path, md_new_path)

    try:
        shutil.rmtree(pdf_stem_dir)
        print(f"Removed intermediate directory: {pdf_stem_dir}")
    except Exception as e:
        print(f"Failed to remove intermediate directory {pdf_stem_dir}: {e}")

    with open(md_new_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    return md_new_path, md_content

def batch_pdf2md_in_surveys(surveys_root: str = "surveys") -> None:
    """
    Batch convert all PDF files in surveys/<cat>/<topic>/<system> to markdown.
    Skips conversion if corresponding markdown file already exists.
    
    Args:
        surveys_root (str, optional): Root directory of surveys. Defaults to "surveys".
    """
    total_pdf = 0
    converted = 0
    skipped = 0
    failed = 0

    for cat in os.listdir(surveys_root):
        cat_path = os.path.join(surveys_root, cat)
        if not os.path.isdir(cat_path):
            continue
        for topic in os.listdir(cat_path):
            topic_path = os.path.join(cat_path, topic)
            if not os.path.isdir(topic_path):
                continue
            for system in os.listdir(topic_path):
                system_path = os.path.join(topic_path, system)
                if not os.path.isdir(system_path):
                    continue
                for file in os.listdir(system_path):
                    if file.lower().endswith(".pdf"):
                        total_pdf += 1
                        pdf_path = os.path.join(system_path, file)
                        md_name = os.path.splitext(file)[0] + ".md"
                        md_path = os.path.join(system_path, md_name)
                        if os.path.exists(md_path):
                            print(f"Skip (md exists): {md_path}")
                            skipped += 1
                            continue
                        print(f"Converting: {pdf_path}")
                        md_new_path, md_content = pdf2md(pdf_path, system_path)
                        if md_new_path:
                            print(f"Converted: {md_new_path}")
                            converted += 1
                        else:
                            print(f"Failed: {pdf_path}")
                            failed += 1

    print("\nBatch Summary:")
    print(f"Total PDF files: {total_pdf}")
    print(f"Converted: {converted}")
    print(f"Skipped (md exists): {skipped}")
    print(f"Failed: {failed}")

# --------------Markdown Processing-------------

def extract_references_from_md(md_path: str) -> list[str]:
    """
    Read and return reference entries from references.json in the same directory as md_path.
    
    Args:
        md_path (str): Path to markdown file
        
    Returns:
        list[str]: List of reference entries, empty list if file not found or invalid
    """
    dir_path = os.path.dirname(md_path)
    json_path = os.path.join(dir_path, "references.json")
    if not os.path.isfile(json_path):
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        try:
            refs = json.load(f)
            refs = [ref.strip() for ref in refs if ref.strip()]
            return refs
        except Exception:
            return []

# --------------Text Analysis and Statistics-------------

def count_md_features(md_content: str) -> dict[str, int]:
    """
    Count images, equations, tables, and sentences in markdown content.
    
    Args:
        md_content (str): Markdown text
        
    Returns:
        dict[str, int]: Counts of {'images': int, 'equations': int, 'tables': int, 'sentences': int}
    """
    img_md = re.findall(r'!\[.*?\]\(.*?\)', md_content)
    img_html = re.findall(r'<img [^>]*src=[\'"].*?[\'"][^>]*>', md_content, re.IGNORECASE)
    img_html2 = re.findall(r'<img [^>]*>', md_content, re.IGNORECASE)
    image_count = len(set(img_md + img_html + img_html2))

    block_eq = re.findall(r'\$\$.*?\$\$', md_content, re.DOTALL)
    block_eq += re.findall(r'\\\[.*?\\\]', md_content, re.DOTALL)
    block_eq += re.findall(r'\\begin\{.*?\}.*?\\end\{.*?\}', md_content, re.DOTALL)
    inline_eq = re.findall(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)', md_content)
    equation_count = len(block_eq) + len(inline_eq)

    md_tables = re.findall(
        r'(?:\|[^\n]*\n)+\|[\s\-:|]+\|(?:\n\|[^\n]*)*', md_content)
    html_tables = re.findall(r'<table[\s\S]*?</table>', md_content, re.IGNORECASE)
    table_count = len(md_tables) + len(html_tables)

    main_content, _ = split_markdown_content_and_refs(md_content)
    sentence_count = count_sentences(main_content)
    return {
        'images': image_count,
        'equations': equation_count,
        'tables': table_count,
        'sentences': sentence_count
    }

# --------------Utility Functions-------------

def extract_high_frequency_words(category: str, top_n: int = 100) -> None:
    """
    Extract high frequency words from outlines and markdown files in a category.
    
    Args:
        category: Category name (e.g., 'econ')
        top_n: Number of top frequency words to extract
    """
    # Initialize stopwords
    stop_words = set(stopwords.words('english'))
    # Add custom stopwords
    custom_stop_words = {
        'et', 'al', 'e.g', 'i.e', 'fig', 'table', 'section', 'chapter',
        'figure', 'figures', 'tables', 'sections', 'chapters', 'paper',
        'study', 'studies', 'research', 'researchers', 'authors', 'author',
        'review', 'reviews', 'survey', 'surveys', 'analysis', 'analyses',
        'method', 'methods', 'approach', 'approaches', 'result', 'results',
        'conclusion', 'conclusions', 'introduction', 'background', 'related',
        'work', 'works', 'literature', 'discussion', 'discussions', 'summary',
        'summaries', 'overview', 'overviews', 'example', 'examples', 'case',
        'cases', 'model', 'models', 'framework', 'frameworks', 'system',
        'systems', 'algorithm', 'algorithms', 'technique', 'techniques',
        'methodology', 'methodologies', 'experiment', 'experiments',
        'evaluation', 'evaluations', 'performance', 'performances',
        'implementation', 'implementations', 'application', 'applications',
        'development', 'developments', 'design', 'designs', 'process',
        'processes', 'analysis', 'analyses', 'evaluation', 'evaluations',
        'comparison', 'comparisons', 'comparative', 'comparatively',
        'experimental', 'experimentally', 'theoretical', 'theoretically',
        'empirical', 'empirically', 'practical', 'practically', 'technical',
        'technically', 'theoretical', 'theoretically', 'empirical',
        'empirically', 'practical', 'practically', 'technical', 'technically',
        'theoretical', 'theoretically', 'empirical', 'empirically',
        'practical', 'practically', 'technical', 'technically'
    }
    stop_words.update(custom_stop_words)
    
    # Initialize word counter
    word_counter = Counter()
    
    # Process all topic directories in the category
    category_dir = os.path.join("surveys", category)
    if not os.path.exists(category_dir):
        print(f"Category directory {category_dir} does not exist")
        return
        
    for topic_dir in os.listdir(category_dir):
        topic_path = os.path.join(category_dir, topic_dir)
        if not os.path.isdir(topic_path):
            continue
            
        pdfs_dir = os.path.join(topic_path, "pdfs")
        if not os.path.exists(pdfs_dir):
            continue
            
        # Process outline files
        outline_path = os.path.join(pdfs_dir, "outline_raw.json")
        if os.path.exists(outline_path):
            try:
                with open(outline_path, 'r', encoding='utf-8') as f:
                    outline_data = json.load(f)
                    # Extract text from outline (assuming list of [level, title] pairs)
                    outline_text = ' '.join([
                        str(item[1]) if isinstance(item, list) and len(item) > 1 else str(item)
                        for item in outline_data
                    ])
                    # Tokenize and count words
                    words = word_tokenize(outline_text.lower())
                    # Filter words
                    words = [
                        word for word in words 
                        if word.isalnum() 
                        and word not in stop_words 
                        and is_valid_word(word)
                    ]
                    word_counter.update(words)
            except Exception as e:
                print(f"Error processing outline {outline_path}: {e}")
        
        # Process markdown files
        for file in os.listdir(pdfs_dir):
            if file.endswith('.md'):
                md_path = os.path.join(pdfs_dir, file)
                try:
                    with open(md_path, 'r', encoding='utf-8') as f:
                        md_text = f.read()
                        # Remove code blocks
                        md_text = re.sub(r'```.*?```', '', md_text, flags=re.DOTALL)
                        # Remove inline code
                        md_text = re.sub(r'`.*?`', '', md_text)
                        # Remove URLs
                        md_text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', md_text)
                        # Remove email addresses
                        md_text = re.sub(r'[\w\.-]+@[\w\.-]+', '', md_text)
                        # Remove special characters
                        md_text = re.sub(r'[^\w\s]', ' ', md_text)
                        # Tokenize and count words
                        words = word_tokenize(md_text.lower())
                        # Filter words
                        words = [
                            word for word in words 
                            if word.isalnum() 
                            and word not in stop_words 
                            and is_valid_word(word)
                        ]
                        word_counter.update(words)
                except Exception as e:
                    print(f"Error processing markdown {md_path}: {e}")
    
    # Get top N words
    top_words = word_counter.most_common(top_n)
    
    # Save results
    output_path = os.path.join(category_dir, "high_frequency_words.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "words": [{"word": word, "count": count} for word, count in top_words]
        }, f, indent=2, ensure_ascii=False)
    
    print(f"Saved top {len(top_words)} frequent words to {output_path}")

def process_and_evaluate_pdf(pdf_path: str) -> dict:
    """
    Process a PDF file and evaluate all metrics (both general and domain-specific).
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        dict: Dictionary containing all evaluation results
    """
    from evaluate import evaluate
    
    # Get directory and filename
    pdf_dir = os.path.dirname(pdf_path)
    pdf_name = os.path.basename(pdf_path)
    pdf_stem = os.path.splitext(pdf_name)[0]
    
    # Step 1: Convert PDF to markdown
    print(f"Converting PDF to markdown: {pdf_path}")
    if os.path.exists(os.path.join(pdf_dir, pdf_stem + ".md")):
        md_path = os.path.join(pdf_dir, pdf_stem + ".md")
    else:
        md_path, md_content = pdf2md(pdf_path, pdf_dir)
        if not md_path:
            print(f"Failed to convert PDF: {pdf_path}")
        return {}
    
    # Step 2: Extract and refine outline

    # Step 3: Extract references using reference.py logic 
    extract_refs(input_file=md_path, output_folder=os.path.dirname(md_path))
    # Save sentence-reference mapping
    
    # Step 4: Evaluate with general criteria
    print("Evaluating with general criteria...")
    general_results = evaluate(
        md_path=md_path,
        do_outline=True,
        do_content=True,
        do_reference=True,
        criteria_type="general"
    )
    
    # Step 5: Evaluate with domain-specific criteria
    print("Evaluating with domain-specific criteria...")
    domain_results = evaluate(
        md_path=md_path,
        do_outline=True,
        do_content=True,
        do_reference=True,
        criteria_type="domain"
    )
    
    # Combine results
    all_results = {
        "general": general_results,
        "domain": domain_results,
        "metadata": {
            "pdf_path": pdf_path,
            "md_path": md_path
        }
    }
    
    # # Save combined results
    # results_path = os.path.join(pdf_dir, "all_results.json")
    # with open(results_path, "w", encoding="utf-8") as f:
    #     json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    # print(f"All results saved to: {results_path}")
    return all_results

if __name__ == "__main__":
    # # 测试提取大纲
    # md_file_path = "surveys/cs/Optimization Techniques for Transformer Inference/pdfs/2307.07982.md"
    # outline = extract_and_save_outline_from_md(md_file_path)
    # print(len(outline))
    # for item in outline:
    #     print(item)
    # # 打印大纲
    # print("Outline:")
    # tree, top_nodes = build_outline_tree_from_levels(outline)
    # print(len(tree))

    # md_path, md_content = pdf2md("surveys/cs/3D Gaussian Splatting Techniques/LLMxMapReduce/5_1_2025, 6_14_21 PM_3D Gaussian Splatting Techniques.pdf", "surveys/cs/3D Gaussian Splatting Techniques/LLMxMapReduce")
    # md_content = read_md("surveys\cs\Large Language Model Based Multi-Agent Systems\pdfs/2402.01680.md")
    # print(count_md_features(md_content))
    # refine_outline_if_single_level("surveys\cs\Optimization Techniques for Transformer Inference\pdfs\outline_raw.json", "surveys\cs\Optimization Techniques for Transformer Inference\pdfs\outline.json")
    # batch_pdf2md_in_surveys()
    # extract_high_frequency_words("econ")
    process_and_evaluate_pdf("temp/2411.15594v5.pdf")
    # print(extract_references_from_md("temp/2411.15594v5.md"))