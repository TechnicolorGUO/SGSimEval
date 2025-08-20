import datetime
import json
import logging
import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from typing import List, Union
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot import radar_factory


load_dotenv()

# Configure logging
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Configure file handlers for different log types
embedding_handler = logging.FileHandler(os.path.join(LOG_DIR, 'embedding_responses.log'))
embedding_handler.setLevel(logging.INFO)
embedding_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

similarity_handler = logging.FileHandler(os.path.join(LOG_DIR, 'similarity_calculations.log'))
similarity_handler.setLevel(logging.INFO)
similarity_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Create separate loggers for embedding and similarity
embedding_logger = logging.getLogger('embedding')
embedding_logger.setLevel(logging.INFO)
embedding_logger.addHandler(embedding_handler)

similarity_logger = logging.getLogger('similarity')
similarity_logger.setLevel(logging.INFO)
similarity_logger.addHandler(similarity_handler)

# Configure root logger for general logs
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Initialize ChromaDB persistent client
CHROMA_DB_DIR = "chromadb"
# if not os.path.exists(CHROMA_DB_DIR):
#     os.makedirs(CHROMA_DB_DIR)
# chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

# --------------OpenAI/LLM Related Functions-------------

def getClient() -> OpenAI: 
    """
    Initialize and return an OpenAI client using environment variables.
    
    Returns:
        OpenAI: Configured OpenAI client instance
    """
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    openai_api_base = os.environ.get("OPENAI_API_BASE")

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    return client

def generateResponse(client: OpenAI, prompt: str, max_tokens: int = 8192, temperature: float = 0.5) -> str:
    """
    Generate a response from the OpenAI model using streaming.
    
    Args:
        client (OpenAI): OpenAI client instance
        prompt (str): Input prompt for the model
        max_tokens (int, optional): Maximum tokens in response. Defaults to 8192.
        temperature (float, optional): Response randomness. Defaults to 0.5.
    
    Returns:
        str: Generated response text
    """
    # Configure logging
    logging.basicConfig(
        filename='llm_responses.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Log request information without content
    request_info = {
        'timestamp': datetime.datetime.now().isoformat(),
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
    
    # Log response information without content
    response_info = {
        'timestamp': datetime.datetime.now().isoformat(),
        'response_length': len(text)
    }
    logging.info(f"Response: {json.dumps(response_info)}")
    
    return text

def embed_text(text: str) -> list[float]:
    """
    Get embedding for a text using OpenAI API.
    
    Args:
        text (str): Text to embed
        
    Returns:
        list[float]: Embedding vector
    """
    try:
        client = getClient()
        response = client.embeddings.create(
            model=os.environ.get("MODEL"),
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"Error getting embedding: {e}")
        return None

def embed_texts_batch(texts: List[str], batch_size: int = 10) -> List[List[float]]:
    """
    Get embeddings for a batch of texts using OpenAI API.
    
    Args:
        texts (List[str]): List of texts to embed
        batch_size (int, optional): Number of texts to process in one batch. Defaults to 10.
        
    Returns:
        List[List[float]]: List of embedding vectors
    """
    try:
        client = getClient()
        embeddings = []
        
        # Process texts in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = client.embeddings.create(
                model=os.environ.get("MODEL"),
                input=batch
            )
            batch_embeddings = [data.embedding for data in response.data]
            embeddings.extend(batch_embeddings)
            
            # Log batch response with model details
            log_info = {
                'timestamp': datetime.datetime.now().isoformat(),
                'batch_size': len(batch),
                'embeddings_generated': len(batch_embeddings),
                'model': os.environ.get("MODEL"),
                'model_response': {
                    'model': response.model,
                    'usage': {
                        'prompt_tokens': response.usage.prompt_tokens,
                        'total_tokens': response.usage.total_tokens
                    },
                    'data': [{
                        'index': data.index,
                        'embedding_length': len(data.embedding)
                    } for data in response.data]
                }
            }
            embedding_logger.info(f"Batch Embedding Response: {json.dumps(log_info)}")
            
        return embeddings
    except Exception as e:
        embedding_logger.error(f"Error getting batch embeddings: {e}")
        return None

def extract_topic_from_path(md_path: str) -> str:
    """
    Extract topic name from markdown file path.
    
    Args:
        md_path (str): Path to markdown file
        
    Returns:
        str: Topic name
    """
    abs_path = os.path.abspath(md_path)
    topic = os.path.basename(os.path.dirname(os.path.dirname(abs_path)))
    return topic

def get_collection(topic: str, suffix: str, mode: str = "create") -> chromadb.Collection:
    """
    Get or create a ChromaDB collection for a specific topic and type.
    
    Args:
        topic (str): Topic name
        suffix (str): Collection type suffix (_outline, _content, or _reference)
        mode (str): "create" to create if not exists, "get" to only get existing
        
    Returns:
        chromadb.Collection: ChromaDB collection
    """
    # Remove spaces and special characters from topic
    topic = topic.replace(" ", "_").replace(".", "").replace("-", "_")
    
    collection_name = f"{topic}{suffix}"
    
    # Initialize a new client for each operation
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    
    try:
        if mode == "create":
            try:
                return client.get_collection(collection_name)
            except Exception as e:
                if "Nothing found on disk" in str(e):
                    # If collection doesn't exist, create it
                    return client.create_collection(collection_name)
                else:
                    raise e
        else:  # mode == "get"
            return client.get_collection(collection_name)
    except Exception as e:
        raise e

def log_embedding_response(text: str, embedding: List[float], collection_name: str, doc_id: str) -> None:
    """
    Log embedding response information.
    
    Args:
        text (str): Original text that was embedded
        embedding (List[float]): Generated embedding vector
        collection_name (str): Name of the ChromaDB collection
        doc_id (str): Document ID in the collection
    """
    log_info = {
        'timestamp': datetime.datetime.now().isoformat(),
        'text_preview': text[:100] + '...' if len(text) > 100 else text,
        'embedding_length': len(embedding),
        'collection': collection_name,
        'doc_id': doc_id
    }
    embedding_logger.info(f"Embedding Response: {json.dumps(log_info)}")

def cal_outline_embedding(md_path: str, batch_size: int = 10) -> List[List[float]]:
    """
    Calculate embeddings for outline paths.
    
    Args:
        md_path (str): Path to markdown file
        batch_size (int, optional): Number of texts to process in one batch. Defaults to 10.
        
    Returns:
        List[List[float]]: List of embedding vectors for each outline path
    """
    try:
        # Get outline paths
        outline_path = os.path.join(os.path.dirname(md_path), "outline.json")
        outline_list = json.load(open(outline_path, "r", encoding="utf-8"))
        paths = build_outline_tree_from_levels(outline_list, md_path)
        
        # Get collection
        topic = extract_topic_from_path(md_path)
        collection = get_collection(topic, "_outline")
        
        # Prepare texts for batch processing
        texts = [" > ".join(path) for path in paths]
        
        # Get embeddings in batches
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating outline embeddings"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = embed_texts_batch(batch_texts, batch_size)
            
            if batch_embeddings:
                embeddings.extend(batch_embeddings)
                # Add to collection
                for j, (text, embedding) in enumerate(zip(batch_texts, batch_embeddings)):
                    doc_id = f"{topic}_outline_{i + j}"
                    collection.add(
                        embeddings=[embedding],
                        documents=[text],
                        ids=[doc_id]
                    )
                    # Log response
                    log_embedding_response(text, embedding, collection.name, doc_id)
        
        return embeddings
    except Exception as e:
        logging.error(f"Error calculating outline embeddings: {e}")
        return None

def cal_content_embedding(md_path: str, batch_size: int = 10) -> List[List[float]]:
    """
    Calculate embeddings for content sections.
    
    Args:
        md_path (str): Path to markdown file
        batch_size (int, optional): Number of texts to process in one batch. Defaults to 10.
        
    Returns:
        List[List[float]]: List of embedding vectors for each content section
    """
    try:
        # Read content
        content_path = os.path.join(os.path.dirname(md_path), "content.json")
        content_list = json.load(open(content_path, "r", encoding="utf-8"))
        content = content_list[0]  # Assuming content is the first element
        
        # Split content by headers
        sections = []
        current_section = []
        lines = content.split('\n')
        
        for line in lines:
            if line.startswith('#'):
                if current_section:
                    sections.append('\n'.join(current_section))
                    current_section = []
            current_section.append(line)
        if current_section:
            sections.append('\n'.join(current_section))
        
        # Get collection
        topic = extract_topic_from_path(md_path)
        collection = get_collection(topic, "_content")
        
        # Get embeddings in batches
        embeddings = []
        for i in tqdm(range(0, len(sections), batch_size), desc="Generating content embeddings"):
            batch_sections = sections[i:i + batch_size]
            batch_embeddings = embed_texts_batch(batch_sections, batch_size)
            
            if batch_embeddings:
                embeddings.extend(batch_embeddings)
                # Add to collection
                for j, (section, embedding) in enumerate(zip(batch_sections, batch_embeddings)):
                    doc_id = f"{topic}_content_{i + j}"
                    collection.add(
                        embeddings=[embedding],
                        documents=[section],
                        ids=[doc_id]
                    )
                    # Log response
                    log_embedding_response(section, embedding, collection.name, doc_id)
        
        return embeddings
    except Exception as e:
        logging.error(f"Error calculating content embeddings: {e}")
        return None

def cal_reference_embedding(md_path: str, batch_size: int = 10) -> List[List[float]]:
    """
    Calculate embeddings for references.
    
    Args:
        md_path (str): Path to markdown file
        batch_size (int, optional): Number of texts to process in one batch. Defaults to 10.
        
    Returns:
        List[List[float]]: List of embedding vectors for each reference
    """
    try:
        # Read references
        ref_path = os.path.join(os.path.dirname(md_path), "references.json")
        references = json.load(open(ref_path, "r", encoding="utf-8"))
        
        # Get collection
        topic = extract_topic_from_path(md_path)
        collection = get_collection(topic, "_reference")
        
        # Get embeddings in batches
        embeddings = []
        for i in tqdm(range(0, len(references), batch_size), desc="Generating reference embeddings"):
            batch_refs = references[i:i + batch_size]
            batch_embeddings = embed_texts_batch(batch_refs, batch_size)
            
            if batch_embeddings:
                embeddings.extend(batch_embeddings)
                # Add to collection
                for j, (ref, embedding) in enumerate(zip(batch_refs, batch_embeddings)):
                    doc_id = f"{topic}_reference_{i + j}"
                    collection.add(
                        embeddings=[embedding],
                        documents=[ref],
                        ids=[doc_id]
                    )
                    # Log response
                    log_embedding_response(ref, embedding, collection.name, doc_id)
        
        return embeddings
    except Exception as e:
        logging.error(f"Error calculating reference embeddings: {e}")
        return None

def build_outline_tree_from_levels(outline_list: list[list[int | str]], md_path: str = None) -> list[list[str]]:
    """
    Parse [level, title] format outline into paths from root to leaf nodes.
    Each path represents a complete path from root to leaf, with siblings merged into the same path.
    
    Args:
        outline_list (list[list[int | str]]): List of [level, title] pairs, levels start from 1
        md_path (str, optional): Path to markdown file, used to extract topic if needed
        
    Returns:
        list[list[str]]: List of paths from root to leaf nodes, with siblings merged
    """
    # Check if there's a single root node
    root_nodes = [item for item in outline_list if item[0] == 1]
    if len(root_nodes) != 1 and md_path:
        # If no single root node, use topic as root
        topic = extract_topic_from_path(md_path)
        # Adjust all levels by adding 1 to make room for topic
        adjusted_outline = [[level + 1, title] for level, title in outline_list]
        # Add topic as root
        outline_list = [[1, topic]] + adjusted_outline
    
    # Build tree structure
    node_objs = []
    for idx, (level, title) in enumerate(outline_list):
        node = {
            "level": level,
            "title": title,
            "index": idx,
            "children": [],
            "parent": None,
            "is_leaf": True  # Initially assume all nodes are leaves
        }
        node_objs.append(node)

    # Build parent-child relationships
    stack = []
    for node in node_objs:
        while stack and stack[-1]["level"] >= node["level"]:
            stack.pop()
        if stack:
            node["parent"] = stack[-1]["index"]
            stack[-1]["children"].append(node)
            stack[-1]["is_leaf"] = False  # Parent is not a leaf
        stack.append(node)

    # Find all paths from root to leaf nodes
    paths = []
    def get_path_to_leaf(node: dict, current_path: list = None) -> None:
        """
        Recursively get paths to all leaf nodes in the outline tree.
        
        Args:
            node (dict): Current node containing title and is_leaf properties
            current_path (list, optional): Current path being built. Defaults to None.
        """
        if current_path is None:
            current_path = []
        
        current_path = current_path + [node["title"]]
        
        if node["is_leaf"]:
            # This is a leaf node, add the complete path
            paths.append(current_path)
        else:
            # For each child, create a new path
            for child in node["children"]:
                get_path_to_leaf(child, current_path.copy())
    
    # Start from root nodes
    root_nodes = [node for node in node_objs if node["parent"] is None]
    for root in root_nodes:
        get_path_to_leaf(root)
    
    # Merge paths with the same parent
    merged_paths = []
    parent_paths = {}  # Dictionary to store paths by their parent path
    
    for path in paths:
        if len(path) <= 2:  # Skip paths with length <= 2
            continue
            
        # Get the parent path (all but the last element)
        parent_path = tuple(path[:-1])
        if parent_path not in parent_paths:
            parent_paths[parent_path] = []
        parent_paths[parent_path].append(path)
    
    # For each parent path, if it has multiple children, merge them
    for parent_path, child_paths in parent_paths.items():
        if len(child_paths) > 1:
            # Merge all children into one path
            merged_path = list(parent_path)
            # Add all leaf nodes
            for path in child_paths:
                merged_path.append(path[-1])
            merged_paths.append(merged_path)
        else:
            # If only one child, keep the original path
            merged_paths.append(child_paths[0])
    
    return merged_paths

def query_similar_documents(query_text: str, topic: str, collection_type: str, top_n: int = 5) -> List[dict]:
    """
    Query the most similar documents from a specific collection.
    
    Args:
        query_text (str): Query text to search for
        topic (str): Topic name to identify the collection
        collection_type (str): Type of collection (_outline, _content, or _reference)
        top_n (int, optional): Number of top results to return. Defaults to 5.
        
    Returns:
        List[dict]: List of dictionaries containing document information and similarity scores
        Each dictionary contains:
        - text: The document text
        - score: Similarity score
        - doc_id: Document ID
    """
    try:
        # Get collection
        collection = get_collection(topic, collection_type)
        
        # Get query embedding
        query_embedding = embed_text(query_text)
        if not query_embedding:
            logging.error("Failed to get query embedding")
            return []
        
        # Query the collection
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_n,
            include=["documents", "distances"]
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'text': results['documents'][0][i],
                'score': 1 - results['distances'][0][i],  # Convert distance to similarity score
                'doc_id': results['ids'][0][i]
            })
        
        # Log query
        log_info = {
            'timestamp': datetime.datetime.now().isoformat(),
            'query': query_text,
            'topic': topic,
            'collection_type': collection_type,
            'top_n': top_n,
            'results_count': len(formatted_results)
        }
        logging.info(f"Query Results: {json.dumps(log_info)}")
        
        return formatted_results
    except Exception as e:
        logging.error(f"Error querying similar documents: {e}")
        return []

def cal_similarity(md_path: str, topic: str, outline_top_n: int = 5, content_top_n: int = 5, reference_top_n: int = 20, recalculate_items: List[str] = None) -> dict:
    """
    Calculate similarity scores between a new document and existing documents in the collection.
    
    Args:
        md_path (str): Path to the new markdown file
        topic (str): Topic name
        outline_top_n (int, optional): Number of top outline matches to return. Defaults to 5.
        content_top_n (int, optional): Number of top content matches to return. Defaults to 5.
        reference_top_n (int, optional): Number of top reference matches to return. Defaults to 20.
        recalculate_items (List[str], optional): List of items to recalculate. Can be ['outline', 'content', 'reference'].
        
    Returns:
        dict: Dictionary containing similarity scores and their means
    """
    # Initialize result dictionary
    result = {
        'outline_scores': [],
        'content_scores': [],
        'reference_scores': [],
        'mean_outline_score': 0,
        'mean_content_score': 0,
        'mean_reference_score': 0
    }
    
    # Get the directory containing the markdown file
    md_dir = os.path.dirname(md_path)
    
    # Process outline if needed
    if recalculate_items is None or 'outline' in recalculate_items:
        max_retries = 3
        retry_delay = 5  # seconds
        
        # Process outline with retries
        for attempt in range(max_retries):
            try:
                outline_path = os.path.join(md_dir, "outline.json")
                outline_list = json.load(open(outline_path, "r", encoding="utf-8"))
                paths = build_outline_tree_from_levels(outline_list, md_path)
                
                # Log outline texts before embedding
                log_info = {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'md_path': md_path,
                    'topic': topic,
                    # 'outline_texts': paths
                }
                embedding_logger.info(f"Processing outline texts: {json.dumps(log_info)}")
                
                outline_texts = [" > ".join(path) for path in paths]
                outline_embeddings = embed_texts_batch(outline_texts, batch_size)
                
                # Get outline collection
                outline_collection = get_collection(topic, "_outline", mode="get")
                
                # Calculate similarities for outline
                outline_scores = []
                if outline_embeddings:
                    for embedding in outline_embeddings:
                        results = outline_collection.query(
                            query_embeddings=[embedding],
                            n_results=1,
                            include=["distances"]
                        )
                        if results['distances'][0]:
                            outline_scores.append(1 - results['distances'][0][0])
                
                # Sort scores and get top N
                outline_scores = sorted(outline_scores, reverse=True)[:outline_top_n]
                result['outline_scores'] = outline_scores
                result['mean_outline_score'] = sum(outline_scores) / len(outline_scores) if outline_scores else 0
                break  # If successful, break the retry loop
            except Exception as e:
                if attempt < max_retries - 1:
                    similarity_logger.warning(f"Attempt {attempt + 1} failed for outline processing: {e} {md_path}, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    similarity_logger.error(f"All attempts failed for outline processing: {e} {md_path}")
                    result['mean_outline_score'] = 0
    
    # Process content if needed
    if recalculate_items is None or 'content' in recalculate_items:
        max_retries = 3
        retry_delay = 5  # seconds
        
        # Process content with retries
        for attempt in range(max_retries):
            try:
                content_path = os.path.join(md_dir, "content.json")
                content_list = json.load(open(content_path, "r", encoding="utf-8"))
                content = content_list[0]
                sections = []
                current_section = []
                for line in content.split('\n'):
                    if line.startswith('#'):
                        if current_section:
                            sections.append('\n'.join(current_section))
                            current_section = []
                    current_section.append(line)
                if current_section:
                    sections.append('\n'.join(current_section))
                
                # Log content sections before embedding
                log_info = {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'md_path': md_path,
                    'topic': topic,
                    'section_count': len(sections),
                    # 'sections': [section[:100] + '...' if len(section) > 100 else section for section in sections]
                }
                embedding_logger.info(f"Processing content sections: {json.dumps(log_info)}")
                
                content_embeddings = embed_texts_batch(sections)
                
                # Get content collection
                content_collection = get_collection(topic, "_content", mode="get")
                
                # Calculate similarities for content
                content_scores = []
                if content_embeddings:
                    for embedding in content_embeddings:
                        results = content_collection.query(
                            query_embeddings=[embedding],
                            n_results=1,
                            include=["distances"]
                        )
                        if results['distances'][0]:
                            content_scores.append(1 - results['distances'][0][0])
                
                # Sort scores and get top N
                content_scores = sorted(content_scores, reverse=True)[:content_top_n]
                result['content_scores'] = content_scores
                result['mean_content_score'] = sum(content_scores) / len(content_scores) if content_scores else 0
                break  # If successful, break the retry loop
            except Exception as e:
                if attempt < max_retries - 1:
                    similarity_logger.warning(f"Attempt {attempt + 1} failed for content processing: {e} {md_path}, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    similarity_logger.error(f"All attempts failed for content processing: {e} {md_path}")
                    result['mean_content_score'] = 0
    
    # Process references if needed
    if recalculate_items is None or 'reference' in recalculate_items:
        max_retries = 3
        retry_delay = 5  # seconds
        
        # Process references with retries
        for attempt in range(max_retries):
            try:
                ref_path = os.path.join(md_dir, "references.json")
                references = json.load(open(ref_path, "r", encoding="utf-8"))
                
                # Log references before embedding
                log_info = {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'md_path': md_path,
                    'topic': topic,
                    'reference_count': len(references),
                    # 'references': [ref[:100] + '...' if len(ref) > 100 else ref for ref in references]
                }
                embedding_logger.info(f"Processing references: {json.dumps(log_info)}")
                
                reference_embeddings = embed_texts_batch(references)
                
                # Get reference collection
                reference_collection = get_collection(topic, "_reference", mode="get")
                
                # Calculate similarities for references
                reference_scores = []
                if reference_embeddings:
                    for embedding in reference_embeddings:
                        results = reference_collection.query(
                            query_embeddings=[embedding],
                            n_results=1,
                            include=["distances"]
                        )
                        if results['distances'][0]:
                            reference_scores.append(1 - results['distances'][0][0])
                
                # Sort scores and get top N
                reference_scores = sorted(reference_scores, reverse=True)[:reference_top_n]
                result['reference_scores'] = reference_scores
                result['mean_reference_score'] = sum(reference_scores) / len(reference_scores) if reference_scores else 0
                break  # If successful, break the retry loop
            except Exception as e:
                if attempt < max_retries - 1:
                    similarity_logger.warning(f"Attempt {attempt + 1} failed for reference processing: {e} {md_path}, retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    similarity_logger.error(f"All attempts failed for reference processing: {e} {md_path}")
                    result['mean_reference_score'] = 0
    
    # Log results
    log_info = {
        'timestamp': datetime.datetime.now().isoformat(),
        'md_path': md_path,
        'topic': topic,
        'recalculated_items': recalculate_items,
        'outline_scores_count': len(result['outline_scores']),
        'content_scores_count': len(result['content_scores']),
        'reference_scores_count': len(result['reference_scores']),
        'mean_outline': result['mean_outline_score'],
        'mean_content': result['mean_content_score'],
        'mean_reference': result['mean_reference_score']
    }
    similarity_logger.info(f"Similarity Calculation Results: {json.dumps(log_info)}")
    
    return result

def process_all_datasets() -> None:
    """
    Process all markdown files in surveys/all_dataset and surveys/cs_dataset directories.
    For each markdown file, generate embeddings for outline, content, and references.
    Skip if collection already exists.
    """
    # Define dataset paths
    dataset_paths = [
        "surveys/all_dataset",
        "surveys/cs_dataset"
    ]
    
    # Collect all markdown files
    all_md_files = []
    for dataset_path in dataset_paths:
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.md'):
                    all_md_files.append(os.path.join(root, file))
    
    print(f"\nFound {len(all_md_files)} markdown files to process")
    
    # Process each file with a progress bar
    for md_path in tqdm(all_md_files, desc="Processing datasets"):
        # print(f"Processing {md_path}")
        # Extract topic from path
        topic = extract_topic_from_path(md_path)
        topic = topic.replace(" ", "_").replace(".", "").replace("-", "_")
        
        # Check if collections already exist
        try:
            outline_collection = get_collection(topic, "_outline")
            content_collection = get_collection(topic, "_content")
            reference_collection = get_collection(topic, "_reference")
            
            # If we can get all collections, skip this file
            if outline_collection and content_collection and reference_collection:
                logging.info(f"Skipping {md_path} - collections already exist")
                # continue
        except:
            # If any collection doesn't exist, proceed with processing
            print(f"Skipping {md_path}")
            pass
        
        # Process the file
        logging.info(f"Processing {md_path}")
        
        # Generate embeddings
        print(f"Generating embeddings for {md_path}")
        outline_embeddings = cal_outline_embedding(md_path)
        if outline_embeddings:
            logging.info(f"Generated {len(outline_embeddings)} outline embeddings for {md_path}")
        
        content_embeddings = cal_content_embedding(md_path)
        if content_embeddings:
            logging.info(f"Generated {len(content_embeddings)} content embeddings for {md_path}")
        
        reference_embeddings = cal_reference_embedding(md_path)
        if reference_embeddings:
            logging.info(f"Generated {len(reference_embeddings)} reference embeddings for {md_path}")

def calculate_and_update_similarity_scores(model_name: str, md_path: str, ground_truth_md_path: str) -> None:
    """
    Calculate similarity scores and update the results JSON file.
    
    Args:
        model_name (str): Name of the model (e.g., 'qwen-plus-latest')
        md_path (str): Path to the markdown file
        ground_truth_md_path (str): Path to the ground truth markdown file
    """
    try:
        # Get topic from md_path
        topic = extract_topic_from_path(md_path)
        
        # Check if similarity.json exists
        md_dir = os.path.dirname(md_path)
        similarity_path = os.path.join(md_dir, "similarity.json")
        
        # Determine which items need to be recalculated
        recalculate_items = []
        if os.path.exists(similarity_path):
            existing_results = json.load(open(similarity_path, "r", encoding="utf-8"))
            if existing_results['mean_outline_score'] == 0:
                recalculate_items.append('outline')
            if existing_results['mean_content_score'] == 0:
                recalculate_items.append('content')
            if existing_results['mean_reference_score'] == 0:
                recalculate_items.append('reference')
            
            if not recalculate_items:
                print(f"Skipping similarity calculation for {md_path} - all scores are non-zero")
                similarity_results = existing_results
            else:
                print(f"Recalculating similarity scores for {md_path} - items: {recalculate_items}")
                # Calculate only the needed similarity scores
                new_results = cal_similarity(md_path, topic, recalculate_items=recalculate_items)
                # Merge new results with existing ones
                similarity_results = existing_results.copy()
                for item in recalculate_items:
                    if item == 'outline':
                        similarity_results['outline_scores'] = new_results['outline_scores']
                        similarity_results['mean_outline_score'] = new_results['mean_outline_score']
                    elif item == 'content':
                        similarity_results['content_scores'] = new_results['content_scores']
                        similarity_results['mean_content_score'] = new_results['mean_content_score']
                    elif item == 'reference':
                        similarity_results['reference_scores'] = new_results['reference_scores']
                        similarity_results['mean_reference_score'] = new_results['mean_reference_score']
        else:
            # Calculate all similarity scores
            print(f"Calculating all similarity scores for {md_path}")
            similarity_results = cal_similarity(md_path, topic)
        
        # Save similarity scores to similarity.json
        with open(similarity_path, "w", encoding="utf-8") as f:
            json.dump(similarity_results, f, indent=4)
        
        # Read results JSON for the current file
        results_path = os.path.join(md_dir, f"results_{model_name}.json")
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        
        # Read results JSON for the ground truth file
        gt_md_dir = os.path.dirname(ground_truth_md_path)
        gt_results_path = os.path.join(gt_md_dir, f"results_{model_name}.json")
        with open(gt_results_path, "r", encoding="utf-8") as f:
            gt_results = json.load(f)
        
        # Calculate similarity scores
        metrics = ["Outline", "Coverage", "Structure", "Relevance", "Language", "Criticalness", "Reference"]
        similarity_scores = {}
        
        for metric in metrics:
            # Get scores from both files
            current_score = results.get(metric, 0)
            gt_score = gt_results.get(metric, 0)
            
            # Calculate similarity score based on metric type
            if metric == "Outline":
                similarity = (1 - similarity_results['mean_outline_score']) * current_score + \
                           similarity_results['mean_outline_score'] * gt_score
            elif metric == "Reference":
                similarity = (1 - similarity_results['mean_reference_score']) * current_score + \
                           similarity_results['mean_reference_score'] * gt_score
            else:
                similarity = (1 - similarity_results['mean_content_score']) * current_score + \
                           similarity_results['mean_content_score'] * gt_score
            
            # Store the result with _sim suffix
            similarity_scores[f"{metric}_sim"] = similarity
        
        # Calculate additional metrics
        # Outline_structure: Uses outline similarity
        current_outline_structure = results.get("Outline_structure", 0)
        gt_outline_structure = gt_results.get("Outline_structure", 0)
        similarity_scores["Outline_structure_sim"] = (1 - similarity_results['mean_outline_score']) * current_outline_structure + \
                                                   similarity_results['mean_outline_score'] * gt_outline_structure
        
        # Faithfulness: Uses content similarity
        current_faithfulness = results.get("Faithfulness", 0)
        gt_faithfulness = gt_results.get("Faithfulness", 0)
        similarity_scores["Faithfulness_sim"] = (1 - similarity_results['mean_content_score']) * current_faithfulness + \
                                              similarity_results['mean_content_score'] * gt_faithfulness
        
        # Reference_quality: Uses reference similarity
        current_reference_quality = results.get("Reference_quality", 0)
        gt_reference_quality = gt_results.get("Reference_quality", 0)
        similarity_scores["Reference_quality_sim"] = (1 - similarity_results['mean_reference_score']) * current_reference_quality + \
                                                   similarity_results['mean_reference_score'] * gt_reference_quality
        
        # Update the results JSON with similarity scores
        results.update(similarity_scores)
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        
        logging.info(f"Updated similarity scores for {md_path}")
        return similarity_scores
        
    except Exception as e:
        print(e)
        logging.error(f"Error calculating similarity scores: {e}")
        return None

def calculate_and_update_similarity_scores_new(model_name: str, md_path: str, ground_truth_md_path: str) -> None:
    """
    Calculate similarity scores and update the results JSON file with additional human-perfect metrics.
    
    Args:
        model_name (str): Name of the model (e.g., 'qwen-plus-latest')
        md_path (str): Path to the markdown file
        ground_truth_md_path (str): Path to the ground truth markdown file
    """
    try:
        # Get topic from md_path
        topic = extract_topic_from_path(md_path)
        
        # Check if similarity.json exists
        md_dir = os.path.dirname(md_path)
        similarity_path = os.path.join(md_dir, "similarity.json")
        
        # Determine which items need to be recalculated
        recalculate_items = []
        if os.path.exists(similarity_path):
            existing_results = json.load(open(similarity_path, "r", encoding="utf-8"))
            if existing_results['mean_outline_score'] == 0:
                recalculate_items.append('outline')
            if existing_results['mean_content_score'] == 0:
                recalculate_items.append('content')
            if existing_results['mean_reference_score'] == 0:
                recalculate_items.append('reference')
            
            if not recalculate_items:
                print(f"Skipping similarity calculation for {md_path} - all scores are non-zero")
                similarity_results = existing_results
            else:
                print(f"Recalculating similarity scores for {md_path} - items: {recalculate_items}")
                # Calculate only the needed similarity scores
                new_results = cal_similarity(md_path, topic, recalculate_items=recalculate_items)
                # Merge new results with existing ones
                similarity_results = existing_results.copy()
                for item in recalculate_items:
                    if item == 'outline':
                        similarity_results['outline_scores'] = new_results['outline_scores']
                        similarity_results['mean_outline_score'] = new_results['mean_outline_score']
                    elif item == 'content':
                        similarity_results['content_scores'] = new_results['content_scores']
                        similarity_results['mean_content_score'] = new_results['mean_content_score']
                    elif item == 'reference':
                        similarity_results['reference_scores'] = new_results['reference_scores']
                        similarity_results['mean_reference_score'] = new_results['mean_reference_score']
        else:
            # Calculate all similarity scores
            print(f"Calculating all similarity scores for {md_path}")
            similarity_results = cal_similarity(md_path, topic)
        
        # Save similarity scores to similarity.json
        with open(similarity_path, "w", encoding="utf-8") as f:
            json.dump(similarity_results, f, indent=4)
        
        # Read results JSON for the current file
        results_path = os.path.join(md_dir, f"results_{model_name}.json")
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        
        # Read results JSON for the ground truth file
        gt_md_dir = os.path.dirname(ground_truth_md_path)
        gt_results_path = os.path.join(gt_md_dir, f"results_{model_name}.json")
        with open(gt_results_path, "r", encoding="utf-8") as f:
            gt_results = json.load(f)
        
        # Calculate similarity scores
        metrics = ["Outline", "Coverage", "Structure", "Relevance", "Language", "Criticalness", "Reference"]
        similarity_scores = {}
        
        # Calculate regular similarity scores
        for metric in metrics:
            # Get scores from both files
            current_score = results.get(metric, 0)
            gt_score = gt_results.get(metric, 0)
            
            # Calculate similarity score based on metric type
            if metric == "Outline":
                similarity = (1 - similarity_results['mean_outline_score']) * current_score + \
                           similarity_results['mean_outline_score'] * gt_score
            elif metric == "Reference":
                similarity = (1 - similarity_results['mean_reference_score']) * current_score + \
                           similarity_results['mean_reference_score'] * gt_score
            else:
                similarity = (1 - similarity_results['mean_content_score']) * current_score + \
                           similarity_results['mean_content_score'] * gt_score
            
            # Store the result with _sim suffix
            similarity_scores[f"{metric}_sim"] = similarity
            
            # Calculate human-perfect similarity scores
            if metric == "Outline":
                similarity_hp = (1 - similarity_results['mean_outline_score']) * current_score + \
                              similarity_results['mean_outline_score'] * 5.0
            elif metric == "Reference":
                similarity_hp = (1 - similarity_results['mean_reference_score']) * current_score + \
                              similarity_results['mean_reference_score'] * 5.0
            else:
                similarity_hp = (1 - similarity_results['mean_content_score']) * current_score + \
                              similarity_results['mean_content_score'] * 5.0
            
            # Store the result with _sim_hp suffix
            similarity_scores[f"{metric}_sim_hp"] = similarity_hp
        
        # Calculate additional metrics
        # Outline_structure: Uses outline similarity
        current_outline_structure = results.get("Outline_structure", 0)
        gt_outline_structure = gt_results.get("Outline_structure", 0)
        similarity_scores["Outline_structure_sim"] = (1 - similarity_results['mean_outline_score']) * current_outline_structure + \
                                                   similarity_results['mean_outline_score'] * gt_outline_structure
        similarity_scores["Outline_structure_sim_hp"] = (1 - similarity_results['mean_outline_score']) * current_outline_structure + \
                                                      similarity_results['mean_outline_score'] * 100.0
        
        # Faithfulness: Uses content similarity
        current_faithfulness = results.get("Faithfulness", 0)
        gt_faithfulness = gt_results.get("Faithfulness", 0)
        similarity_scores["Faithfulness_sim"] = (1 - similarity_results['mean_content_score']) * current_faithfulness + \
                                              similarity_results['mean_content_score'] * gt_faithfulness
        similarity_scores["Faithfulness_sim_hp"] = (1 - similarity_results['mean_content_score']) * current_faithfulness + \
                                                 similarity_results['mean_content_score'] * 100.0
        
        # Reference_quality: Uses reference similarity
        current_reference_quality = results.get("Reference_quality", 0)
        gt_reference_quality = gt_results.get("Reference_quality", 0)
        similarity_scores["Reference_quality_sim"] = (1 - similarity_results['mean_reference_score']) * current_reference_quality + \
                                                   similarity_results['mean_reference_score'] * gt_reference_quality
        similarity_scores["Reference_quality_sim_hp"] = (1 - similarity_results['mean_reference_score']) * current_reference_quality + \
                                                      similarity_results['mean_reference_score'] * 100.0
        
        # Update the results JSON with similarity scores
        results.update(similarity_scores)
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        
        logging.info(f"Updated similarity scores for {md_path}")
        return similarity_scores
        
    except Exception as e:
        print(e)
        logging.error(f"Error calculating similarity scores: {e}")
        return None

def batch_calculate_similarity_scores(
    system_list: list[str],
    model: str,
    domains: list[str] = ["cs"],
    tasks_json_path: str = "surveys/tasks.json",
) -> None:
    """
    Batch calculate similarity scores for all tasks for specified systems.
    
    Args:
        system_list (list[str]): List of system names to process
        model (str): Model name for evaluation
        domains (list[str], optional): List of domains to process. Defaults to ["cs"].
        tasks_json_path (str, optional): Path to tasks mapping JSON. Defaults to "surveys/tasks.json".
        num_workers (int, optional): Number of worker threads. Defaults to 1.
    """
    # Read tasks.json
    with open(tasks_json_path, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    
    tasks_to_run = []
    print("\nCollecting tasks...")
    for system in tqdm(system_list, desc="Processing systems"):
        if system not in tasks:
            print(f"System {system} not found in {tasks_json_path}, skip.")
            continue
        for topic_map in tasks[system]:
            for topic, rel_path in topic_map.items():
                sys_path = os.path.join("surveys", rel_path)
                # Check if the system path exists
                if not os.path.exists(sys_path):
                    print(f"[{topic}/{system}] System path does not exist: {sys_path}, skip.")
                    continue
                
                # Find markdown files
                try:
                    md_files = [f for f in os.listdir(sys_path) if f.lower().endswith(".md")]
                except Exception as e:
                    print(f"[{topic}/{system}] Error accessing directory {sys_path}: {e}, skip.")
                    continue
                
                if not md_files:
                    print(f"[{topic}/{system}] No md found, skip.")
                    continue
                
                md_path = os.path.join(sys_path, md_files[0])
                
                # Check each domain for ground truth files
                found_gt = False
                for domain in domains:
                    pdfs_path = os.path.join("surveys", domain, topic, "pdfs")
                    if not os.path.exists(pdfs_path):
                        continue
                    
                    gt_files = [f for f in os.listdir(pdfs_path) if f.lower().endswith(".md")]
                    if not gt_files:
                        continue
                    
                    gt_path = os.path.join(pdfs_path, gt_files[0])
                    tasks_to_run.append((model, md_path, gt_path))
                    found_gt = True
                    break
                
                if not found_gt:
                    print(f"[{topic}/{system}] No ground truth md found in any domain, skip.")
    
    print(f"\nFound {len(tasks_to_run)} tasks to process")
    
    # Process tasks sequentially to avoid database access conflicts
    for args in tqdm(tasks_to_run, desc="Processing tasks"):
        try:
            calculate_and_update_similarity_scores_new(*args)
            # Add a small delay between tasks to ensure database operations complete
            time.sleep(0.5)
        except Exception as e:
            print(f"Error processing {args[1]}: {e}")
            logging.error(f"Error processing {args[1]}: {e}")

def clear_similarity(domain: str, system: str) -> None:
    """
    Clear similarity files for a specific domain and system.
    
    Args:
        domain (str): Domain name (e.g., 'cs', 'all_dataset')
        system (str): System name (e.g., 'AutoSurvey', 'SurveyForge')
    """
    try:
        # Read tasks.json to get the paths
        with open("surveys/tasks.json", 'r', encoding='utf-8') as f:
            tasks = json.load(f)
        
        if system not in tasks:
            print(f"System {system} not found in tasks.json")
            return
        
        cleared_count = 0
        for topic_map in tasks[system]:
            for topic, rel_path in topic_map.items():
                # Get the directory containing the markdown file
                sys_path = os.path.join("surveys", rel_path)
                if not os.path.exists(sys_path):
                    continue
                
                # Find similarity.json in the directory
                similarity_path = os.path.join(sys_path, "similarity.json")
                if os.path.exists(similarity_path):
                    os.remove(similarity_path)
                    cleared_count += 1
                    print(f"Cleared similarity file for {topic}/{system}")
        
        print(f"\nCleared {cleared_count} similarity files for {system} in {domain}")
        
    except Exception as e:
        print(f"Error clearing similarity files: {e}")
        logging.error(f"Error clearing similarity files: {e}")

def clear_all_similarity() -> None:
    """
    Clear all similarity files across all domains and systems.
    """
    try:
        # Read tasks.json to get all systems
        with open("surveys/tasks.json", 'r', encoding='utf-8') as f:
            tasks = json.load(f)
        
        total_cleared = 0
        for system in tasks.keys():
            cleared_count = 0
            for topic_map in tasks[system]:
                for topic, rel_path in topic_map.items():
                    # Get the directory containing the markdown file
                    sys_path = os.path.join("surveys", rel_path)
                    if not os.path.exists(sys_path):
                        continue
                    
                    # Find similarity.json in the directory
                    similarity_path = os.path.join(sys_path, "similarity.json")
                    if os.path.exists(similarity_path):
                        os.remove(similarity_path)
                        cleared_count += 1
            
            if cleared_count > 0:
                print(f"Cleared {cleared_count} similarity files for {system}")
                total_cleared += cleared_count
        
        print(f"\nTotal cleared similarity files: {total_cleared}")
        
    except Exception as e:
        print(f"Error clearing all similarity files: {e}")
        logging.error(f"Error clearing all similarity files: {e}")

def normalize_score(score: float) -> float:
    """
    Normalize a score from 0-100 to 0-5 scale.
    
    Args:
        score (float): Original score in 0-100 scale
        
    Returns:
        float: Normalized score in 0-5 scale
    """
    return score / 20.0

def generate_evaluation_tex(system_list: List[str], model: str, output_path: str = "evaluation_results.tex") -> None:
    """
    Generate a tex table with evaluation results for all systems.
    
    Args:
        system_list (List[str]): List of system names
        model (str): Model name used for evaluation
        output_path (str): Path to save the tex file
    """
    # Define the metrics in order
    metrics = [
        "Outline",
        "Outline_structure",
        "Coverage",
        "Structure",
        "Relevance",
        "Language",
        "Criticalness",
        "Faithfulness",
        "Reference",
        "Reference_quality"
    ]
    
    # Define the corresponding similarity metrics
    sim_metrics = [
        "Outline_sim",
        "Outline_structure_sim",
        "Coverage_sim",
        "Structure_sim",
        "Relevance_sim",
        "Language_sim",
        "Criticalness_sim",
        "Faithfulness_sim",
        "Reference_sim",
        "Reference_quality_sim"
    ]
    
    # Define the corresponding human-perfect similarity metrics
    sim_hp_metrics = [
        "Outline_sim_hp",
        "Outline_structure_sim_hp",
        "Coverage_sim_hp",
        "Structure_sim_hp",
        "Relevance_sim_hp",
        "Language_sim_hp",
        "Criticalness_sim_hp",
        "Faithfulness_sim_hp",
        "Reference_sim_hp",
        "Reference_quality_sim_hp"
    ]
    
    # Read tasks.json to get all paths
    with open("surveys/tasks.json", 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    
    # Initialize results dictionary
    results = {system: {"original": [], "sim": [], "sim_hp": []} for system in system_list}
    
    # Collect results for each system
    for system in system_list:
        if system not in tasks and system != "pdfs":
            print(f"System {system} not found in tasks.json")
            continue
            
        if system == "pdfs":
            # Special handling for pdfs
            for domain in ["cs", "econ", "eess", "math", "physics", "q-bio", "q-fin", "stat"]:
                pdfs_dir = os.path.join("surveys", domain)
                if not os.path.exists(pdfs_dir):
                    continue
                    
                for topic in os.listdir(pdfs_dir):
                    pdfs_path = os.path.join(pdfs_dir, topic, "pdfs")
                    if not os.path.exists(pdfs_path):
                        continue
                        
                    for file in os.listdir(pdfs_path):
                        if file.endswith(".md"):
                            results_path = os.path.join(pdfs_path, f"results_{model}.json")
                            if not os.path.exists(results_path):
                                continue
                                
                            with open(results_path, "r", encoding="utf-8") as f:
                                result_data = json.load(f)
                            
                            # Get original scores
                            original_scores = [result_data.get(metric, 0) for metric in metrics]
                            results[system]["original"].append(original_scores)
                            
                            # Get similarity scores (for pdfs, these are the same as original scores)
                            sim_scores = original_scores.copy()
                            results[system]["sim"].append(sim_scores)
                            results[system]["sim_hp"].append(sim_scores)
        else:
            for topic_map in tasks[system]:
                for topic, rel_path in topic_map.items():
                    sys_path = os.path.join("surveys", rel_path)
                    if not os.path.exists(sys_path):
                        continue
                    
                    # Read results JSON
                    results_path = os.path.join(sys_path, f"results_{model}.json")
                    if not os.path.exists(results_path):
                        continue
                    
                    with open(results_path, "r", encoding="utf-8") as f:
                        result_data = json.load(f)
                    
                    # Get original scores
                    original_scores = [result_data.get(metric, 0) for metric in metrics]
                    results[system]["original"].append(original_scores)
                    
                    # Get similarity scores
                    sim_scores = [result_data.get(metric, 0) for metric in sim_metrics]
                    results[system]["sim"].append(sim_scores)
                    
                    # Get human-perfect similarity scores
                    sim_hp_scores = [result_data.get(metric, 0) for metric in sim_hp_metrics]
                    results[system]["sim_hp"].append(sim_hp_scores)
    
    # Calculate averages for each system
    for system in results:
        if results[system]["original"]:
            results[system]["original_avg"] = [sum(scores) / len(scores) for scores in zip(*results[system]["original"])]
            results[system]["sim_avg"] = [sum(scores) / len(scores) for scores in zip(*results[system]["sim"])]
            results[system]["sim_hp_avg"] = [sum(scores) / len(scores) for scores in zip(*results[system]["sim_hp"])]
            
            # Calculate normalized average for specific metrics
            normalized_metrics = ["Outline_structure", "Faithfulness", "Reference_quality"]
            normalized_indices = [metrics.index(metric) for metric in normalized_metrics]
            
            # Calculate average of normalized scores for original
            normalized_scores = []
            for idx in normalized_indices:
                normalized_scores.append(normalize_score(results[system]["original_avg"][idx]))
            results[system]["normalized_avg"] = sum(normalized_scores) / len(normalized_scores)
            
            # Calculate average of normalized scores for similarity
            sim_normalized_scores = []
            for idx in normalized_indices:
                sim_normalized_scores.append(normalize_score(results[system]["sim_avg"][idx]))
            results[system]["sim_normalized_avg"] = sum(sim_normalized_scores) / len(sim_normalized_scores)
            
            # Calculate average of normalized scores for human-perfect similarity
            sim_hp_normalized_scores = []
            for idx in normalized_indices:
                sim_hp_normalized_scores.append(normalize_score(results[system]["sim_hp_avg"][idx]))
            results[system]["sim_hp_normalized_avg"] = sum(sim_hp_normalized_scores) / len(sim_hp_normalized_scores)
    
    # Generate tex content
    tex_content = []
    
    # Add results for each system
    for system in system_list:
        if system not in results or not results[system]["original"]:
            continue
            
        # Original scores with normalized format
        original_scores = []
        for i, score in enumerate(results[system]["original_avg"]):
            if metrics[i] in ["Outline_structure", "Faithfulness", "Reference_quality"]:
                normalized = normalize_score(score)
                original_scores.append(f"${normalized:.2f}_{{{score:.2f}}}$")
            else:
                original_scores.append(f"{score:.2f}")
        
        # Add SGEval label before normalized average
        tex_content.append(f"\\textbf{{{system}}} & \\textit{{SGEval}} & {results[system]['normalized_avg']:.2f} & {' & '.join(original_scores)} \\\\")
        
        # Similarity scores with normalized format
        sim_scores = []
        for i, score in enumerate(results[system]["sim_avg"]):
            if sim_metrics[i] in ["Outline_structure_sim", "Faithfulness_sim", "Reference_quality_sim"]:
                normalized = normalize_score(score)
                sim_scores.append(f"${normalized:.2f}_{{{score:.2f}}}$")
            else:
                sim_scores.append(f"{score:.2f}")
        
        tex_content.append(f"& \\textit{{SGEval-Sim}} & {results[system]['sim_normalized_avg']:.2f} & {' & '.join(sim_scores)} \\\\")
        
        # Human-perfect similarity scores with normalized format
        sim_hp_scores = []
        for i, score in enumerate(results[system]["sim_hp_avg"]):
            if sim_hp_metrics[i] in ["Outline_structure_sim_hp", "Faithfulness_sim_hp", "Reference_quality_sim_hp"]:
                normalized = normalize_score(score)
                sim_hp_scores.append(f"${normalized:.2f}_{{{score:.2f}}}$")
            else:
                sim_hp_scores.append(f"{score:.2f}")
        
        tex_content.append(f"& \\textit{{SGEval-Sim-HP}} & {results[system]['sim_hp_normalized_avg']:.2f} & {' & '.join(sim_hp_scores)} \\\\")
        
        if system != system_list[-1]:
            tex_content.append("\\midrule")
    
    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_content))
    
    print(f"Generated tex table at {output_path}")

def generate_evaluation_csv(system_list: List[str], model: str, output_path: str = "evaluation_results.csv") -> None:
    """
    Generate a CSV table with evaluation results for all systems.
    
    Args:
        system_list (List[str]): List of system names
        model (str): Model name used for evaluation
        output_path (str): Path to save the csv file
    """
    # Define the metrics in order
    metrics = [
        "Outline",
        "Outline_structure",
        "Coverage",
        "Structure",
        "Relevance",
        "Language",
        "Criticalness",
        "Faithfulness",
        "Reference",
        "Reference_quality"
    ]
    
    # Define the corresponding similarity metrics
    sim_metrics = [
        "Outline_sim",
        "Outline_structure_sim",
        "Coverage_sim",
        "Structure_sim",
        "Relevance_sim",
        "Language_sim",
        "Criticalness_sim",
        "Faithfulness_sim",
        "Reference_sim",
        "Reference_quality_sim"
    ]
    
    # Define the corresponding human-perfect similarity metrics
    sim_hp_metrics = [
        "Outline_sim_hp",
        "Outline_structure_sim_hp",
        "Coverage_sim_hp",
        "Structure_sim_hp",
        "Relevance_sim_hp",
        "Language_sim_hp",
        "Criticalness_sim_hp",
        "Faithfulness_sim_hp",
        "Reference_sim_hp",
        "Reference_quality_sim_hp"
    ]
    
    # Read tasks.json to get all paths
    with open("surveys/tasks.json", 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    
    # Initialize results dictionary
    results = {system: {"original": [], "sim": [], "sim_hp": []} for system in system_list}
    
    # Collect results for each system
    for system in system_list:
        if system not in tasks and system != "pdfs":
            print(f"System {system} not found in tasks.json")
            continue
            
        if system == "pdfs":
            # Special handling for pdfs
            for domain in ["cs", "econ", "eess", "math", "physics", "q-bio", "q-fin", "stat"]:
                pdfs_dir = os.path.join("surveys", domain)
                if not os.path.exists(pdfs_dir):
                    continue
                    
                for topic in os.listdir(pdfs_dir):
                    pdfs_path = os.path.join(pdfs_dir, topic, "pdfs")
                    if not os.path.exists(pdfs_path):
                        continue
                        
                    for file in os.listdir(pdfs_path):
                        if file.endswith(".md"):
                            results_path = os.path.join(pdfs_path, f"results_{model}.json")
                            if not os.path.exists(results_path):
                                continue
                                
                            with open(results_path, "r", encoding="utf-8") as f:
                                result_data = json.load(f)
                            
                            # Get original scores
                            original_scores = [result_data.get(metric, 0) for metric in metrics]
                            results[system]["original"].append(original_scores)
                            
                            # Get similarity scores (for pdfs, these are the same as original scores)
                            sim_scores = original_scores.copy()
                            results[system]["sim"].append(sim_scores)
                            results[system]["sim_hp"].append(sim_scores)
        else:
            for topic_map in tasks[system]:
                for topic, rel_path in topic_map.items():
                    sys_path = os.path.join("surveys", rel_path)
                    if not os.path.exists(sys_path):
                        continue
                    
                    # Read results JSON
                    results_path = os.path.join(sys_path, f"results_{model}.json")
                    if not os.path.exists(results_path):
                        continue
                    
                    with open(results_path, "r", encoding="utf-8") as f:
                        result_data = json.load(f)
                    
                    # Get original scores
                    original_scores = [result_data.get(metric, 0) for metric in metrics]
                    results[system]["original"].append(original_scores)
                    
                    # Get similarity scores
                    sim_scores = [result_data.get(metric, 0) for metric in sim_metrics]
                    results[system]["sim"].append(sim_scores)
                    
                    # Get human-perfect similarity scores
                    sim_hp_scores = [result_data.get(metric, 0) for metric in sim_hp_metrics]
                    results[system]["sim_hp"].append(sim_hp_scores)
    
    # Calculate averages for each system
    for system in results:
        if results[system]["original"]:
            results[system]["original_avg"] = [sum(scores) / len(scores) for scores in zip(*results[system]["original"])]
            results[system]["sim_avg"] = [sum(scores) / len(scores) for scores in zip(*results[system]["sim"])]
            results[system]["sim_hp_avg"] = [sum(scores) / len(scores) for scores in zip(*results[system]["sim_hp"])]
            
            # Calculate normalized average for specific metrics
            normalized_metrics = ["Outline_structure", "Faithfulness", "Reference_quality"]
            normalized_indices = [metrics.index(metric) for metric in normalized_metrics]
            
            # Calculate average of normalized scores for original
            normalized_scores = []
            for idx in normalized_indices:
                normalized_scores.append(normalize_score(results[system]["original_avg"][idx]))
            results[system]["normalized_avg"] = sum(normalized_scores) / len(normalized_scores)
            
            # Calculate average of normalized scores for similarity
            sim_normalized_scores = []
            for idx in normalized_indices:
                sim_normalized_scores.append(normalize_score(results[system]["sim_avg"][idx]))
            results[system]["sim_normalized_avg"] = sum(sim_normalized_scores) / len(sim_normalized_scores)
            
            # Calculate average of normalized scores for human-perfect similarity
            sim_hp_normalized_scores = []
            for idx in normalized_indices:
                sim_hp_normalized_scores.append(normalize_score(results[system]["sim_hp_avg"][idx]))
            results[system]["sim_hp_normalized_avg"] = sum(sim_hp_normalized_scores) / len(sim_hp_normalized_scores)
    
    # Generate CSV content
    csv_content = []
    
    # Add header row
    header = ["System", "Type", "Normalized Average"] + metrics
    csv_content.append(",".join(header))
    
    # Add results for each system
    for system in system_list:
        if system not in results or not results[system]["original"]:
            continue
            
        # Original scores with normalization for specific metrics
        original_scores = []
        for i, score in enumerate(results[system]["original_avg"]):
            if metrics[i] in ["Outline_structure", "Faithfulness", "Reference_quality"]:
                normalized = normalize_score(score)
                original_scores.append(f"{normalized:.2f}")
            else:
                original_scores.append(f"{score:.2f}")
        csv_content.append(f"{system},SGEval,{results[system]['normalized_avg']:.2f}," + ",".join(original_scores))
        
        # Similarity scores with normalization for specific metrics
        sim_scores = []
        for i, score in enumerate(results[system]["sim_avg"]):
            if sim_metrics[i] in ["Outline_structure_sim", "Faithfulness_sim", "Reference_quality_sim"]:
                normalized = normalize_score(score)
                sim_scores.append(f"{normalized:.2f}")
            else:
                sim_scores.append(f"{score:.2f}")
        csv_content.append(f"{system},SGEval-Sim,{results[system]['sim_normalized_avg']:.2f}," + ",".join(sim_scores))
        
        # Human-perfect similarity scores with normalization for specific metrics
        sim_hp_scores = []
        for i, score in enumerate(results[system]["sim_hp_avg"]):
            if sim_hp_metrics[i] in ["Outline_structure_sim_hp", "Faithfulness_sim_hp", "Reference_quality_sim_hp"]:
                normalized = normalize_score(score)
                sim_hp_scores.append(f"{normalized:.2f}")
            else:
                sim_hp_scores.append(f"{score:.2f}")
        csv_content.append(f"{system},SGEval-Sim-HP,{results[system]['sim_hp_normalized_avg']:.2f}," + ",".join(sim_hp_scores))
    
    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_content))
    
    print(f"Generated CSV table at {output_path}")

def generate_radar_plots(system_list: List[str], metrics: List[str], system_name_map: dict, output_path: str = "radar_plots.pdf") -> None:
    """
    Generate radar plots for SGEval, SGEval-Sim, and SGEval-Sim-HP results in a 1x3 layout.
    Combines Coverage, Structure, Relevance, Language, and Criticalness into a single Content metric.
    
    Args:
        system_list (List[str]): List of system names to plot
        metrics (List[str]): List of metrics to plot
        system_name_map (dict): Dictionary mapping system names to display names
        output_path (str): Path to save the output figure
    """
    # Read the CSV file
    df = pd.read_csv("evaluation_results.csv")
    
    # Define content metrics to combine
    content_metrics = ["Coverage", "Structure", "Relevance", "Language", "Criticalness"]
    
    # Create new dataframe with combined Content metric
    df_new = pd.DataFrame()
    
    # Process each system and type
    for system in system_list:
        for eval_type in ["SGEval", "SGEval-Sim", "SGEval-Sim-HP"]:
            df_filtered = df[(df["System"] == system) & (df["Type"] == eval_type)]
            if not df_filtered.empty:
                # Calculate average of content metrics
                content_avg = df_filtered[content_metrics].mean(axis=1).iloc[0]
                
                # Create new row with combined metrics
                new_row = df_filtered[["System", "Type", "Normalized Average"]].iloc[0].copy()
                new_row["Content"] = content_avg
                
                # Add other metrics
                for metric in metrics:
                    if metric not in content_metrics:
                        new_row[metric] = df_filtered[metric].iloc[0]
                
                df_new = pd.concat([df_new, pd.DataFrame([new_row])], ignore_index=True)
    
    # Define new metrics list with Content instead of individual content metrics
    new_metrics = ["Outline", "Structure", "Content", "Faithfulness", "Reference", "Supportiveness"]
    
    # Define metric name mapping (CSV column name -> Display name)
    metric_name_map = {
        "Outline": "Outline",
        "Outline_structure": "Structure",
        "Content": "Content",
        "Faithfulness": "Faithfulness",
        "Reference": "Reference",
        "Reference_quality": "Supportiveness"
    }
    
    # Create the figure with three subplots
    N = len(new_metrics)  # Number of axes in each radar plot
    theta = radar_factory(N, frame='polygon')
    
    fig, axs = plt.subplots(1, 3, figsize=(20, 7),
                           subplot_kw=dict(projection='radar'))
    
    # Generate colors for systems
    colors = plt.cm.Set2(np.linspace(0, 1, len(system_list)))
    color_dict = dict(zip(system_list, colors))
    
    # Left plot: SGEval
    ax1 = axs[0]
    ax1.set_rgrids([1,2,3,4,5])
    ax1.set_ylim(1.0, 5.0)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    
    for system in system_list:
        df_filtered = df_new[(df_new["System"] == system) & (df_new["Type"] == "SGEval")]
        if not df_filtered.empty:
            # Map values to new metrics
            values = []
            for metric in new_metrics:
                if metric == "Structure":
                    values.append(df_filtered["Outline_structure"].iloc[0])
                elif metric == "Supportiveness":
                    values.append(df_filtered["Reference_quality"].iloc[0])
                else:
                    values.append(df_filtered[metric].iloc[0])
            
            display_name = system_name_map[system]
            linewidth = 3 if system == "pdfs" else 3
            ax1.plot(theta, values, color=color_dict[system], label=display_name, linewidth=linewidth)
            ax1.fill(theta, values, facecolor=color_dict[system], alpha=0.15)
    
    ax1.set_varlabels(new_metrics)
    ax1.set_title("Vanilla", weight='bold', size='medium')
    
    # Middle plot: SGEval-Sim
    ax2 = axs[1]
    ax2.set_rgrids([1,2,3,4,5])
    ax2.set_ylim(1.0, 5.0)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    
    for system in system_list:
        df_filtered = df_new[(df_new["System"] == system) & (df_new["Type"] == "SGEval-Sim")]
        if not df_filtered.empty:
            # Map values to new metrics
            values = []
            for metric in new_metrics:
                if metric == "Structure":
                    values.append(df_filtered["Outline_structure"].iloc[0])
                elif metric == "Supportiveness":
                    values.append(df_filtered["Reference_quality"].iloc[0])
                else:
                    values.append(df_filtered[metric].iloc[0])
            
            display_name = system_name_map[system]
            linewidth = 3 if system == "pdfs" else 3
            ax2.plot(theta, values, color=color_dict[system], label=display_name, linewidth=linewidth)
            ax2.fill(theta, values, facecolor=color_dict[system], alpha=0.15)
    
    ax2.set_varlabels(new_metrics)
    ax2.set_title("SGSimEval-B", weight='bold', size='medium')
    
    # Right plot: SGEval-Sim-HP
    ax3 = axs[2]
    ax3.set_rgrids([1,2,3,4,5])
    ax3.set_ylim(1.0, 5.0)
    ax3.tick_params(axis='both', which='major', labelsize=14)
    
    for system in system_list:
        df_filtered = df_new[(df_new["System"] == system) & (df_new["Type"] == "SGEval-Sim-HP")]
        if not df_filtered.empty:
            # Map values to new metrics
            values = []
            for metric in new_metrics:
                if metric == "Structure":
                    values.append(df_filtered["Outline_structure"].iloc[0])
                elif metric == "Supportiveness":
                    values.append(df_filtered["Reference_quality"].iloc[0])
                else:
                    values.append(df_filtered[metric].iloc[0])
            
            display_name = system_name_map[system]
            linewidth = 3 if system == "pdfs" else 3
            ax3.plot(theta, values, color=color_dict[system], label=display_name, linewidth=linewidth)
            ax3.fill(theta, values, facecolor=color_dict[system], alpha=0.15)
    
    ax3.set_varlabels(new_metrics)
    ax3.set_title("SGSimEval-HP", weight='bold', size='medium')
    
    # Add a single legend for all systems
    handles, labels = [], []
    for system in system_list:
        display_name = system_name_map[system]
        handles.append(plt.Line2D([0], [0], color=color_dict[system], label=display_name))
        labels.append(display_name)
    
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.98, 0.5), fontsize=12)
    
    # Adjust layout
    fig.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    # Save figure
    fig.savefig(output_path, format="pdf", dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"Radar plots saved to: {output_path}")

if __name__ == "__main__":
    # Example usage
    systems = ["AutoSurvey", "SurveyForge", "InteractiveSurvey", "LLMxMapReduce", "SurveyX", "pdfs"]
    model = "qwen-plus-latest"
    domains = ["cs", "econ", "eess", "math", "physics", "q-bio", "q-fin", "stat"]  # Example domains
    
    # Example: Clear similarity for a specific system in a domain
    # clear_similarity("cs", "AutoSurvey")
    
    # Example: Clear all similarity files
    # clear_all_similarity()
    
    # batch_calculate_similarity_scores(systems, model, domains)
    # calculate_and_update_similarity_scores_new(model, "surveys/cs/Data-Driven Co-Speech Gesture Generation/SurveyForge/Data-Driven Co-Speech Gesture Generation.md", "surveys/cs/Data-Driven Co-Speech Gesture Generation/pdfs/2301.05339.md")
    
    # Generate tex table
    # generate_evaluation_tex(systems, model)

    # Generate CSV table
    # generate_evaluation_csv(systems, model)

    # Generate radar plots
    systems = ["InteractiveSurvey", "SurveyX", "LLMxMapReduce"]
    systems = ["AutoSurvey", "SurveyForge"]
    system_name_map = {
        "AutoSurvey": "AutoSurvey",
        "InteractiveSurvey": "InteractiveSurvey",
        "SurveyForge": "SurveyForge",
        "SurveyX": "SurveyX",
        "LLMxMapReduce": "LLMxMapReduce",
        "pdfs": "Human"
    }
    metrics = [
        "Outline",
        "Outline_structure",
        "Coverage",
        "Structure",
        "Relevance",
        "Language",
        "Criticalness",
        "Faithfulness",
        "Reference",
        "Reference_quality"
    ]
    generate_radar_plots(systems, metrics, system_name_map, "figures/radar_plots.pdf")

    # Generate domain averages table
    # generate_domain_average_tex(systems, model)
