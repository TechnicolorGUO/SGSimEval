"""
Similarity calculation module for SGSimEval.
Handles similarity metrics between generated and human-authored surveys.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import os
import json
import logging
from dataclasses import dataclass
from openai import OpenAI
import dotenv
from tqdm import tqdm

@dataclass
class SimilarityConfig:
    """Configuration for similarity calculation settings."""
    use_human_as_perfect: bool = True
    use_balanced_weighting: bool = True
    outline_weight: float = 0.3
    content_weight: float = 0.4
    reference_weight: float = 0.3
    batch_size: int = 10

class SimilarityCalculator:
    """Calculates similarity between surveys."""
    def __init__(self, config: Optional[SimilarityConfig] = None):
        self.config = config or SimilarityConfig()
        dotenv.load_dotenv()
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_API_BASE")
        )
        
        # Configure logging
        logging.basicConfig(
            filename='similarity.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def embed_text(self, text: str) -> Optional[List[float]]:
        """Get embedding for a text using OpenAI API."""
        try:
            response = self.client.embeddings.create(
                model=os.environ.get("MODEL"),
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logging.error(f"Error getting embedding: {e}")
            return None

    def embed_texts_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a batch of texts using OpenAI API."""
        try:
            embeddings = []
            for i in range(0, len(texts), self.config.batch_size):
                batch = texts[i:i + self.config.batch_size]
                response = self.client.embeddings.create(
                    model=os.environ.get("MODEL"),
                    input=batch
                )
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
            return embeddings
        except Exception as e:
            logging.error(f"Error getting batch embeddings: {e}")
            return []

    def calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        except Exception as e:
            logging.error(f"Error calculating cosine similarity: {e}")
            return 0.0

    def calculate_outline_similarity(self, gen_outline: Dict, ref_outline: Dict) -> float:
        """Calculate similarity between outlines."""
        try:
            # Convert outlines to strings
            gen_str = json.dumps(gen_outline, ensure_ascii=False)
            ref_str = json.dumps(ref_outline, ensure_ascii=False)
            
            # Get embeddings
            gen_embedding = self.embed_text(gen_str)
            ref_embedding = self.embed_text(ref_str)
            
            if gen_embedding and ref_embedding:
                return self.calculate_cosine_similarity(gen_embedding, ref_embedding)
            return 0.0
        except Exception as e:
            logging.error(f"Error calculating outline similarity: {e}")
            return 0.0

    def calculate_content_similarity(self, gen_content: str, ref_content: str) -> float:
        """Calculate similarity between content."""
        try:
            # Split content into sentences
            gen_sentences = [s.strip() for s in gen_content.split('.') if s.strip()]
            ref_sentences = [s.strip() for s in ref_content.split('.') if s.strip()]
            
            # Get embeddings for all sentences
            gen_embeddings = self.embed_texts_batch(gen_sentences)
            ref_embeddings = self.embed_texts_batch(ref_sentences)
            
            if not gen_embeddings or not ref_embeddings:
                return 0.0
            
            # Calculate pairwise similarities
            similarities = []
            for gen_emb in gen_embeddings:
                for ref_emb in ref_embeddings:
                    sim = self.calculate_cosine_similarity(gen_emb, ref_emb)
                    similarities.append(sim)
            
            # Return average similarity
            return float(np.mean(similarities)) if similarities else 0.0
        except Exception as e:
            logging.error(f"Error calculating content similarity: {e}")
            return 0.0

    def calculate_reference_similarity(self, gen_refs: List[str], ref_refs: List[str]) -> float:
        """Calculate similarity between references."""
        try:
            # Get embeddings for all references
            gen_embeddings = self.embed_texts_batch(gen_refs)
            ref_embeddings = self.embed_texts_batch(ref_refs)
            
            if not gen_embeddings or not ref_embeddings:
                return 0.0
            
            # Calculate pairwise similarities
            similarities = []
            for gen_emb in gen_embeddings:
                for ref_emb in ref_embeddings:
                    sim = self.calculate_cosine_similarity(gen_emb, ref_emb)
                    similarities.append(sim)
            
            # Return average similarity
            return float(np.mean(similarities)) if similarities else 0.0
        except Exception as e:
            logging.error(f"Error calculating reference similarity: {e}")
            return 0.0

    def calculate_overall_similarity(
        self,
        gen_survey: Dict,
        ref_survey: Dict
    ) -> Dict[str, float]:
        """Calculate overall similarity between surveys."""
        results = {}
        
        # Calculate individual similarities
        outline_sim = self.calculate_outline_similarity(
            gen_survey['outline'],
            ref_survey['outline']
        )
        content_sim = self.calculate_content_similarity(
            gen_survey['content'],
            ref_survey['content']
        )
        ref_sim = self.calculate_reference_similarity(
            gen_survey['references'],
            ref_survey['references']
        )
        
        # Calculate weighted average
        if self.config.use_balanced_weighting:
            overall_sim = (
                self.config.outline_weight * outline_sim +
                self.config.content_weight * content_sim +
                self.config.reference_weight * ref_sim
            )
        else:
            overall_sim = (outline_sim + content_sim + ref_sim) / 3
        
        results.update({
            'outline_similarity': outline_sim,
            'content_similarity': content_sim,
            'reference_similarity': ref_sim,
            'overall_similarity': overall_sim
        })
        
        return results

def calculate_similarity(
    gen_survey: Dict,
    ref_survey: Dict,
    config: Optional[SimilarityConfig] = None
) -> Dict[str, float]:
    """Convenience function to calculate similarity between surveys."""
    calculator = SimilarityCalculator(config)
    return calculator.calculate_overall_similarity(gen_survey, ref_survey) 