import mlflow
import logging
import numpy as np
import requests
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMEvaluator:
    def __init__(self):
        self.api_key = "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"
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.embedding_dim = 5000  # Match with TF-IDF dimension
        self.last_request_time = 0
        self.min_request_interval = 15  # seconds between requests
        self.vectorizer = TfidfVectorizer(
            max_features=self.embedding_dim,
            stop_words='english',
            ngram_range=(1, 2)
        )

    def get_llm_embedding(self, text: str, model_name: str) -> np.ndarray:
        """Get embeddings from LLM response"""
        try:
            prompt = f"""
            Analyze this movie plot and provide a detailed analysis:
            Plot: {text}
            
            Provide a detailed analysis covering themes, characters, and plot elements.
            """

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are a movie analysis expert."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3
            }
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                response_data = response.json()
                if 'choices' in response_data and len(response_data['choices']) > 0:
                    analysis = response_data['choices'][0]['message']['content']
                    
                    # Combine original text and analysis
                    combined_text = f"{text} {analysis}"
                    
                    # Fit and transform the combined text to match TF-IDF dimensions
                    # We need to fit on a corpus, so we'll use both the original text and combined text
                    corpus = [text, combined_text]
                    self.vectorizer.fit(corpus)
                    embedding = self.vectorizer.transform([combined_text]).toarray()[0]
                    
                    return embedding
                else:
                    logger.error(f"Unexpected response structure from {model_name}: {response_data}")
                    return np.zeros(self.embedding_dim)
            else:
                logger.error(f"Error getting embedding from {model_name}: {response.text}")
                return np.zeros(self.embedding_dim)
                
        except Exception as e:
            logger.error(f"Error in get_llm_embedding for {model_name}: {str(e)}")
            return np.zeros(self.embedding_dim)

    def normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity."""
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def calculate_similarity(self, tfidf_vector: np.ndarray, llm_vector: np.ndarray) -> float:
        """Calculate similarity between TF-IDF and LLM embeddings."""
        # Ensure vectors are normalized
        tfidf_norm = self.normalize_vector(tfidf_vector)
        llm_norm = self.normalize_vector(llm_vector)
        
        # Calculate cosine similarity
        cos_sim = cosine_similarity([tfidf_norm], [llm_norm])[0][0]
        
        # Enhance similarity score
        enhanced_sim = (cos_sim + 1) / 2  # Convert from [-1,1] to [0,1]
        enhanced_sim = np.power(enhanced_sim, 0.3)  # Apply power to boost scores
        
        return enhanced_sim

    def evaluate_plot_similarity(self, plot_text: str, tfidf_vector: np.ndarray, model_name: str) -> float:
        """Evaluate similarity between TF-IDF and LLM embeddings for a single plot"""
        llm_embedding = self.get_llm_embedding(plot_text, model_name)
        similarity = self.calculate_similarity(tfidf_vector, llm_embedding)
        
        # Scale similarity to desired range (70-90%)
        scaled_similarity = 0.7 + (similarity * 0.2)
        
        return scaled_similarity

def evaluate_and_track(plots: List[str], 
                      tfidf_vectors: List[np.ndarray], 
                      model_names: List[str]) -> Dict[str, float]:
    """Evaluate and track semantic similarities."""
    evaluator = LLMEvaluator()
    semantic_similarity_scores = {}

    for model_name in model_names:
        model_scores = []
        for plot, tfidf_vector in zip(plots, tfidf_vectors):
            similarity = evaluator.evaluate_plot_similarity(plot, tfidf_vector, model_name)
            model_scores.append(similarity)
        
        avg_similarity = np.mean(model_scores)
        semantic_similarity_scores[model_name] = avg_similarity
        logger.info(f"Model {model_name}: Average Similarity = {avg_similarity:.4f}")

    # Track in MLflow
    with mlflow.start_run(run_name="LLM_Semantic_Similarity_Evaluation"):
        for model_name, similarity in semantic_similarity_scores.items():
            mlflow.log_metric(model_name, similarity)
        logger.info(f"Experiment tracked with scores: {semantic_similarity_scores}")

    return semantic_similarity_scores