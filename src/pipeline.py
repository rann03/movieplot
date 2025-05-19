import os
import sys
import uuid
import subprocess
import logging
from datetime import datetime
from typing import List

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import mlflow

from zenml.pipelines import pipeline
from zenml.steps import step
from zenml.client import Client

try:
    from feast import FeatureStore, Entity, FeatureView, Field, ValueType
    from feast.types import Float32, String
    from feast.infra.offline_stores.file_source import FileSource
except ImportError:
    logging.error("Feast or its dependencies not installed.")
    FeatureStore = Entity = FeatureView = Field = ValueType = None

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure MLflow directory exists
tracking_dir = "mlruns"
if not os.path.exists(tracking_dir):
    os.makedirs(tracking_dir)

# MLflow Setup
mlflow.set_tracking_uri(tracking_dir)  # Set tracking URI to local directory path
mlflow.set_experiment("Movie Recommender Experiments")

# Path Configuration
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# Import local modules
from ingestion import ingest_data
from preprocessing import preprocess_data
from validation import validate_data
from versioning import save_versioned_csv

# Feast-specific paths
FEAST_REPO_DIR = os.path.join(ROOT_DIR, "movie_plot_feast", "feature_repo")
RAW_DATA_CSV = os.path.join(ROOT_DIR, "data", "wiki_movie_plots.csv")
FEAST_DATA_PATH = os.path.join(FEAST_REPO_DIR, "data", "movie_features.parquet")

# Ensure Feast directories exist
os.makedirs(os.path.join(FEAST_REPO_DIR, "data"), exist_ok=True)

@step
def ingest_data_step() -> pd.DataFrame:
    """Ingest raw movie data."""
    try:
        df = ingest_data(RAW_DATA_CSV)
        logger.info(f"Successfully ingested data. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        raise

@step
def validate_data_step(df: pd.DataFrame) -> pd.DataFrame:
    """Validate ingested data."""
    try:
        df_valid, _ = validate_data(df)
        logger.info(f"Data validation complete. Remaining rows: {len(df_valid)}")
        return df_valid
    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        raise

@step
def preprocess_data_step(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess validated data."""
    try:
        df_prep = preprocess_data(df)
        logger.info(f"Preprocessing complete. Shape: {df_prep.shape}")
        return df_prep
    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}")
        raise

@step
def version_data_step(df: pd.DataFrame) -> str:
    """Version data using DVC."""
    try:
        save_path = save_versioned_csv(df)
        
        # DVC and Git operations
        dvc_commands = [
            ["dvc", "add", save_path],
            ["git", "add", "-A"],
            ["git", "commit", "-m", "Track ingested & preprocessed movie plots data with DVC"],
            ["dvc", "push"]
        ]
        
        for cmd in dvc_commands:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Command {' '.join(cmd)} failed: {result.stderr}")
                raise subprocess.CalledProcessError(result.returncode, cmd, result.stderr)
        
        logger.info(f"Successfully version tracked {save_path}")
        return save_path
    except Exception as e:
        logger.error(f"Version tracking failed: {e}")
        raise

@step(enable_cache=False)
def feature_engineering_step(df: pd.DataFrame) -> dict:
    """Create TF-IDF features and materialize to Feast."""
    try:
        if "movie_id" not in df.columns:
            df = df.copy()
            df["movie_id"] = [str(uuid.uuid4()) for _ in range(len(df))]

        # TF-IDF Vectorization with fixed number of features
        vectorizer = TfidfVectorizer(
            max_features=5000,  # Match embedding dimension
            stop_words="english",
            ngram_range=(1,2)
        )
        tfidf_matrix = vectorizer.fit_transform(df["Plot"])
        tfidf_dense = tfidf_matrix.toarray()

        # Save vectorizer
        vectorizer_path = os.path.join(FEAST_REPO_DIR, "data", "tfidf_vectorizer.pkl")
        joblib.dump(vectorizer, vectorizer_path)
        
        return {
            "movie_ids": df["movie_id"].tolist()[:10],
            "plots": df["Plot"].tolist()[:10],
            "tfidf_vectors": tfidf_dense[:10].tolist()  # Convert to list for serialization
        }

    except Exception as e:
        logger.error(f"Feature engineering failed: {e}", exc_info=True)
        raise

@step(enable_cache=False)
def prepare_evaluation_data(df_preprocessed: pd.DataFrame) -> dict:
    """Prepare data for evaluation."""
    try:
        # Get TF-IDF vectors and plots
        vectorizer_path = os.path.join(FEAST_REPO_DIR, "data", "tfidf_vectorizer.pkl")
        vectorizer = joblib.load(vectorizer_path)
        
        # Get plots from DataFrame
        plots = df_preprocessed['Plot'].tolist()[:10]  # Taking first 10 plots
        
        # Generate TF-IDF vectors
        tfidf_vectors = vectorizer.transform(plots).toarray()
        
        return {
            "plots": plots,
            "tfidf_vectors": tfidf_vectors.tolist()
        }
    except Exception as e:
        logger.error(f"Error preparing evaluation data: {e}")
        raise

@step(enable_cache=False)
def evaluate_models(evaluation_data: dict) -> dict:
    """Evaluate models and generate similarity scores."""
    try:
        plots = evaluation_data["plots"]
        tfidf_vectors = evaluation_data["tfidf_vectors"]
        
        model_names = [
            "qwen/qwen3-4b:free", 
            "deepseek/deepseek-chat:free",
            "meta-llama/llama-4-maverick:free"
        ]

        # Import evaluation functions
        from evaluate import evaluate_and_track, LLMEvaluator
        
        # Get semantic similarity scores
        semantic_similarity_scores = evaluate_and_track(
            plots=plots,
            tfidf_vectors=tfidf_vectors,
            model_names=model_names
        )
        
        return semantic_similarity_scores
    except Exception as e:
        logger.error(f"Error in model evaluation: {e}")
        raise

@step(enable_cache=False)
def track_experiment_step(scores: dict) -> None:
    """Track experiment details using MLflow."""
    try:
        with mlflow.start_run(run_name="Semantic Similarity Experiment"):
            for model_name, similarity in scores.items():
                mlflow.log_metric(model_name, similarity)
            logger.info(f"Experiment tracked with scores: {scores}")
    except Exception as e:
        logger.error(f"Error tracking experiment: {e}")
        raise

@step(enable_cache=False)
def train_model_step(features: dict) -> str:
    """
    Placeholder for model training logic.
    Accepts TF-IDF features and returns a placeholder model path.
    """
    try:
        logger.info("Training step placeholder: Replace with actual training logic.")
        
        # Simulate model training output
        placeholder_model_path = os.path.join(FEAST_REPO_DIR, "data", "placeholder_model.pkl")
        
        # Save a placeholder object
        joblib.dump({"note": "This is a placeholder model object."}, placeholder_model_path)
        
        logger.info(f"Placeholder model saved at: {placeholder_model_path}")
        return placeholder_model_path
    except Exception as e:
        logger.error(f"Error in training placeholder step: {e}")
        raise

@pipeline
def movie_plot_ml_pipeline():
    """Main pipeline for movie plot processing and evaluation."""
    # Data processing steps
    df_raw = ingest_data_step()
    df_valid = validate_data_step(df_raw)
    df_preprocessed = preprocess_data_step(df_valid)
    version_path = version_data_step(df_preprocessed)
    features = feature_engineering_step(df_preprocessed)

    # Placeholder training step
    model_path = train_model_step(features)

    # Evaluation steps
    evaluation_data = prepare_evaluation_data(df_preprocessed)
    similarity_scores = evaluate_models(evaluation_data)
    
    # Track results
    track_experiment_step(similarity_scores)

    return features


if __name__ == "__main__":
    client = Client()
    pipeline_instance = movie_plot_ml_pipeline()
    run = pipeline_instance
    print(f"Started pipeline run: {run}")