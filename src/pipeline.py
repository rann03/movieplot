import sys
import os
import uuid
import subprocess
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from feast import ValueType
import pandas as pd
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from zenml.pipelines import pipeline
from zenml.steps import step
from zenml.client import Client

try:
    from feast import FeatureStore, Entity, FeatureView, Field
    from feast.types import Float32, String
    from feast.infra.offline_stores.file_source import FileSource
except ImportError:
    logging.error("Feast or its dependencies not installed.")
    FeatureStore = Entity = FeatureView = Field = None

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

# Feast Entity and Feature View Definition
def create_feast_resources():
    """Create Feast entity and feature view."""
    if not all([FeatureStore, Entity, FeatureView, Field]):
        raise ImportError("Feast resources cannot be created")

    # Define Movie Entity
    movie_entity = Entity(
        name="movie_entity", 
        description="Movie identifier entity"
    )

    # Define File Source
    movie_features_source = FileSource(
        path=FEAST_DATA_PATH,
        timestamp_field="event_timestamp",
    )

    # Define Feature View
    movie_features_view = FeatureView(
        name="movie_features_view",
        entities=[movie_entity],
        schema=[
            Field(name="movie_id", dtype=String),
            Field(name="event_timestamp", dtype=String),
            Field(name="tfidf_1", dtype=Float32),
            Field(name="tfidf_2", dtype=Float32),
            Field(name="Plot", dtype=String)
        ],
        source=movie_features_source,
        online=True,
    )

    return movie_entity, movie_features_view

@step(enable_cache=False)
def ingest_data_step() -> pd.DataFrame:
    """Ingest raw movie data."""
    try:
        df = ingest_data(RAW_DATA_CSV)
        logger.info(f"Successfully ingested data. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        raise

@step(enable_cache=False)
def validate_data_step(df: pd.DataFrame) -> pd.DataFrame:
    """Validate ingested data."""
    try:
        df_valid, _ = validate_data(df)
        logger.info(f"Data validation complete. Remaining rows: {len(df_valid)}")
        return df_valid
    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        raise

@step(enable_cache=False)
def preprocess_data_step(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess validated data."""
    try:
        df_prep = preprocess_data(df)
        logger.info(f"Preprocessing complete. Shape: {df_prep.shape}")
        return df_prep
    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}")
        raise

@step(enable_cache=False)
def version_data_step(df: pd.DataFrame) -> Optional[str]:
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
def create_feast_resources(features_df: pd.DataFrame):
    """Create Feast entity and feature view."""
    if not all([FeatureStore, Entity, FeatureView, Field]):
        raise ImportError("Feast resources cannot be created")

    # Define Movie Entity
    movie_entity = Entity(
        name="movie_entity", 
        description="Movie identifier entity",
        # Explicitly set value_type to address deprecation warning
        value_type=ValueType.STRING
    )

    # Define File Source
    movie_features_source = FileSource(
        path=FEAST_DATA_PATH,
        timestamp_field="event_timestamp",
    )

    # Define Feature View
    movie_features_view = FeatureView(
        name="movie_features_view",
        entities=[movie_entity],
        schema=[
            Field(name="movie_id", dtype=String),
            Field(name="event_timestamp", dtype=String),
            Field(name="tfidf_1", dtype=Float32),
            Field(name="tfidf_2", dtype=Float32),
            Field(name="Plot", dtype=String)
        ],
        source=movie_features_source,
        online=True,
        ttl=None
    )

    return movie_entity, movie_features_view

@step(enable_cache=False)
def feast_apply_and_materialize_step(df: pd.DataFrame) -> Dict[str, Any]:
    """Apply and materialize Feast feature store."""
    try:
        # Prepare features DataFrame with explicit timestamp
        features_df = pd.DataFrame({
            "movie_id": df["movie_id"],
            "event_timestamp": pd.Timestamp.utcnow(),  # Use pandas Timestamp
            "tfidf_1": df.get("tfidf_1", 0),
            "tfidf_2": df.get("tfidf_2", 0),
            "Plot": df["Plot"]
        })

        # Save to Parquet
        features_df.to_parquet(FEAST_DATA_PATH, index=False)

        # Create Feast resources
        movie_entity, movie_features_view = create_feast_resources(features_df)

        # Initialize Feast store with explicit project name
        store = FeatureStore(
            repo_path=FEAST_REPO_DIR,
            project="movie_plot_retrieval"  # Add explicit project name
        )

        # Debug: Print registered entities before applying
        print("Current registered entities:", store.list_entities())

        # Apply entities and feature views
        try:
            store.apply([movie_entity, movie_features_view])
        except Exception as apply_error:
            logger.error(f"Failed to apply Feast resources: {apply_error}")
            raise

        # Materialize features
        end_time = datetime.utcnow()
        try:
            store.materialize_incremental(end_time)
        except Exception as materialize_error:
            logger.error(f"Failed to materialize features: {materialize_error}")
            raise

        logger.info("Feast feature store registered and materialized")
        
        # Return a dictionary instead of a DataFrame
        return {
            "movie_ids": features_df["movie_id"].tolist(),
            "tfidf_1": features_df["tfidf_1"].tolist(),
            "tfidf_2": features_df["tfidf_2"].tolist(),
            "plots": features_df["Plot"].tolist()
        }
    except Exception as e:
        logger.error(f"Feast operations failed: {e}")
        raise

@pipeline
def movie_plot_ml_pipeline():
    """Main ML pipeline for movie plot features."""
    df_raw = ingest_data_step()
    df_valid = validate_data_step(df_raw)
    df_preprocessed = preprocess_data_step(df_valid)
    _ = version_data_step(df_preprocessed)
    df_featured = create_feast_resources(df_preprocessed)
    feast_df = feast_apply_and_materialize_step(df_featured)

if __name__ == "__main__":
    client = Client()
    pipeline_instance = movie_plot_ml_pipeline()
    run = pipeline_instance
    print(f"Started pipeline run: {run}")
