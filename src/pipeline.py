import sys
import os
import subprocess
import pandas as pd
from zenml.pipelines import pipeline
from zenml.steps import step
from zenml.client import Client

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

RAW_DATA_CSV = os.path.join(ROOT_DIR, "data", "wiki_movie_plots.csv")
FEAST_REPO_DIR = os.path.join(ROOT_DIR, "move_plot_feast", "feature_repo")
FEAST_DATA_PATH = os.path.join(FEAST_REPO_DIR, "data", "movie_features.parquet")

from ingestion import ingest_data
from preprocessing import preprocess_data
from validation import validate_data
from versioning import save_versioned_csv

@step(enable_cache=False)
def ingest_data_step() -> pd.DataFrame:
    df = ingest_data(RAW_DATA_CSV)
    return df

@step(enable_cache=False)
def validate_data_step(df: pd.DataFrame) -> pd.DataFrame:
    df_valid, _ = validate_data(df)  
    return df_valid

@step(enable_cache=False)
def preprocess_data_step(df: pd.DataFrame) -> pd.DataFrame:
    df_prep = preprocess_data(df)
    return df_prep

@step(enable_cache=False)
def version_data_step(df: pd.DataFrame) -> str:
    save_path = save_versioned_csv(df)
    try:
        subprocess.run(["dvc", "add", save_path], check=True)
        subprocess.run(["git", "add", "-A"], check=True)
        subprocess.run(["git", "commit", "-m", "Track ingested & preprocessed movie plots data with DVC"], check=True)
        subprocess.run(["dvc", "push"], check=True)
        print(f"Successfully version tracked {save_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during DVC versioning: {e}")
    return save_path

@step(enable_cache=False)
def create_feature_parquet_step(df: pd.DataFrame) -> str:
    import numpy as np
    import uuid
    import datetime

    from sklearn.feature_extraction.text import TfidfVectorizer

    # Add movie_id if missing
    if "movie_id" not in df.columns:
        df = df.copy()
        df["movie_id"] = [str(uuid.uuid4()) for _ in range(len(df))]

    # Compute TF-IDF features (using only first 2 components for example)
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1,2))
    tfidf_matrix = vectorizer.fit_transform(df["Plot"])
    tfidf_dense = tfidf_matrix.toarray()

    # Build dataframe with movie_id, timestamp and feature columns
    features_df = pd.DataFrame({
        "movie_id": df["movie_id"],
        "event_timestamp": datetime.datetime.utcnow(),
        "tfidf_1": tfidf_dense[:, 0],
        "tfidf_2": tfidf_dense[:, 1],
    })

    os.makedirs(os.path.join(FEAST_REPO_DIR, "data"), exist_ok=True)
    features_df.to_parquet(FEAST_DATA_PATH, index=False)

    print(f"Created Feast feature parquet at: {FEAST_DATA_PATH}")
    return FEAST_DATA_PATH

@step(enable_cache=False)
def feast_apply_and_materialize_step():
    try:
        # Run feast apply to register feature definitions
        subprocess.run(["feast", "apply"], cwd=FEAST_REPO_DIR, check=True)

        # Materialize features for all available data (incremental)
        subprocess.run(["feast", "materialize-incremental"], cwd=FEAST_REPO_DIR, check=True)

        print("Feast feature store registered and materialized.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing Feast commands: {e}")

@pipeline
def movie_plot_ml_pipeline():
    df_raw = ingest_data_step()
    df_valid = validate_data_step(df_raw)
    df_preprocessed = preprocess_data_step(df_valid)
    _ = version_data_step(df_preprocessed)
    _ = create_feature_parquet_step(df_preprocessed)
    feast_apply_and_materialize_step()

if __name__ == "__main__":
    client = Client()
    pipeline_instance = movie_plot_ml_pipeline()
    run = pipeline_instance
    print(f"Started pipeline run: {run}")