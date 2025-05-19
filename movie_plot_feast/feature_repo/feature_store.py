# movie_plot_feast/feature_repo/feature_store.py

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, String
import os

# Dynamically set ROOT_DIR
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define Entity with value_type to avoid deprecation warning
movie_entity = Entity(
    name="movie",  # Changed to match what's expected in materialization
    description="A unique identifier for movies",
    value_type=ValueType.STRING  # Add value_type to resolve deprecation warning
)

# Define File Source
movie_features_source = FileSource(
    path=os.path.join(ROOT_DIR, "movie_plot_feast", "feature_repo", "data", "movie_features.parquet"),
    timestamp_field="event_timestamp",
)

# Define Feature View - renamed to match references
movie_features_view = FeatureView(
    name="movie_features",  # Changed to match what's expected in materialization
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