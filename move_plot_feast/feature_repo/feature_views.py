# feature_views.py
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, String
import os
from datetime import datetime

# Define entity
movie = Entity(name="movie_id", value_type=ValueType.STRING, description="Unique movie id")

# Define your data source
data_dir = os.path.join(os.path.dirname(__file__), "data")
movie_features_source = FileSource(
    path=os.path.join(data_dir, "movie_features.parquet"),
    event_timestamp_column="event_timestamp",
    # Remove created_timestamp_column or use a different column
)

# Define feature view
movie_features_view = FeatureView(
    name="movie_features",
    entities=[movie],
    ttl=None,
    schema=[
        Field(name="tfidf_1", dtype=Float32),
        Field(name="tfidf_2", dtype=Float32),
    ],
    source=movie_features_source,
    online=True,
)