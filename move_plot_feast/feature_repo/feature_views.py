from feast import Entity, Feature, FeatureView, FileSource, ValueType
import os

# Define entity
movie = Entity(name="movie_id", value_type=ValueType.STRING, description="Unique movie id")

# Define your data source
data_dir = os.path.join(os.path.dirname(__file__), "data")
movie_features_source = FileSource(
    path=os.path.join(data_dir, "movie_features.parquet"),
    event_timestamp_column="event_timestamp"
)

# Define the features you want to track
movie_features_view = FeatureView(
    name="movie_features",
    entities=["movie_id"],
    ttl=None,
    schema=[
        Feature(name="tfidf_1", dtype=ValueType.FLOAT),  # Changed from FLOAT64 to FLOAT
        Feature(name="tfidf_2", dtype=ValueType.FLOAT),  # Changed from FLOAT64 to FLOAT
        # Add more feature columns here as needed
    ],
    source=movie_features_source,
    online=True,
)