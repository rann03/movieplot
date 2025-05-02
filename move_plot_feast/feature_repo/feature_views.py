from feast import Entity, FeatureView, Field, FileSource, ValueType
import os

movie = Entity(name="movie_id", value_type=ValueType.STRING, description="Unique movie id")

data_dir = os.path.join(os.path.dirname(__file__), "data")
movie_features_source = FileSource(
    path=os.path.join(data_dir, "movie_features.parquet"),
    event_timestamp_column="event_timestamp",
    created_timestamp_column="event_timestamp",  # Added this
)


movie_features_view = FeatureView(
    name="movie_features",
    entities=[movie],  # Pass the entity object, not string
    ttl=None,
    schema=[
        Field(name="tfidf_1", dtype=ValueType.FLOAT),  # Changed Feature to Field
        Field(name="tfidf_2", dtype=ValueType.FLOAT),  # Changed Feature to Field
    ],
    source=movie_features_source,
    online=True,
)