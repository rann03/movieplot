# In movie_plot_feast/feature_repo/feature_views.py

from feast import Field, FeatureView, Entity
from feast.types import Float32, String
from feast.infra.offline_stores.file_source import FileSource
import os

# Assuming you have a function to get the root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define the file source
movie_features_source = FileSource(
    path=os.path.join(ROOT_DIR, "movie_plot_feast", "feature_repo", "data", "movie_features.parquet"),
    timestamp_field="event_timestamp",
)

# Create an entity
movie_entity = Entity(name="movie")

# Modify the feature view definition
movie_features_view = FeatureView(
    name="movie_features",
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