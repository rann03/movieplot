from typing import Type, Any
import pandas as pd
from zenml.materializers.base_materializer import BaseMaterializer

class PandasMaterializer(BaseMaterializer):
    """Materializer for pandas DataFrames."""
    
    ASSOCIATED_TYPES = [pd.DataFrame]

    def save(self, data: pd.DataFrame) -> None:
        """Save the DataFrame."""
        data.to_parquet(self.uri)

    def load(self, data_type: Type[pd.DataFrame] = pd.DataFrame) -> pd.DataFrame:
        """Load the DataFrame."""
        return pd.read_parquet(self.uri)

    @classmethod
    def can_load_type(cls, type_: Type[Any]) -> bool:
        """Check if the materializer can load the given type."""
        return type_ == pd.DataFrame or type_ is pd.DataFrame