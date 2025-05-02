import pandas as pd

def ingest_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df