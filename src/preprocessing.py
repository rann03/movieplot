import pandas as pd
import re

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Plot"] = df["Plot"].str.lower()
    df["Plot"] = df["Plot"].apply(lambda x: re.sub(r"\s+", " ", x).strip())
    df["Title"] = df["Title"].str.strip()
    if "Genre" in df.columns:
        df["Genre"] = df["Genre"].fillna("unknown").str.strip()
    return df.reset_index(drop=True)