import pandas as pd

MIN_PLOT_LEN = 50
MAX_PLOT_LEN = 5000

def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    anomalies = {}

    missing_titles = df['Title'].isnull().sum() + (df['Title'].str.strip() == '').sum()
    if missing_titles > 0:
        anomalies['missing_titles'] = missing_titles

    short_plots = df[df['Plot'].str.len() < MIN_PLOT_LEN]
    long_plots = df[df['Plot'].str.len() > MAX_PLOT_LEN]
    anomalies['short_plots'] = len(short_plots)
    anomalies['long_plots'] = len(long_plots)

    duplicate_mask = df.duplicated(subset=['Title', 'Plot'])
    duplicates = df[duplicate_mask]
    anomalies['duplicates'] = len(duplicates)

    print("Data Validation Summary:")
    for k, v in anomalies.items():
        print(f"  {k}: {v}")

    df_clean = df.dropna(subset=['Title'])
    df_clean = df_clean[~duplicate_mask]
    df_clean = df_clean[(df_clean['Plot'].str.len() >= MIN_PLOT_LEN) & (df_clean['Plot'].str.len() <= MAX_PLOT_LEN)]

    print(f"Cleaned data size: {len(df_clean)} records (from {len(df)})")
    return df_clean.reset_index(drop=True), anomalies
