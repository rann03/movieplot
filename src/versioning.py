import os
import datetime
import subprocess

def save_versioned_csv(df, base_dir="data/ingested", base_name="movie_plots"):
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_name}_{timestamp}.csv"
    filepath = os.path.join(base_dir, filename)
    df.to_csv(filepath, index=False)
    return filepath

def dvc_track(filepath: str):
    try:
        subprocess.run(["dvc", "add", filepath], check=True)
        print(f"Tracked {filepath} with DVC.")
    except Exception as e:
        print(f"Failed to track {filepath} with DVC: {e}")


        