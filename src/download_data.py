import pandas as pd
import urllib.request
import os


def fetch_heart_disease_data(output_path="data/raw/heart.csv"):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Cleveland dataset column names
    columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
    ]

    print(f"Downloading dataset from {url}...")
    urllib.request.urlretrieve(url, output_path)

    # Read and clean structural missing values denoted by '?'
    df = pd.read_csv(output_path, names=columns, na_values="?")

    # Target modification for binary classification (0 = no disease, >0 = disease)
    df['target'] = (df['target'] > 0).astype(int)

    # Handle missing entries deterministically for baseline
    df = df.fillna(df.median())

    df.to_csv(output_path, index=False)
    print(f"Dataset securely saved to {output_path}")


if __name__ == "__main__":
    fetch_heart_disease_data()