import pandas as pd
import os


def load_dataset(file_path):
    """
    Load the full landmark feature dataset.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path)
    print(f"Loaded dataset from {file_path}")
    print(f"Dataset shape: {df.shape}")
    return df
