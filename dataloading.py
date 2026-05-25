import pandas as pd

DATA_URL = "https://huggingface.co/datasets/electricsheepafrica/african-tuberculosis-dataset/raw/main/tb_ssa_large_5000.csv"


def load_data():
    """
    Loads the raw TB dataset from source.
    Returns a pandas DataFrame.
    """
    df = pd.read_csv(DATA_URL)

    print(f"Data loaded with shape: {df.shape}")
    return df


print(load_data())