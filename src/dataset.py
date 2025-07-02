"""
Utilities to load Plant-Pathology 2020 FGVC7.
If data is present in DATA_RAW we just read it.
"""
from pathlib import Path
import pandas as pd
from .config import DATA_RAW

def prepare_dataframe() -> pd.DataFrame:
    """
    Returns a DataFrame with columns: filepath, label
    (label is one of healthy/multiple_diseases/rust/scab)
    """
    df = pd.read_csv(DATA_RAW / "train.csv")
    label_cols = df.columns[1:]            # 4 one-hot columns
    df["label"] = df[label_cols].idxmax(axis=1)
    df["filepath"] = df["image_id"].apply(
        lambda x: str(DATA_RAW / "images" / f"{x}.jpg"))
    return df[["filepath", "label"]]

if __name__ == "__main__":
    print(prepare_dataframe().head())
