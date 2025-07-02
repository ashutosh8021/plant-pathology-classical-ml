from pathlib import Path
from src.features import extract_features

def test_vector_size():
    sample = Path("data/raw/images/Train_0.jpg")
    if not sample.exists():
        return          # skip if dataset not present
    vec = extract_features([sample])
    assert vec.shape == (1, 529)
