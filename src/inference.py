import joblib, numpy as np
from pathlib import Path
from config    import MODELS_DIR
from features  import extract_features

LABELS = ["healthy", "multiple_diseases", "rust", "scab"]
_model = joblib.load(MODELS_DIR / "best_model.joblib")

def predict(image_path: str | Path):
    feats = extract_features([image_path])
    probs = _model.predict_proba(feats)[0]
    idx   = int(np.argmax(probs))
    return LABELS[idx], dict(zip(LABELS, probs.round(3)))
