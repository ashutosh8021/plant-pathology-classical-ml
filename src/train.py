"""
Train SVM, Random-Forest, Gradient-Boosting; save the best.
"""
import json, joblib
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from .config   import DATA_PROCESSED, MODELS_DIR, RANDOM_STATE
from .dataset  import prepare_dataframe
from .features import extract_features

def main():
    df = prepare_dataframe()
    X = extract_features(df["filepath"])
    y = df["label"].values

    # split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    # models
    experiments = {
        "svm": Pipeline([
            ("scale", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced"))
        ]),
        "rf": RandomForestClassifier(
            n_estimators=500, class_weight="balanced",
            random_state=RANDOM_STATE),
        "gb": GradientBoostingClassifier(random_state=RANDOM_STATE)
    }

    MODELS_DIR.mkdir(exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    np.save(DATA_PROCESSED / "features.npy", X)      # optional caching

    scores = {}
    for name, model in experiments.items():
        print(f"\n── {name.upper()} ─────────────────────")
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        report = classification_report(y_val, preds, digits=3)
        print(report)
        print("Confusion matrix:\n", confusion_matrix(y_val, preds))
        f1 = classification_report(y_val, preds, output_dict=True)["macro avg"]["f1-score"]
        scores[name] = f1
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")

    best = max(scores, key=scores.get)
    print(f"\nBest = {best}  (macro-F1={scores[best]:.3f})")
    (MODELS_DIR / "best_model.joblib").write_bytes(
        (MODELS_DIR / f"{best}.joblib").read_bytes())
    json.dump(scores, open(MODELS_DIR / "results.json", "w"), indent=2)

if __name__ == "__main__":
    main()
