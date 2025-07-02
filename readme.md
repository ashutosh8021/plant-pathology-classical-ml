# Plant-Pathology 2020 FGVC7 – Classical Machine‑Learning Baseline

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Live&nbsp;Demo-brightgreen?logo=streamlit)

> **Live demo:** <https://plant-pathology-classical-ml.streamlit.app/>

This project is an **end‑to‑end classical ML pipeline** for the Plant‑Pathology 2020 dataset:

* ⬇️ Reads the Kaggle CSVs & JPEGs
* 🧑‍🔬 Extracts hand‑crafted colour / texture / shape features (529‑D)
* 🤖 Trains SVM, Random‑Forest, Gradient‑Boost — picks best macro‑F1
* 🌐 Ships a Streamlit app so anyone can upload a leaf photo & see a prediction

---

## Quick start

```bash
git clone https://github.com/your‑handle/plant-pathology-classical-ml.git
cd plant-pathology-classical-ml
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# copy dataset into data/raw/
python -m src.train           # 5‑10 min CPU
streamlit run app/app.py
```

Open <http://localhost:8501> and drag‑drop any JPEG from the images folder.

---

## Project structure

```
plant-pathology-classical-ml/
├─ app/                Streamlit UI
├─ src/                Feature + training code
├─ data/raw/           place CSVs & images here
├─ models/             .joblib files after training
├─ tests/              pytest sanity checks
└─ requirements.txt
```

---

## How it works

1. **Feature engineering**
   | type | dim | why |
   |------|-----|-----|
   | colour histogram | 512 | disease hues (rust ≈ orange) |
   | LBP texture      | 10  | micro‑patterns on leaves |
   | Hu moments       | 7   | lesion shapes |

2. **Models** – SVM, RF, GB; macro‑F1 used to choose winner.
3. **Inference** – extract features for uploaded image → predict_proba → bar chart.

---

## Troubleshooting

| problem | fix |
|---------|-----|
| `ModuleNotFoundError: cv2` | `pip install opencv-python-headless` |
| missing `Train_0.jpg`      | dataset not copied to `data/raw/images/` |
| Streamlit redirect loop    | in Cloud dashboard set **Main file** = `app/app.py` |
