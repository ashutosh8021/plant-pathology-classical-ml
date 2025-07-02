import streamlit as st
from PIL import Image
from pathlib import Path
import sys

# expose src to PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from inference import predict

st.set_page_config(page_title="Plant Disease Classifier")
st.title("🌿 Plant-Pathology (classical ML)")

file = st.file_uploader("Upload a leaf image …", type=["jpg", "jpeg", "png"])
if file:
    img = Image.open(file)
    st.image(img, caption="Input", use_column_width=True)
    tmp = Path("tmp.jpg")
    img.save(tmp)

    with st.spinner("Analysing …"):
        label, probs = predict(tmp)
    tmp.unlink(missing_ok=True)

    st.success(f"Prediction: **{label}**")
    st.json(probs)
    st.caption("Demo only – not medical advice.")
