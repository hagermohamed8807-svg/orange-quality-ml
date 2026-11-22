"""Orange quality predictor with a SHAP explanation for every prediction.

Run:  streamlit run app.py      (train first:  python -m src.train)
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

from src.data_utils import LABELS
from src.modeling import FULL, group_onehot, shap_explanation

st.set_page_config(page_title="Orange Quality Predictor", page_icon="🍊", layout="wide")
MODEL = Path(__file__).parent / "models" / "quality_model.joblib"

if not MODEL.exists():
    st.error("No trained model found. Run `python -m src.train` first.")
    st.stop()


@st.cache_resource
def get_model():
    return joblib.load(MODEL)


bundle = get_model()
pipe, meta, background = bundle["pipeline"], bundle["meta"], bundle["background"]
R, CATS = meta["num_ranges"], meta["categories"]

st.title("🍊 Orange Quality Predictor")
st.caption(f"Model: **{meta['model_name']}** · test RMSE {meta['test_metrics']['RMSE']:.2f} · "
           f"test R² {meta['test_metrics']['R2']:.2f}")

# ---------- Inputs ----------
st.sidebar.header("Describe the orange")


def slider(col, step):
    lo, hi, med = R[col]
    return st.sidebar.slider(LABELS[col], float(lo), float(hi), float(med), step)


size = slider("size_cm", 0.1)
weight = slider("weight_g", 1.0)
brix = slider("brix", 0.1)
ph = slider("ph", 0.1)
harvest = int(slider("harvest_days", 1.0))
softness = st.sidebar.select_slider(LABELS["softness"], options=[1, 2, 3, 4, 5], value=3)
ripeness = st.sidebar.select_slider(LABELS["ripeness"], options=[1, 2, 3, 4, 5], value=3)
variety = st.sidebar.selectbox("Variety", CATS["variety"])
color = st.sidebar.selectbox("Peel colour", CATS["color"])
has_blemish = st.sidebar.radio("Visible blemish?", ["No", "Yes"], horizontal=True) == "Yes"
blemish_type = st.sidebar.selectbox("Blemish type", CATS["blemish_type"])

row = pd.DataFrame([{
    "size_cm": size, "weight_g": weight, "density_proxy": weight / size ** 3,
    "has_blemish": int(has_blemish),
    "softness": softness, "ripeness": ripeness, "harvest_days": harvest,
    "brix": brix, "ph": ph, "sugar_acid_ratio": brix / ph,
    "color": color, "variety": variety, "blemish_type": blemish_type,
}])

# ---------- Prediction ----------
score = float(min(max(pipe.predict(row)[0], 1.0), 5.0))
grade = "High" if score > 4 else "Medium" if score > 3 else "Low"
emoji = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}[grade]

c1, c2 = st.columns([1, 2])
with c1:
    st.metric("Predicted quality (1–5)", f"{score:.2f}")
    st.markdown(f"### {emoji} {grade} quality")
    st.caption(f"Typical error ±{meta['test_metrics']['MAE']:.2f} points (test MAE).")

with c2:
    st.subheader("Why this score?")
    expl = group_onehot(shap_explanation(pipe, row, background=background), row)
    expl.feature_names = [LABELS.get(f, f.replace("_", " ").title()) for f in expl.feature_names]
    plt.figure()
    shap.plots.waterfall(expl[0], max_display=10, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)
    st.caption("Red bars push the score up, blue bars push it down, starting from the "
               "average prediction E[f(x)].")
