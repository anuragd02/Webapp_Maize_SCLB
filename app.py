import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# -------------------- Page Setup --------------------
st.set_page_config(
    page_title="Maize Leaf Disease Classification (SCLB)",
    layout="wide",
    page_icon="🌽",
)

st.markdown(
    """
    <style>
        .main-title {font-size: 40px; font-weight: 700; color: #228B22; text-align:center;}
        .sub-title {font-size: 22px; font-weight: 600; color: #444;}
        .stTable td {font-size:16px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🌽 Maize Leaf Disease Dashboard</div>', unsafe_allow_html=True)
st.write("---")

# -------------------- Model Loading --------------------
MODEL_FILE_ID = "1SKKWEE_UP8IBDnGGNDB5IdK3PjhEKKC1"
MODEL_PATH = "sclb_vgg_net16.h5"

@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH)

model = download_and_load_model()
CLASS_NAMES = ["Unhealthy", "Healthy"]

# -------------------- Layout --------------------
col1, col2 = st.columns([1.2, 1.4])

# ========== Column 1 : Upload & Prediction ==========
with col1:
    st.markdown('<div class="sub-title">Upload an Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a maize leaf image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocess
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)
        confidence = np.max(predictions) * 100
        predicted_class = CLASS_NAMES[np.argmax(predictions)]

        if confidence >= 50 and predicted_class == "Unhealthy":
            final_prediction = "Southern Corn Leaf Blight Detected"
            conf_display = f"{confidence:.2f}%"
        else:
            final_prediction = "Healthy"
            conf_display = f"{(100 - confidence):.2f}%" if predicted_class == "Unhealthy" else f"{confidence:.2f}%"

        st.success(f"**Prediction:** {final_prediction}")
        st.info(f"**Confidence:** {conf_display}")

# ========== Column 2 : Fungicide Advisory ==========
with col2:
    st.markdown('<div class="sub-title">🌱 Southern Corn Leaf Blight – Fungicide Advisory</div>', unsafe_allow_html=True)

    with st.expander("General Guidelines", expanded=True):
        st.markdown(
            """
            • Monitor crop regularly from **knee-high** to **tasseling** stage (most vulnerable).  
            • Spray based on **disease severity** and **crop stage** for maximum effect.  
            • Ensure good spray coverage: **500 L water/ha** with knapsack or motorized sprayer.
            """
        )

    with st.expander("Severity-Based Spraying", expanded=True):
        st.markdown(
            """
            | Disease Severity (PDI) | Crop Stage | Recommended Spray |
            |------------------------|-----------|-------------------|
            | **Low (≤10%)** | Knee-high (30–35 DAS) | Preventive **Carbendazim + Mancozeb** or **Mancozeb** |
            | **Moderate (10–20%)** | Pre-tasseling (45–55 DAS) | **Azoxystrobin + Cyproconazole** or **Pyraclostrobin + Epoxiconazole** |
            | **High (>20%)** | Tasseling–grain filling (60–80 DAS) | **Azoxystrobin + Difenoconazole** (best) – repeat after 15–20 days if humid/warm |
            """
        )

    with st.expander("Advisory Highlights", expanded=True):
        st.markdown(
            """
            • **First Spray:** At first disease appearance or knee-high stage (whichever is earlier).  
            • **Second Spray:** Pre-tasseling (45–55 DAS) depending on severity.  
            • **Third Spray:** Tasseling to grain-filling (60–80 DAS) under high disease pressure.  
            • **Spray Volume:** 500 L water/ha for good coverage.  
            • **Rotation:** Rotate fungicides with different **FRAC codes** to prevent resistance.  
            • **Most Effective:** **Azoxystrobin + Difenoconazole**.  
              Alternatives: **Azoxystrobin + Cyproconazole**, **Pyraclostrobin + Epoxiconazole**.
            """
        )

st.write("---")
st.caption("Developed by Anurag • Powered by Streamlit ❤️")

