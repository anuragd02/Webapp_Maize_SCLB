import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Maize Leaf Disease Classification Dashboard (SCLB)",
    layout="wide",
)

st.title("Maize Leaf Disease Classification Dashboard (SCLB)")

# ---------------- Load Model -----------------
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

# ---------------- Layout ---------------------
col1, col2 = st.columns([1, 1])

# Left column: Upload & Prediction
with col1:
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocess image
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

        st.write(f"**Prediction:** {final_prediction}")
        st.write(f"**Confidence:** {conf_display}")

# Right column: Fungicide Advisory
with col2:
    st.subheader("Southern Corn Leaf Blight (SCLB) – Fungicide Advisory")

    st.markdown(
        """
        **General Guidelines**
        * Monitor crop regularly for early symptoms, especially from knee-high stage to tasseling (most vulnerable).
        * Spray based on **disease severity** and **crop stage** for maximum benefit.
        * Ensure good spray coverage using **500 L water/ha** with a knapsack or motorized sprayer.

        **Severity-Based Spraying**

        | Disease Severity (PDI) | Crop Stage | Spray Recommendation |
        |------------------------|-----------|----------------------|
        | **Low (≤10%)**         | Knee-high (30–35 DAS) | Preventive spray with **Carbendazim + Mancozeb** or **Mancozeb** (if no prior infection). |
        | **Moderate (10–20%)**  | Pre-tasseling (45–55 DAS) | **Azoxystrobin + Cyproconazole** or **Pyraclostrobin + Epoxiconazole** |
        | **High (>20%)**        | Tasseling–grain filling (60–80 DAS) | **Azoxystrobin + Difenoconazole** (best). Repeat after 15–20 days if humid/warm. |

        **Advisory Highlights**
        * **First Spray:** At disease appearance or knee-high stage (whichever is earlier).
        * **Second Spray:** Pre-tasseling stage (45–55 DAS) depending on severity.
        * **Third Spray:** Tasseling to grain-filling stage (60–80 DAS) under high disease pressure.
        * **Spray Volume:** 500 L water/ha for good coverage.
        * **Rotation:** Rotate fungicides with different **FRAC codes** to avoid resistance.
        * **Most Effective:** **Azoxystrobin + Difenoconazole**; good alternatives include **Azoxystrobin + Cyproconazole** and **Pyraclostrobin + Epoxiconazole**.
        """
    )

st.markdown("---")
st.write("Developed by Anurag using Streamlit ❤️")

