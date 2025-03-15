import streamlit as st

# Set the page configuration at the very beginning
st.set_page_config(layout="wide", page_title="Maize Leaf Disease Classification (SCLB)")

import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# Google Drive file ID for the maize disease model
MODEL_FILE_ID = "1SKKWEE_UP8IBDnGGNDB5IdK3PjhEKKC1"
MODEL_PATH = "sclb_vgg_net16.h5"

# Function to download the model from Google Drive
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):  # Download only if the model is not already present
        url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    
    # Load the model
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

# Load the model
model = download_and_load_model()

# Define class labels
CLASS_NAMES = ["Unhealthy", "Healthy"]

st.title("Maize Leaf Disease Classification Dashboard (SCLB)")

# Creating two columns for a split-screen layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Preprocess the image
        img = image.resize((224, 224))  # Ensure this matches the model's expected input size
        img_array = np.array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make prediction
        predictions = model.predict(img_array)
        confidence = np.max(predictions) * 100
        predicted_class = CLASS_NAMES[np.argmax(predictions)]

        if confidence >= 50:
            final_prediction = predicted_class + ": Southern Corn Leaf Blight Disease"
            st.write(f"**Prediction:** {final_prediction}")
            st.write(f"**Confidence:** {confidence:.2f}%")
        else:
            final_prediction = "Healthy"
            st.write(f"**Prediction:** {final_prediction}")
            st.write(f"**Confidence:** {100 - confidence:.2f}%")

with col2:
    st.subheader("Maize Leaf Diseases")
    st.markdown(
        """
        **Common Maize Diseases:**
        - **Southern Corn Leaf Blight (SCLB):** Caused by *Bipolaris maydis*, leading to elongated lesions on leaves.
        - **Common Rust:** Caused by *Puccinia sorghi*, resulting in rust-colored pustules on leaves.
        - **Gray Leaf Spot:** Caused by *Cercospora zeae-maydis*, leading to grayish necrotic lesions.
        
        **Management Strategies:**
        - Use resistant maize varieties.
        - Apply fungicides such as *Azoxystrobin* or *Mancozeb* as per recommendations.
        - Maintain proper field sanitation and crop rotation.
        """
    )

st.markdown("---")
st.write("Developed by Anurag using Streamlit❤️")
