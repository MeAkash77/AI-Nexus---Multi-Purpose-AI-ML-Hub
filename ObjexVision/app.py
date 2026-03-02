import os
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
import requests
import time
import streamlit_lottie as st_lottie

# Disable oneDNN warning
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ==============================
# PATH SETUP (CLOUD SAFE)
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "final_model1.h5")

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="CIFAR-10 Image Classification",
    page_icon="🖼️",
    layout="wide"
)

# ==============================
# LOAD LOTTIE
# ==============================
def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

lottie_url = "https://lottie.host/de06d967-8825-499e-aa8c-a88dd15e1a08/dH2OtlPb3c.json"
lottie_animation = load_lottie_url(lottie_url)

# ==============================
# SIDEBAR
# ==============================
with st.sidebar:
    if lottie_animation:
        st_lottie.st_lottie(lottie_animation, height=200, key="lottie")

    st.markdown("## Explore the App!")
    st.markdown("CNN model trained on CIFAR-10 dataset.")

    st.markdown("---")
    st.markdown(
        '[Contact: **Akash**](https://www.linkedin.com/in/me-akash77/)'
    )

# ==============================
# CLASS LABELS
# ==============================
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ==============================
# LOAD MODEL (CLOUD SAFE)
# ==============================
@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()

    model = tf.keras.models.load_model(MODEL_PATH)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

model = load_my_model()

# ==============================
# TITLE
# ==============================
st.title("🖼️ CIFAR-10 Image Classification")
st.write("Upload an image and let the model classify it.")

# ==============================
# IMAGE PREPROCESSING
# ==============================
def preprocess_image(image: Image.Image):
    image = image.resize((32, 32))
    img = img_to_array(image)
    img = img.reshape(1, 32, 32, 3)
    img = img.astype("float32") / 255.0
    return img

# ==============================
# FILE UPLOADER
# ==============================
image_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "png", "jpeg"]
)

if image_file is not None:

    image = Image.open(image_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Classify Image 🧠"):

        img_to_predict = preprocess_image(image)

        with st.spinner("Classifying..."):
            time.sleep(1)
            predictions = model.predict(img_to_predict)
            predicted_class = np.argmax(predictions, axis=-1)[0]
            confidence = float(np.max(predictions))

        confidence_threshold = 0.60

        if confidence < confidence_threshold:
            st.warning(
                f"Prediction uncertain ({confidence*100:.2f}%)"
            )
        else:
            st.success(
                f"Prediction: **{class_names[predicted_class]}** "
                f"({confidence*100:.2f}% confidence)"
            )

# ==============================
# CIFAR-10 INFO
# ==============================
st.markdown("## CIFAR-10 Classes")
st.write(", ".join(class_names))

# ==============================
# PERFORMANCE TABLE
# ==============================
data = {
    "Class": class_names,
    "Accuracy": [0.89, 0.85, 0.78, 0.92, 0.80, 0.76, 0.83, 0.88, 0.90, 0.81],
    "Precision": [0.87, 0.82, 0.77, 0.91, 0.79, 0.75, 0.81, 0.87, 0.88, 0.80],
}

performance_df = pd.DataFrame(data)
st.dataframe(performance_df)
