import tensorflow as tf
import os
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import io
from tensorflow import keras
import pandas as pd
import pickle
import time
import numpy as np
from PIL import Image, ImageOps   
import requests
import streamlit_lottie as st_lottie
import base64

# Absolute path to this file's folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SEQ_MODEL_PATH = os.path.join(BASE_DIR, "Seq_model.h5")
CNN_MODEL_PATH = os.path.join(BASE_DIR, "cnn_model.h5")

@st.cache_resource
def load_models():

    if not os.path.exists(SEQ_MODEL_PATH):
        st.error(f"Sequential Model NOT found at: {SEQ_MODEL_PATH}")
        st.write("Files inside folder:", os.listdir(BASE_DIR))
        st.stop()

    if not os.path.exists(CNN_MODEL_PATH):
        st.error(f"CNN Model NOT found at: {CNN_MODEL_PATH}")
        st.write("Files inside folder:", os.listdir(BASE_DIR))
        st.stop()

    seq_model = tf.keras.models.load_model(SEQ_MODEL_PATH)
    cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)

    return seq_model, cnn_model

seq_model, cnn_model = load_models()

fas_data = keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels)=fas_data.load_data()

class_names = ['👕 Tshirt/Top', '👖 Trouser', '🧥 Pullover', '👗 Dress', '🧥 Coat',
               '👡 Sandal', '👔 Shirt', '👟 Sneaker', '👜 Bag', '👢 Ankle boot']

st.set_page_config(page_title="Fashion MNIST Classification", page_icon="👗", layout="wide")

st.markdown("""
    <h1 style="text-align:center; font-family: 'Courier New', Courier, monospace; animation: rainbow 3s ease-in-out infinite;">
    <span style="color: #ffcc00;">👗</span> Fashion MNIST Classification
    </h1>
""", unsafe_allow_html=True)

# ---------------- LOTTIE ----------------

def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url = "https://lottie.host/f96a3f24-0dac-4074-bdd2-df3c96371fa8/W1HZ5XC0hw.json"
lottie_animation = load_lottie_url(lottie_url)

with st.sidebar:
    if lottie_animation:
        st_lottie.st_lottie(lottie_animation, height=200, width=200)

    model_selection = st.selectbox(
        'Select the model for classification',
        ('🔢 Sequential', '🤖 CNN')
    )

    about_data_checked = st.checkbox('ℹ️ About Data')
    pretrained_network_checked = st.checkbox('🧠 Pretrained Neural Network')
    demo_images_checked = st.checkbox('👕 Demo Images')
    working_demo_checked = st.checkbox('🎥 Working Demo')
    contact_us_checked = st.checkbox('📞 Contact Us')

    st.markdown('Contact us at: [**Akash**](https://www.linkedin.com/in/me-akash77/)')

# ---------------- IMAGE CLASSIFICATION ----------------

file_uploader = st.file_uploader(
    "📂 Upload cloth image for classification",
    type=["jpg", "jpeg", "png"]
)

if file_uploader is not None:
    image = Image.open(file_uploader).resize((180, 180))
    st.image(image, caption='Uploaded image:')

    def classify_image(image, model):
        img = ImageOps.grayscale(image).resize((28, 28))
        img = np.array(img)

        if model_selection == '🤖 CNN':
            img = np.expand_dims(img, axis=(0, -1))
        else:
            img = np.expand_dims(img, 0)

        img = 1 - (img / 255.0)

        pred = model.predict(img)
        predicted_class = class_names[np.argmax(pred)]
        confidence = np.max(pred) * 100

        st.markdown(f"### 🎉 Predicted: {predicted_class}")
        st.markdown(f"### 🔮 Confidence: {confidence:.2f}%")

        chart_data = pd.DataFrame(pred.squeeze(), index=class_names, columns=['Confidence'])
        st.bar_chart(chart_data)

    if st.button('🧠 Classify Image'):
        model = cnn_model if model_selection == '🤖 CNN' else seq_model
        classify_image(image, model)
        st.success("✅ Image successfully classified!")
        st.balloons()

# ---------------- PERFORMANCE TABLES ----------------

data_cnn_updated = {
    "Class": class_names,
    "Accuracy": [0.72, 0.75, 0.72, 0.81, 0.70, 0.74, 0.70, 0.89, 0.88, 0.70],
    "Precision": [0.71, 0.78, 0.71, 0.80, 0.73, 0.83, 0.89, 0.87, 0.86, 0.69]
}

data_seq_updated = {
    "Class": class_names,
    "Accuracy": [0.89, 0.70, 0.63, 0.76, 0.72, 0.55, 0.68, 0.87, 0.90, 0.65],
    "Precision": [0.77, 0.62, 0.61, 0.75, 0.60, 0.54, 0.77, 0.85, 0.88, 0.63]
}

df_cnn_updated = pd.DataFrame(data_cnn_updated)
df_seq_updated = pd.DataFrame(data_seq_updated)

def create_styled_table(df, model_name):
    st.markdown(f"## {model_name} Performance")
    st.dataframe(df)

create_styled_table(df_cnn_updated, "CNN Model")
create_styled_table(df_seq_updated, "Sequential Model")
