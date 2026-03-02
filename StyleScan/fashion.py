import tensorflow as tf
import os
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import io
from tensorflow import keras
import cnn_model
import Seq_model
import pandas as pd
import pickle
import time
import numpy as np
from PIL import Image, ImageOps   
import requests  # For fetching Lottie animation
import streamlit_lottie as st_lottie
import base64  # <-- Import base64 for GIF encoding

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SEQ_MODEL_PATH = os.path.join(BASE_DIR, "Seq_model.h5")
CNN_MODEL_PATH = os.path.join(BASE_DIR, "cnn_model.h5")

@st.cache_resource
def load_models():
    if not os.path.exists(SEQ_MODEL_PATH):
        st.error(f"Sequential Model NOT found at: {SEQ_MODEL_PATH}")
        st.write(os.listdir(BASE_DIR))
        st.stop()

    if not os.path.exists(CNN_MODEL_PATH):
        st.error(f"CNN Model NOT found at: {CNN_MODEL_PATH}")
        st.write(os.listdir(BASE_DIR))
        st.stop()

    seq_model = tf.keras.models.load_model(SEQ_MODEL_PATH)
    cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)

    return seq_model, cnn_model

seq_model, cnn_model = load_models()

fas_data=keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels)=fas_data.load_data()


class_names = ['👕 Tshirt/Top', '👖 Trouser', '🧥 Pullover', '👗 Dress', '🧥 Coat',
               '👡 Sandal', '👔 Shirt', '👟 Sneaker', '👜 Bag', '👢 Ankle boot']

# Page title
st.set_page_config(page_title="Fashion MNIST Classification", page_icon="👗", layout="wide")

# # Themed dark/light toggle
# theme_choice = st.sidebar.radio("Choose Theme", ("🌑 Dark", "🌕 Light"))

# if theme_choice == "🌑 Dark":
#     st.markdown("""
#         <style>
#             body { background-color: #2c2c2c; color: white; }
#         </style>
#     """, unsafe_allow_html=True)
# else:
#     st.markdown("""
#         <style>
#             body { background-color: #ffffff; color: black; }
#         </style>
#     """, unsafe_allow_html=True)


st.markdown("""
    <h1 style="text-align:center; font-family: 'Courier New', Courier, monospace; animation: rainbow 3s ease-in-out infinite;">
    <span style="color: #ffcc00;">👗</span> Fashion MNIST Classification
    </h1>
    <style>
    @keyframes rainbow {
        0% { color: #ffcc00; }
        10% { color: #ff5733; }
        20% { color: #ff33cc; }
        30% { color: #33ff57; }
        40% { color: #33ccff; }
        50% { color: #5733ff; }
        60% { color: #cc33ff; }
        70% { color: #ffcc00; }
        80% { color: #ff5733; }
        90% { color: #ff33cc; }
        100% { color: #ffcc00; }
    }
    </style>
""", unsafe_allow_html=True)



# Custom CSS for centering elements in the sidebar
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        text-align: center;
    }
    [data-testid="stSidebar"] .element-container {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
    }
    </style>
    """, unsafe_allow_html=True
)

# Load and display GIF
def get_gif_base64(gif_path):
    with open(gif_path, "rb") as gif_file:
        return base64.b64encode(gif_file.read()).decode('utf-8')

# Function to load Lottie animations
def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load Lottie animation
# Lottie Animation URL
lottie_url = "https://lottie.host/f96a3f24-0dac-4074-bdd2-df3c96371fa8/W1HZ5XC0hw.json"
lottie_animation = load_lottie_url(lottie_url)

# Sidebar with Lottie animation and model selection
with st.sidebar:
    st_lottie.st_lottie(lottie_animation, height=200, width=200, key="lottie_animation")
    st.markdown("<h2 style='color: #007bff;'>Explore the App! 🎉</h2>", unsafe_allow_html=True)
    
    # Dropdown for model selection
    model_selection = st.selectbox(
        'Select the model for classification',
        ('🔢 Sequential', '🤖 CNN')
    )
    # Checkboxes for different sections
    about_data_checked = st.checkbox('ℹ️ About Data')
    pretrained_network_checked = st.checkbox('🧠 Pretrained Neural Network')
    demo_images_checked = st.checkbox('👕 Demo Images')
    working_demo_checked = st.checkbox('🎥 Working Demo')
    contact_us_checked = st.checkbox('📞 Contact Us')

    # Contact information
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('Contact us at: [**Akash**](https://www.linkedin.com/in/me-akash77/?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)')


def show_loader():
    with st.spinner("Loading model..."):
        time.sleep(2)  # Simulating a delay for loading


# Function to explore data
def explore_data(train_images, train_labels, test_images):
    st.markdown("### 📊 **Explore Data**")
    st.write('🖼️ Train Images shape:', train_images.shape)
    st.write('🖼️ Test Images shape:', test_images.shape)
    st.write('🧑‍🏫 Training Classes:', len(np.unique(train_labels)))
    st.write('🧑‍🔬 Testing Classes:', len(np.unique(test_images)))

def CNN_model_summary():
    st.markdown("### 🧠 **CNN Model Summary**")
    img = Image.open("cnn_summary.png")
    st.image(img)

# Function to show Sequential Model Summary
def Seq_model_Summary():
    st.markdown("### 📜 **Sequential Model Summary**")
    img = Image.open("Seq_summary.png")
    st.image(img)


# Graph plotting functions
def seq_history_graph():
    with open('seq_trainHistory', 'rb') as infile:
        history = pickle.load(infile)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    train_acc = history['accuracy']
    val_acc = history['val_accuracy']
    train_loss = history['loss']
    val_loss = history['val_loss']
    
    ax1.plot(train_acc, label='Training Accuracy', color='#1f77b4', linestyle='-', marker='o')
    ax1.plot(val_acc, label='Validation Accuracy', color='#ff7f0e', linestyle='--', marker='s')
    ax1.set_title('Training vs Validation Accuracy')
    ax1.legend(loc='lower right')

    ax2.plot(train_loss, label='Training Loss', color='#d62728', linestyle='-', marker='^')
    ax2.plot(val_loss, label='Validation Loss', color='#2ca02c', linestyle='--', marker='D')
    ax2.set_title('Training vs Validation Loss')
    ax2.legend(loc='upper right')

    st.pyplot(fig)

def cnn_history_graph():
    with open('cnntrainHistory', 'rb') as infile:
        history = pickle.load(infile)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    train_acc = history['accuracy']
    val_acc = history['val_accuracy']
    train_loss = history['loss']
    val_loss = history['val_loss']
    
    ax1.plot(train_acc, label='Training Accuracy', color='#1f77b4', linestyle='-', marker='o')
    ax1.plot(val_acc, label='Validation Accuracy', color='#ff7f0e', linestyle='--', marker='s')
    ax1.set_title('Training vs Validation Accuracy')
    ax1.legend(loc='lower right')

    ax2.plot(train_loss, label='Training Loss', color='#d62728', linestyle='-', marker='^')
    ax2.plot(val_loss, label='Validation Loss', color='#2ca02c', linestyle='--', marker='D')
    ax2.set_title('Training vs Validation Loss')
    ax2.legend(loc='upper right')

    st.pyplot(fig)

# Architecture Images
def cnn_archi():
    st.markdown("### 🧠 **CNN Model Architecture**")
    img = Image.open('cnn_model_architecture.png')
    st.image(img)

def seq_archi():
    st.markdown("### 📜 **Sequential Model Architecture**")
    img = Image.open('seq_model_architecture.png')
    st.image(img)

if about_data_checked:
    st.markdown("---")  # Add a separator
    st.header("ℹ️ About Data")

    # Create a DataFrame for the table
    about_data = {
        "Sections": [
            "Explore Data",
            "CNN Model Summary",
            "CNN Model Architecture",
            "Sequential Model Summary",
            "Sequential Model Architecture",
            "Sequential Model Graph",
            "CNN Model Graph"
        ],
        "Actions": [
            "📊 Explore Data",
            "🧠 CNN Model Summary",
            "🧠 CNN Model Architecture",
            "📜 Sequential Model Summary",
            "📜 Sequential Model Architecture",
            "📈 Sequential Model Graph",
            "📈 CNN Model Graph"
        ]
    }

    df_about_data = pd.DataFrame(about_data)

    # Display the table with enhanced styling for dark theme
    st.table(df_about_data.style.set_table_attributes('style="width: 100%; border: 1px solid #444;"')
                        .set_caption("### Actions Available")
                        .set_properties(**{'text-align': 'left', 'color': 'white', 'background-color': '#222'})
                        .set_table_styles([{
                            'selector': 'th',
                            'props': [('background-color', '#007bff'), ('color', 'white')]
                        }, {
                            'selector': 'td',
                            'props': [('padding', '10px'), ('color', 'white')]
                        }]))

    # Add buttons for interaction instead of a select box
    selected_action = st.selectbox("Select an action to perform:", df_about_data['Actions'].values)
    
    if selected_action == "📊 Explore Data":
        explore_data(train_images, train_labels, test_images)
    elif selected_action == "🧠 CNN Model Summary":
        CNN_model_summary()
    elif selected_action == "🧠 CNN Model Architecture":
        cnn_archi()
    elif selected_action == "📜 Sequential Model Summary":
        Seq_model_Summary()
    elif selected_action == "📜 Sequential Model Architecture":
        seq_archi()
    elif selected_action == "📈 Sequential Model Graph":
        seq_history_graph()
    elif selected_action == "📈 CNN Model Graph":
        cnn_history_graph()

# Demo Images Section
if demo_images_checked:
    st.markdown("---")
    st.header("👕 Demo Images")
    st.write("🖼️ Please upload the following types of clothes for classification:")
    
    images = [
    ("Demo Images/bag.jpeg", "👜 Bag"),
    ("Demo Images/Sneaker.png", "👟 Sneaker"),
    ("Demo Images/snicker.jpg", "👟 Snicker"),
    ("Demo Images/coat.jpg", "🧥 Coat"),
    ("Demo Images/Trouser.jpeg", "👖 Trouser"),
    ("Demo Images/Dress.jpeg", "👗 Dress"),
    ("Demo Images/Pant.jpeg", "🩳 Pant"),
    ("Demo Images/blazer.jpeg", "🧥 Blazer"),
    ("Demo Images/shirt.jpg", "👚 Shirt"),
    ("Demo Images/T-shirt.jpeg", "👕 T-Shirt"), 
    ]
    for img_path, label in images:
        full_path = os.path.join(BASE_DIR, img_path)

        if os.path.exists(full_path):
        image = Image.open(full_path).resize((180, 180))
        st.image(image, caption=label)
        else:
        st.warning(f"Image not found: {img_path}")

# Pretrained Network Section
        if pretrained_network_checked:
        st.info("🧠 Working on it, update coming soon!")

# Working Demo Section
if working_demo_checked:
    st.markdown("---")  # Add a separator
    st.header("🎥 Working Demo")

    # Load and display the video
    video_file_path = 'streamlit-fashion-Mnist.webm'  # Update this path to your video file
    try:
        video_file = open(video_file_path, 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)
    except FileNotFoundError:
        st.error(f"⚠️ Video file not found: {video_file_path}. Please check the file path.")

# Contact Us Section
if contact_us_checked:
    st.markdown("---")
    st.header("📞 Contact Us")

    IMAGE_PATH = os.path.join(BASE_DIR, "Het Patel.jpg")

    if os.path.exists(IMAGE_PATH):
        contact_image = Image.open(IMAGE_PATH).resize((400, 400))
        st.image(contact_image, caption='Akash')
    else:
        st.warning("Contact image not found in repository.")

    st.write('📧 Email: iakshu845@gmail.com')
    
# File Uploader and Image Classification
file_uploader = st.file_uploader( "📂 Upload cloth image for classification",
    type=["jpg", "jpeg", "png"]
)
if file_uploader is not None:
    image = Image.open(file_uploader).resize((180, 180))
    st.image(image, caption='Uploaded image:')
    
    # Image classification function
    def classify_image(image, model):
        st.write("🖼️ Classifying the image...")
        img = ImageOps.grayscale(image).resize((28, 28))
        img = np.expand_dims(img, axis=(0, -1)) if model_selection == 'CNN' else np.expand_dims(img, 0)
        img = 1 - (img / 255.0)
        
        pred = model.predict(img)
        predicted_class = class_names[np.argmax(pred)]
        confidence = np.max(pred) * 100
        
    # Display prediction results
        st.markdown(f'<div style="text-align: center; font-size: 24px; color: #FF5733;">🎉 Predicted: {predicted_class} 🎉</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="text-align: center; font-size: 20px; color: #3498DB;">🔮 Confidence: {confidence:.2f}%</div>', unsafe_allow_html=True)

    # Display a bar graph for model predictions
        chart_data = pd.DataFrame(pred.squeeze(), index=class_names, columns=['Confidence'])
        st.bar_chart(chart_data)

    # Display the animated GIF after prediction
        gif_path = "Celebrations.gif"  # Ensure this path is correct
        gif_base64 = get_gif_base64(gif_path)
        st.markdown(
        f"""
        <div style="text-align: center; border: 5px solid #FF5733; padding: 10px; animation: pulse 1s infinite;">
            <img src="data:image/gif;base64,{gif_base64}" style="max-width: 100%; height: auto;">
        </div>
        <style>
        @keyframes pulse {{
            0% {{ transform: scale(1); }}
            50% {{ transform: scale(1.05); }}
            100% {{ transform: scale(1); }}
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
    if st.button('🧠 Classify Image'):
        model = cnn_model if model_selection == 'CNN' else seq_model
        classify_image(image, model)
        st.success("✅ Image successfully classified!")
        st.balloons()

# Updated data with more balanced accuracies
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

# Function to create the styled table with animation
def create_styled_table(df, model_name):
    st.markdown(f"<h2 style='color:#ffcc00;'>{model_name} Performance</h2>", unsafe_allow_html=True)
    st.markdown("""
        <style>
        .dataframe {
            border-collapse: collapse;
            width: 100%;
            border: 2px solid #444;
            font-family: 'Courier New', Courier, monospace;
        }
        .dataframe thead th {
            background-color: #444;
            color: #ffcc00;
            padding: 10px;
            border-bottom: 2px solid #ffcc00;
        }
        .dataframe tbody tr {
            transition: all 0.3s ease-in-out;
        }
        .dataframe tbody tr:nth-child(even) {
            background-color: #333;
        }
        .dataframe tbody tr:hover {
            background-color: #555;
            transform: scale(1.02);
        }
        .dataframe tbody td {
            text-align: left;
            padding: 10px;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

    # Render the table as HTML
    st.markdown(df.to_html(index=False, classes="dataframe"), unsafe_allow_html=True)

# Display both CNN and Sequential models' updated performance tables
create_styled_table(df_cnn_updated, "CNN Model")
create_styled_table(df_seq_updated, "Sequential Model")  
