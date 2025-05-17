import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st 
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Flower Recognition",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply dark theme styling
st.markdown("""
<style>
    /* Title styling */
    h1 {
        font-size: 2.5em !important;
        font-weight: bold !important;
        text-align: center !important;
        margin-bottom: 30px !important;
        color: white !important;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #6c63ff !important;
        color: white !important;
        font-size: 1.2em !important;
        padding: 8px 16px !important;
        border-radius: 8px !important;
        border: none !important;
        width: 100% !important;
    }
    
    /* Hide success message */
    .st-emotion-cache-16txtl3 {
        display: none;
    }
    
    /* Upload widget label */
    .uploadedFileData {
        font-size: 1.2em !important;
    }
    
    /* Columns padding */
    .block-container {
        padding: 2rem 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Main app title
st.markdown('<h1>AI PROJECT FLOWER RECOGNITION MODEL</h1>', unsafe_allow_html=True)

# Flower class names (same order as in training)
flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Define the path to your saved model file
model_path = r'C:\Users\tusha\OneDrive\Desktop\Flower recognition model\Flower_Recog_Model.h5'

# Create a temporary directory for uploaded images
UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load the model
try:
    model = load_model(model_path)
    # Success message hidden by CSS
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Please make sure the model file 'Flower_Recog_Model.h5' exists in the specified location.")

def classify_image(image_path):
    """Function to classify an uploaded flower image"""
    try:
        # Load and preprocess the image
        input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
        input_image_array = tf.keras.utils.img_to_array(input_image)
        input_image_exp_dim = tf.expand_dims(input_image_array, 0)
        
        # Make prediction
        predictions = model.predict(input_image_exp_dim)
        result = tf.nn.softmax(predictions[0])
        
        # Get the predicted class and confidence
        flower = flower_names[np.argmax(result)]
        confidence = np.max(result) * 100
        
        # Styled card output
        card_html = f'''
        <div style="background-color:#1e1e2e; border-radius:16px; padding:24px; width:340px; box-shadow:0 4px 12px rgba(0,0,0,0.15); margin-left:auto; margin-right:auto;">
            <div style="font-size:2em; margin-bottom:8px; color:#6c63ff; text-align:center;">🌸</div>
            <div style="font-size:1.8em; font-weight:600; color:white; text-align:center;">{flower.capitalize()}</div>
            <div style="color:#a0a0a0; font-size:1.2em; margin-bottom:16px; text-align:center;">Confidence: {confidence:.2f}%</div>
            <div style="display:block; background:#6c63ff; color:#fff; padding:8px 20px; border-radius:8px; text-decoration:none; font-weight:500; text-align:center;">Prediction Complete</div>
        </div>
        '''
        return card_html
    except Exception as e:
        return f"<div style='color:red; text-align:center;'>Error classifying image: {e}</div>"

# Create two columns for side-by-side layout
left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Upload a flower image for classification")
    uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Save the uploaded file
        image_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(image_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        # Display the uploaded image
        st.image(uploaded_file, width=300, caption="Uploaded Image")
        
        # Show a "Classify" button
        classify_button = st.button("Classify")

# Right column for results
with right_col:
    st.subheader("Classification Result")
    
    # Display the result if classify button was clicked
    if 'uploaded_file' in locals() and uploaded_file is not None and 'classify_button' in locals() and classify_button:
        with st.spinner("Classifying..."):
            st.markdown(classify_image(image_path), unsafe_allow_html=True)
