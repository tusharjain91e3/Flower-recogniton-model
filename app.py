import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st 
import numpy as np

# Set page title and header
st.markdown('<h1 style="text-align:center; font-size:2.5em;">Flower Classification CNN Model</h1>', unsafe_allow_html=True)

# Flower class names (same order as in training)
flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Define the path to your saved model file (not the flowers directory)
model_path = r'C:\Users\tusha\OneDrive\Desktop\Flower recognition model\Flower_Recog_Model.h5'

# Create a temporary directory for uploaded images
UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load the model
try:
    model = load_model(model_path)
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
        <div style="background-color:#f8f9ff; border-radius:16px; padding:24px; width:340px; box-shadow:0 2px 8px rgba(0,0,0,0.05);">
            <div style="font-size:2em; margin-bottom:8px; color:#6c63ff;">🌸</div>
            <div style="font-size:1.5em; font-weight:600; color:#222;">{flower.capitalize()}</div>
            <div style="color:#888; font-size:1.1em; margin-bottom:16px;">Confidence: {confidence:.2f}%</div>
            <div style="display:inline-block; background:#6c63ff; color:#fff; padding:8px 20px; border-radius:8px; text-decoration:none; font-weight:500;">Prediction Complete</div>
        </div>
        '''
        return card_html
    except Exception as e:
        return f"<div style='color:red'>Error classifying image: {e}</div>"

# File uploader widget
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
    if st.button("Classify"):
        with st.spinner("Classifying..."):
            # Display classification result
            st.markdown(classify_image(image_path), unsafe_allow_html=True)