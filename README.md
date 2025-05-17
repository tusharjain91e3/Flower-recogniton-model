Flower Recognition AI Project
This repository contains a machine learning project that classifies images of flowers into 5 categories: daisy, dandelion, rose, sunflower, and tulip. It includes both the model training script and a Streamlit web interface for easy interaction with the trained model.

Project Overview
The project uses TensorFlow/Keras to train a Convolutional Neural Network (CNN) on a flower dataset. Once trained, the model can identify flower types from uploaded images with a prediction confidence score.

Features
Image classification of 5 different flower types
Detailed model training with data augmentation
Interactive web interface using Streamlit
Real-time predictions with confidence scores
Dataset
This project uses the Flowers Recognition dataset from Kaggle, which contains images of 5 types of flowers:

Daisy
Dandelion
Rose
Sunflower
Tulip
Dataset Download Instructions
Sign up/login to Kaggle
Go to the Flowers Recognition dataset
Click "Download" to get the dataset ZIP file
Extract the ZIP file to create a flowers directory with subdirectories for each flower type
Place the flowers directory in your project folder
Directory Structure
After setup, your project directory should look like this:

Flower_Recognition_Model/
├── flowers/               # Dataset directory
│   ├── daisy/             # Flower type subdirectories
│   ├── dandelion/
│   ├── rose/
│   ├── sunflower/
│   └── tulip/
├── train_model.py         # Script to train the model
├── app.py                 # Streamlit web application
├── Flower_Recog_Model.h5  # Saved model file (created after training)
└── uploaded_images/       # Directory for temporary image storage
Installation
Clone this repository:
bash
git clone https://github.com/tusharjain91e3/Flower-recogniton-model.git
cd Flower-recogniton-model
Create and activate a virtual environment (recommended):
bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python -m venv venv
source venv/bin/activate
Install required dependencies:
bash
pip install tensorflow numpy streamlit
Usage
Step 1: Train the Model
Run the training script:

bash
python train_model.py
This will:

Load the flower dataset
Create and train a CNN model
Save the trained model as Flower_Recog_Model.h5
Note: Training may take several minutes depending on your hardware. The script will display progress and final accuracy.

Step 2: Launch the Web Application
After training is complete, start the Streamlit web application:

bash
streamlit run app.py
This will:

Open a browser window with the application interface
Allow you to upload and classify flower images
Important Notes
Path Configuration: Make sure to update the paths in both scripts if your directory structure differs:
In train_model.py: Update base_path variable
In app.py: Update model_path variable
Model File: Ensure the model file (Flower_Recog_Model.h5) is in the correct location as specified in app.py
System Requirements:
Python 3.7+
Sufficient RAM for training (4GB minimum recommended)
GPU support is helpful but not required
Troubleshooting
Missing dataset: Ensure the flowers directory is correctly placed and contains subdirectories for each flower type
Model loading error: Check if the model file exists at the specified path in app.py
Memory errors: Reduce batch size in train_model.py if you encounter memory issues
Import errors: Make sure all required libraries are installed
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
The Flowers Recognition dataset creators
TensorFlow and Keras documentation
Streamlit for the interactive web application framework
