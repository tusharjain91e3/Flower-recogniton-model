# FLower-recogniton-model


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

bashgit clone https://github.com/yourusername/flower-recognition-ai.git
cd flower-recognition-ai

Create and activate a virtual environment (recommended):

bash# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python -m venv venv
source venv/bin/activate

Install required dependencies:

bashpip install tensorflow numpy streamlit
Usage
Step 1: Train the Model
Run the training script:
bashpython train_model.py
This will:

Load the flower dataset
Create and train a CNN model
Save the trained model as Flower_Recog_Model.h5

Note: Training may take several minutes depending on your hardware. The script will display progress and final accuracy.
Step 2: Launch the Web Application
After training is complete, start the Streamlit web application:
bashstreamlit run app.py
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

Acknowledgments

The Flowers Recognition dataset creators
TensorFlow and Keras documentation
Streamlit for the interactive web application framework
