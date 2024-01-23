import streamlit as st
from PIL import Image
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import joblib
import efficientnet.tfkeras as efn
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import joblib
import efficientnet.tfkeras as efn
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input
import cv2
import os

loaded_svm_model = joblib.load('svm_model_updated.joblib')
list_of_classes  = ['Rice___Brown_Spot', 'Corn___Common_Rust', 'Rice___Healthy', 'Corn___Healthy', 'Potato___Late_Blight', 'Potato___Healthy']

def load_model():
	with open('enet_updated.json', 'r') as json_file:
		loaded_model_json = json_file.read()

	loaded_model = tf.keras.models.model_from_json(loaded_model_json)
	return loaded_model

model = load_model()
model.load_weights('efficientnet_b0_model_updated.h5')

def predict(image_path):
	img = cv2.imread(image_path)
	img_resized = cv2.resize(img, (128, 128))
	img_resized = img_resized.astype('float32') / 255.0
	img_resized = np.expand_dims(img_resized, axis=0)

	# model = loaded_model()
	pred = model.predict(img_resized)
	new_data_flat = pred.reshape(1, -1) 
	prediction = loaded_svm_model.predict(new_data_flat)[0]
	return list_of_classes[prediction]

# Streamlit app
st.title("Crop Disease Prediction App")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    image.save("image.jpg")
    st.image(image, use_column_width=True)
    prediction = predict("image.jpg")
    if prediction in ['Rice___Brown_Spot', 'Corn___Common_Rust', 'Potato___Late_Blight']:
        st.markdown(f"<h1 style='text-align: center; color: red;'>Prediction: {prediction}</h1>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h1 style='text-align: center; color: green;'>Prediction: {prediction}</h1>", unsafe_allow_html=True)
