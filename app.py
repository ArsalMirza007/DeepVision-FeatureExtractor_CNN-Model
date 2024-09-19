import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import os

# Load the trained model (adjust the path if necessary)
model_path = 'saved_model/my_cnn_model.h5'  # Use forward slashes or double backslashes
cnn = tf.keras.models.load_model(model_path)

# Streamlit UI
st.title("Image Recognition with Deep Learning")
st.header("Upload an Image to Predict")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = image.load_img(uploaded_file, target_size=(64, 64))
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Prepare the image for prediction
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale the image as the model was trained on rescaled images
    
    # Prediction
    result = cnn.predict(img_array)
    class_indices = {'cat': 0, 'dog': 1}  # Adjust based on your training set
    class_labels = {v: k for k, v in class_indices.items()}
    
    # Get prediction label
    predicted_class = 'Dog' if result[0][0] >= 0.5 else 'Cat'
    confidence = result[0][0] if predicted_class == 'Dog' else 1 - result[0][0]

    # Display prediction
    st.write(f"Prediction: **{predicted_class}**")
    st.write(f"Confidence: **{confidence:.2f}**")

    # Add a probability bar
    st.progress(float(confidence))

# Additional Information
st.write("This model predicts whether an uploaded image is of a cat or a dog.")
st.write("Ensure that the image is properly centered and of good quality for better results.")
