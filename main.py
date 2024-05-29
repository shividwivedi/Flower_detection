import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import json

class_names = ['Azalea', 'Birds of Paradise', 'California Poppy', 'China Pink', 'Coltsfoot', 'Columbine', 'Garden Nasturtium', 'Gladiolus', 'Iris', 'Japanese Camellia', 'Jimsonweed', 'Nelumbo Nucifera', "Painter's Palette", 'Poinsettia', 'Purple Coneflower', 'Pygmy Water Lily', 'Red Frangipani', 'Daisy', 'Dandelion', 'Roses', 'Sunflowers', 'Tulips']

# Save class_names to a JSON file for later use in applications
with open('class_names.json', 'w') as f:
    json.dump(class_names, f)

# Load the saved model
model = tf.keras.models.load_model('Model.h5')

# Load class names from JSON file
with open('class_names.json') as f:
    class_names = json.load(f)

st.title('Flower Classification App')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    image = load_img(uploaded_file, target_size=(180, 180))
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)
    predicted_class = class_names[np.argmax(predictions)]
    st.write(f"Prediction: {predicted_class}")
