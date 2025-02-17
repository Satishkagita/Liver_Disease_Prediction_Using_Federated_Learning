import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

MODEL_PATH = "global_model.keras" 
CLASS_NAMES = ["ballooning", "fibrosis", "inflammation", "steatosis"]  

TEST_DATASET_PATH = "server/test_data/"

@st.cache_data
def load_model():
    test_datagen = ImageDataGenerator(rescale=1/255.)
    test_data = test_datagen.flow_from_directory(
        TEST_DATASET_PATH, target_size=(224, 224), batch_size=32, class_mode='categorical'
        )
    model = tf.keras.models.load_model(MODEL_PATH)
    loss, acc = model.evaluate(test_data)
    print(f"Server Model Evaluation - Loss: {loss:.4f}, Accuracy: {acc:.4f}")
    return model

model = load_model()

def preprocess_image(image):
    image = image.resize((224, 224)) 
    image = np.array(image) / 255.0 
    image = np.expand_dims(image, axis=0)  #
    return image


st.title("Liver Disease Classification")
st.write("Upload an image to predict its class.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    input_tensor = preprocess_image(image)

    # Perform prediction
    predictions = model.predict(input_tensor)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    st.write(f"### 🏥 Predicted Class: **{predicted_class}**")
    st.write(f"### 🔍 Confidence: **{confidence:.2f}%**")
