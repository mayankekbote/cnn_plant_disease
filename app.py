import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# ------------------------------
# CONFIG
# ------------------------------
st.set_page_config(page_title="Plant Disease Detector 🌿", page_icon="🌱", layout="centered")

# Load trained model
MODEL_PATH = "best_model.h5"
model = load_model(MODEL_PATH)

# Class names (23 total)
class_names = [
    'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
    'Corn_(maize)___Cercospora_leaf_spot','Corn_(maize)___Common_rust_','Corn_(maize)___healthy',
    'Corn_(maize)___Northern_Leaf_Blight','Pepper__bell___Bacterial_spot','Pepper__bell___healthy',
    'Potato___Early_blight','Potato___healthy','Potato___Late_blight',
    'Tomato__Target_Spot','Tomato__Tomato_mosaic_virus','Tomato__Tomato_YellowLeaf_Curl_Virus',
    'Tomato__Bacterial_spot','Tomato__Early_blight','Tomato__healthy','Tomato__Late_blight',
    'Tomato__Leaf_Mold','Tomato__Septoria_leaf_spot','Tomato__Spider_mites_Two_spotted_spider_mite'
]

# ------------------------------
# UI
# ------------------------------
st.title("🌿 Plant Disease Detection")
st.markdown("Upload a leaf image and the model will predict the disease type.")

uploaded_file = st.file_uploader("📤 Upload a plant leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image_data = Image.open(uploaded_file)
    st.image(image_data, caption="Uploaded Image", use_column_width=True)


    # Preprocess image
    img = image_data.resize((128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    with st.spinner("🔍 Analyzing..."):
        pred = model.predict(img_array)
        predicted_class = np.argmax(pred, axis=1)[0]
        confidence = float(np.max(pred))

    # Display result
    result=class_names[predicted_class]
    if "healthy" in result:
        st.success(f"✅ **Prediction:** {result}")
    else:
        st.error(f"⚠️ **Prediction:** {result}")
    st.info(f"💡 **Confidence:** {confidence*100:.2f}%")

    # Optionally show top 3 predictions
    if st.checkbox("Show top 3 predictions"):
        top_indices = pred[0].argsort()[-3:][::-1]
        for i in top_indices:
            st.write(f"- {class_names[i]} → {pred[0][i]*100:.2f}%")

else:
    st.warning("👆 Upload an image to get started.")
