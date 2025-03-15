import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import google.generativeai as genai
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import os
from gtts import gTTS
import tempfile
import base64

# Set up Gemini API (Replace "YOUR_API_KEY" with your actual key)
GEMINI_API_KEY = "AIzaSyDD0BzkWZhyylV-l6euP8s3shySnkPPnug"
genai.configure(api_key=GEMINI_API_KEY)

@st.cache_resource
def load_body_part_model():
    return load_model("ResNet50_BodyParts.h5")

@st.cache_resource
def load_hand_model():
    return load_model("ResNet50_Hand_frac.h5")

@st.cache_resource
def load_elbow_model():
    return load_model("ResNet50_Elbow_frac.h5")

@st.cache_resource
def load_shoulder_model():
    return load_model("ResNet50_Shoulder_frac.h5")

body_part_model = load_body_part_model()
hand_model = load_hand_model()
elbow_model = load_elbow_model()
shoulder_model = load_shoulder_model()

body_part_labels = ["Elbow", "Hand", "Shoulder"]
fracture_labels = ["Fractured", "Normal"]

# Get preferred language from the user
language = st.text_input("Enter your preferred language (e.g., English, Tamil, Hindi, Spanish)", "English")
language_code = language.lower()[:2]  # Take the first two letters as a rough language code

def preprocess_image(image):
    image = image.convert("L")
    image = np.array(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image = clahe.apply(image)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=[0, -1])
    image = np.repeat(image, 3, axis=-1)
    image = preprocess_input(image)
    return image

def predict_body_part(image):
    processed_img = preprocess_image(image)
    prediction = body_part_model.predict(processed_img)
    return body_part_labels[np.argmax(prediction)]

def predict_fracture(image, body_part):
    processed_img = preprocess_image(image)
    model_dict = {"Hand": hand_model, "Elbow": elbow_model, "Shoulder": shoulder_model}
    model = model_dict.get(body_part)
    if model is None:
        return "Unknown"
    prediction = model.predict(processed_img)
    return fracture_labels[np.argmax(prediction)]

def detect_fracture_region(image):
    image_np = np.array(image.resize((224, 224)))
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    expansion = 10
    x = max(0, x - expansion)
    y = max(0, y - expansion)
    w = min(224, x + w + expansion) - x
    h = min(224, y + h + expansion) - y
    cv2.rectangle(image_np, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return Image.fromarray(image_np)

def estimate_healing_time(severity, age):
    healing_times = {
        "Mild": "2-4 weeks",
        "Moderate": "4-8 weeks",
        "Severe": "8-12+ weeks"
    }
    healing_time = healing_times.get(severity, "Unknown")
    
    if age:
        try:
            age = int(age)
            if age > 50:
                healing_time = "Healing may take longer than usual: " + healing_time
            elif age < 18:
                healing_time = "Healing is generally faster: " + healing_time
        except ValueError:
            pass
    
    return healing_time

def query_gemini(user_query, fracture_info, age, lang):
    prompt = f"""
    Translate the response into {lang}. The X-ray analysis results:
    - Body Part: {fracture_info['body_part']}
    - Fracture Status: {fracture_info['fracture_status']}
    - Fracture Severity: {fracture_info['severity']}
    - Estimated Healing Time: {fracture_info['healing_time']}
    - Patient Age: {age}
    
    User Query: "{user_query}"
    """
    response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
    return response.text if response else "Sorry, I couldn't process your request."

def text_to_speech(text, lang_code):
    try:
        tts = gTTS(text=text, lang=lang_code)
    except:
        tts = gTTS(text=text, lang="en")  # Default to English if language is unsupported
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name

st.title("Bone Fracture Detection Chatbot 🤖")
uploaded_file = st.file_uploader("Upload an X-ray Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray Image", use_column_width=True)
    
    body_part = predict_body_part(image)
    fracture_status = predict_fracture(image, body_part)

    st.write(f"Predicted Body Part: {body_part}")
    st.write(f"Fracture Status: {fracture_status}")

    if fracture_status == "Fractured":
        highlighted_image = detect_fracture_region(image)
        st.image(highlighted_image, caption="Fracture Location Highlighted", use_column_width=True)
        severity_levels = ["Mild", "Moderate", "Severe"]
        severity = np.random.choice(severity_levels, p=[0.4, 0.4, 0.2])
        
        age = st.text_input("Enter Patient's Age", placeholder="e.g., 25")
        
        healing_time = estimate_healing_time(severity, age)
        st.write(f"Fracture Severity: {severity}")
        st.write(f"⏳ Estimated Healing Time: {healing_time}")
        
        st.subheader("Chat with the Fracture Bot 🤖")
        user_query = st.text_input("Ask me anything about the X-ray!")

        if user_query:
            fracture_info = {
                "body_part": body_part,
                "fracture_status": fracture_status,
                "severity": severity,
                "healing_time": healing_time
            }
            response = query_gemini(user_query, fracture_info, age, language)
            st.write(f"🗨 Bot: {response}")
            audio_file = text_to_speech(response, language_code)
            with open(audio_file, "rb") as f:
                audio_bytes = f.read()
                st.audio(audio_bytes, format="audio/mp3")
    else:
        st.success("✅ No Fracture Detected.")
