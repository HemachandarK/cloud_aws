import streamlit as st
import numpy as np
import cv2
import tensorflow
import google.generativeai as genai
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import os
import json

os.system("pip uninstall -y opencv-python opencv-python-headless")
os.system("pip install opencv-python-headless==4.9.0.80")


# Set up Gemini API (Replace with your actual API key)
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

@st.cache_resource
def load_doctor_data():
    with open("https://demovijay.s3.us-east-1.amazonaws.com/details.json", "r") as file:
        return json.load(file)

doctor_data = load_doctor_data()

# Session state initialization
if "page" not in st.session_state:
    st.session_state["page"] = "home"

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
    Translate the response into {lang}.
    The X-ray analysis results:
    - Body Part: {fracture_info['body_part']}
    - Fracture Status: {fracture_info['fracture_status']}
    - Fracture Severity: {fracture_info['severity']}
    - Estimated Healing Time: {fracture_info['healing_time']}
    - Patient Age: {age}
    
    User Query: "{user_query}"
    """
    response = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
    return response.text if response else "Sorry, I couldn't process your request."


def home_page():
    st.title("XrayGuard: AI-Powered Fracture Detection")
    uploaded_file = st.file_uploader("Upload an X-ray Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-ray Image", use_column_width=True)
        
        body_part = predict_body_part(image)
        fracture_status = predict_fracture(image, body_part)

        st.write(f"Predicted Body Part: {body_part}")
        st.write(f"Fracture Status: {fracture_status}")
        
        if fracture_status == "Fractured":
            severity_levels = ["Mild", "Moderate", "Severe"]
            severity = np.random.choice(severity_levels, p=[0.4, 0.4, 0.2])

            st.write(f"Fracture Severity: {severity}")
            
            age = st.text_input("Enter Patient's Age", placeholder="e.g., 25")
            language = st.text_input("Enter Preferred Language", placeholder="e.g., Hindi, Tamil, Spanish")

            healing_time = estimate_healing_time(severity, age)

            st.subheader("Chat with the Fracture Bot 🤖")
            user_query = st.text_input("Ask me anything about the X-ray!")

            if user_query and language:
                fracture_info = {
                    "body_part": body_part,
                    "fracture_status": fracture_status,
                    "severity": severity,
                    "healing_time": healing_time
                }
                response = query_gemini(user_query, fracture_info, age, language)
                st.write(f"🗨 Bot: {response}")
        
        else:
            st.success("✅ No Fracture Detected.")
    if st.button("📞 Connect to Doctor"):
        st.session_state["page"] = "doctors"
        st.rerun()

def doctor_page():
    st.title("Available Doctors ")

    for doctor in doctor_data["doctors"]:
        st.subheader(doctor["name"])
        st.write(f"**Specialization:** {doctor['specialization']}")
        st.write(f"**Rating:**  {doctor['rating']}/5")
        st.write(f"**Consultation Fee:** ₹{doctor['charge']}")

        if st.button(f"Connect with {doctor['name']}"):
            st.success(" The doctor will contact you shortly!")

    if st.button("⬅ Back to Home"):
        st.session_state["page"] = "home"
        st.rerun()

def main():
    if st.session_state["page"] == "home":
        home_page()
    else:
        doctor_page()

if __name__ == "__main__":
    main()