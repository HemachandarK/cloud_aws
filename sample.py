import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import google.generativeai as genai
from PIL import Image
import json
from gtts import gTTS
import tempfile
import gc
from concurrent.futures import ThreadPoolExecutor

# Security: Use Streamlit secrets for API keys
GEMINI_API_KEY = "AIzaSyDD0BzkWZhyylV-l6euP8s3shySnkPPnug"
genai.configure(api_key=GEMINI_API_KEY)

# Optimized Model Loading with TFLite
@st.cache_resource(show_spinner=False)
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

@st.cache_resource
def load_all_models():
    with ThreadPoolExecutor() as executor:
        futures = {
            'body_part': executor.submit(load_tflite_model, 'ResNet50_BodyParts.tflite'),
            'hand': executor.submit(load_tflite_model, 'ResNet50_Hand_frac.tflite'),
            'elbow': executor.submit(load_tflite_model, 'ResNet50_Elbow_frac.tflite'),
            'shoulder': executor.submit(load_tflite_model, 'ResNet50_Shoulder_frac.tflite')
        }
        return {key: future.result() for key, future in futures.items()}

models = load_all_models()

# Model configuration
body_part_labels = ["Elbow", "Hand", "Shoulder"]
fracture_labels = ["Fractured", "Normal"]

# Load Doctor Details
@st.cache_resource
def load_doctor_data():
    with open("details.json", "r") as file:
        return json.load(file)

doctor_data = load_doctor_data()

# Session state initialization
if "page" not in st.session_state:
    st.session_state.page = "home"

# Optimized Image Preprocessing
def preprocess_image(image):
    try:
        img = np.array(image.convert("L"), dtype=np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)
        img = cv2.resize(img, (224, 224))
        return np.repeat(img[..., np.newaxis], 3, axis=-1).astype(np.float32)
    except Exception as e:
        st.error(f"Image processing error: {str(e)}")
        return None

# Optimized Prediction Functions
def predict_with_tflite(interpreter, image):
    try:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], image[np.newaxis, ...])
        interpreter.invoke()
        
        return interpreter.get_tensor(output_details[0]['index'])
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def predict_body_part(image):
    processed_img = preprocess_image(image)
    if processed_img is None:
        return "Error"
    prediction = predict_with_tflite(models['body_part'], processed_img)
    return body_part_labels[np.argmax(prediction)] if prediction is not None else "Error"

def predict_fracture(image, body_part):
    processed_img = preprocess_image(image)
    if processed_img is None or body_part not in models:
        return "Error"
    
    prediction = predict_with_tflite(models[body_part.lower()], processed_img)
    return fracture_labels[np.argmax(prediction)] if prediction is not None else "Error"

# Rest of the functions remain similar with additional error handling
# ... (keep estimate_healing_time, query_gemini, text_to_speech functions from previous version)

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

def text_to_speech(text, lang_code):
    try:
        tts = gTTS(text=text, lang=lang_code)
    except:
        tts = gTTS(text=text, lang="en")  # Default to English if language is unsupported
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name

def memory_cleanup():
    tf.keras.backend.clear_session()
    gc.collect()

def home_page():
    st.title("Bone Fracture Detection Chatbot")
    uploaded_file = st.file_uploader("Upload an X-ray Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        try:
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
            
            with st.spinner("Analyzing X-ray..."):
                body_part = predict_body_part(image)
                fracture_status = predict_fracture(image, body_part)

            st.write(f"**Predicted Body Part:** {body_part}")
            st.write(f"**Fracture Status:** {fracture_status}")
            
            if fracture_status == "Fractured":
                severity = st.selectbox("Select Fracture Severity", 
                                       ["Mild", "Moderate", "Severe"],
                                       index=1)
                
                age = st.number_input("Patient's Age", min_value=1, max_value=120, value=25)
                language = st.selectbox("Preferred Language", 
                                      ["English", "Hindi", "Tamil", "Spanish"])

                healing_time = estimate_healing_time(severity, age)
                st.write(f"**Estimated Healing Time:** {healing_time}")

                if user_query := st.text_input("Ask about your X-ray results:"):
                    with st.spinner("Generating response..."):
                        fracture_info = {
                            'body_part': body_part,
                            'fracture_status': fracture_status,
                            'severity': severity,
                            'healing_time': healing_time
                        }
                        response = query_gemini(user_query, fracture_info, age, language)
                        st.write(f"**Bot:** {response}")
                        if audio_file := text_to_speech(response, language[:2].lower()):
                            st.audio(audio_file, format="audio/mp3")

            else:
                st.success("No Fracture Detected")
                
            memory_cleanup()

        except Exception as e:
            st.error(f"Processing failed: {str(e)}")

    if st.button("Connect to Doctor"):
        st.session_state.page = "doctors"
        st.rerun()

# Keep doctor_page() and main() similar to previous version
# ... (same doctor page implementation with error handling)

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
    st.set_page_config(
        page_title="Bone Fracture Analyzer",
        page_icon="🦴",
        layout="wide"
    )
    
    if st.session_state.page == "home":
        home_page()
    else:
        doctor_page()

if __name__ == "__main__":
    main()