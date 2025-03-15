import streamlit as st
import json
from PIL import Image

# Load Doctor Details from JSON
def load_doctor_data():
    with open("details.json", "r") as file:
        return json.load(file)

doctor_data = load_doctor_data()

# Initialize session state for navigation
if "page" not in st.session_state:
    st.session_state["page"] = "home"

# Function to Display Home Page
def home_page():
    st.title("Bone Fracture Detection Chatbot 🤖")

    uploaded_file = st.file_uploader("Upload an X-ray Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-ray Image", use_column_width=True)
        
        # Simulated Predictions
        body_part = "Hand"
        fracture_status = "Fractured"
        
        st.write(f"**Predicted Body Part:** {body_part}")
        st.write(f"**Fracture Status:** {fracture_status}")
        
        if fracture_status == "Fractured":
            st.error("Fracture detected! Consider consulting a doctor.")

        # Connect with Doctor Button
        if st.button("Connect with a Doctor"):
            st.session_state["page"] = "doctors"
            st.rerun()  # Fixed from st.experimental_rerun()

# Function to Display Doctor Page
def doctor_page():
    st.title("Available Doctors 👨‍⚕️👩‍⚕️")

    for doctor in doctor_data["doctors"]:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(doctor["photo"], width=100)
        with col2:
            st.subheader(doctor["name"])
            st.write(f"**Specialization:** {doctor['specialization']}")
            st.write(f"**Rating:** ⭐ {doctor['rating']}/5")
            st.write(f"**Consultation Fee:** ₹{doctor['charge']}")

        if st.button(f"Connect with {doctor['name']}"):
            st.success("✅ The doctor will contact you shortly!")

    if st.button("⬅ Back to Home"):
        st.session_state["page"] = "home"
        st.rerun()  # Fixed from st.experimental_rerun()

# Main function to control navigation
def main():
    if st.session_state["page"] == "home":
        home_page()
    elif st.session_state["page"] == "doctors":
        doctor_page()

if __name__ == "__main__":
    main()
