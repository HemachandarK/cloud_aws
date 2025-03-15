import streamlit as st
import torch
import pymongo
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries


MONGO_URI = "mongodb+srv://sneelulatha2005:CFwDtfRS2yjR644x@feedback.oq3u0.mongodb.net/?retryWrites=true&w=majority&appName=feedback"
client = pymongo.MongoClient(MONGO_URI)
db = client["PneumoniaDetection"]  # Database Name
feedback_collection = db["Feedback"]  # Collection Name


# Load the trained model
@st.cache_resource()
def load_model():
    model = models.resnet101(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 2)  # Binary classification (Pneumonia / Normal)
    )
    model.load_state_dict(torch.load("pneumonia.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()
device = torch.device("cpu")

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Grad-CAM
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor, class_idx=None):
        input_tensor = input_tensor.unsqueeze(0).to(device)
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output)
        self.model.zero_grad()
        output[:, class_idx].backward()
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        return cam

# Overlay heatmap
def overlay_heatmap(image, heatmap):
    heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    image_np = np.array(image)
    superimposed_img = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)
    return superimposed_img

# LIME Explainability
explainer = lime_image.LimeImageExplainer()

def preprocess_lime(image):
    return transform(image).unsqueeze(0)

def predict_lime(images):
    batch = torch.stack([preprocess_lime(Image.fromarray(img)) for img in images]).squeeze().to(device)
    outputs = model(batch)
    return torch.nn.functional.softmax(outputs, dim=1).detach().cpu().numpy()

# Streamlit UI
st.title("Pneumonia Detection with Explainability")
st.write("Upload a chest X-ray to analyze for pneumonia and visualize the AI's decision-making.")

uploaded_file = st.file_uploader("Upload an X-ray image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)
    
    input_tensor = transform(image)
    target_layer = model.layer4[-1]
    grad_cam = GradCAM(model, target_layer)
    heatmap = grad_cam.generate_heatmap(input_tensor)
    gradcam_result = overlay_heatmap(image, heatmap)
    
    explanation = explainer.explain_instance(np.array(image), predict_lime, top_labels=2, hide_color=0, num_samples=200)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=False)
    
    st.subheader("Grad-CAM Heatmap")
    st.image(gradcam_result, caption="Grad-CAM Visualization", use_column_width=True)
    
    st.subheader("LIME Explanation")
    st.image(mark_boundaries(temp, mask), caption="LIME Superpixel Explanation", use_column_width=True)
    
    # Textual Interpretation
    output_probs = torch.nn.functional.softmax(model(input_tensor.unsqueeze(0).to(device)), dim=1).detach().cpu().numpy()
    prediction = np.argmax(output_probs)
    confidence = output_probs[0][prediction] * 100
    class_labels = ["Normal", "Pneumonia"]
    
    st.subheader("Model Interpretation")
    st.write(f"Prediction: **{class_labels[prediction]}**")
    st.write(f"Confidence Score: **{confidence:.2f}%**")
    
    if prediction == 1:
        st.write("The model predicts pneumonia. The Grad-CAM heatmap highlights areas contributing to this decision, while LIME explains which regions influenced the outcome.")
    else:
        st.write("The model predicts no pneumonia. The heatmap and LIME analysis show the decision-making process for transparency.")

st.subheader("Expert Feedback")
feedback = st.text_area("Provide feedback on the model's prediction:")
if st.button("Submit Feedback"):
    feedback_entry = {
        "prediction": class_labels[prediction],
        "confidence": confidence,
        "feedback": feedback
    }
    feedback_collection.insert_one(feedback_entry)
    st.success("Feedback submitted successfully and stored in MongoDB Atlas!")