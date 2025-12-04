import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import requests
import os

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Quick Checker-Leaf Diseases",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Quick Checker-Leaf Diseases")
st.write("Upload a leaf image and the system will predict if it's Healthy or Diseased.")

FILE_ID = "16CydnM03zA-NpjfV5oIahxjJq94IqLex"
MODEL_PATH = "model.pth"


# ===============================
# Download Large Google Drive File
# ===============================
def download_from_google_drive(file_id, destination):
    """Handles large file downloads from Google Drive."""
    URL = "https://drive.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)

    # Detect confirmation token
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    # Save the model file
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)


def ensure_model_exists():
    if not os.path.exists(MODEL_PATH):
        st.warning("Downloading model from Google Drive… This may take 10–30 seconds ⏳")

        download_from_google_drive(FILE_ID, MODEL_PATH)

        st.success("Model downloaded successfully ✔")


# ===============================
# Load Model
# ===============================
@st.cache_resource
def load_model():
    ensure_model_exists()

    device = torch.device("cpu")

    num_classes = 2
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    return model


model = load_model()
class_names = ['Diseased', 'Healthy']


# ===============================
# Transform
# ===============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ===============================
# Upload Section
# ===============================
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf", use_column_width=True)

    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, idx = torch.max(probs, 1)

    predicted_class = class_names[idx.item()]
    confidence = conf.item() * 100

    st.success(f"Prediction: **{predicted_class}**")
    st.info(f"Confidence: **{confidence:.2f}%**")
