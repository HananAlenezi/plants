import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from streamlit.components.v1 import html
import torchvision.models as models
import os




st.title("Web")
st.write("PyTorch version:", torch.__version__)


# Model file names
MODEL_FILES = {
    "EfficientNet-B0": "efficientnet_b0_retrained.pth",
    "MobileNetV2": "mobilenet_v2_retrained.pth",
    "ResNet50": "resnet50_retrained.pth",
    "AlexNet": "alexnet_retrained.pth"
}

PRETRAINED_FILES = {
    "EfficientNet-B0": "efficientnet_b0.pth",
    "MobileNetV2": "mobilenet_v2.pth",
    "ResNet50": "resnet50.pth",
    "AlexNet": "alexnet.pth"
}

# === Class names ===
CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy","background",
]

# === Image transformation (matching training preprocessing) ===
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === Load model function ===
def load_model(model_name):
    """Loads a model architecture, applies pretrained and retrained weights."""
    
    # Paths to the model files
    pretrained_path = os.path.join(PRETRAINED_FILES[model_name])
    retrained_path = os.path.join(MODEL_FILES[model_name])
    
    # Define model architecture
    if model_name == "EfficientNet-B0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
    elif model_name == "MobileNetV2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
    elif model_name == "ResNet50":
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    elif model_name == "AlexNet":
        model = models.alexnet(weights=None)
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, len(CLASS_NAMES))
    else:
        raise ValueError("⚠️ Unknown model type!")

    # Load pretrained weights
    # Load pretrained weights
    if os.path.exists(pretrained_path):
        pretrained_state = torch.load(pretrained_path, map_location=torch.device('cpu'), weights_only=False)

        # Remove classifier weights to prevent mismatch
        if model_name == "EfficientNet-B0":
            pretrained_state = {k: v for k, v in pretrained_state.items() if "classifier.1" not in k}
        elif model_name == "AlexNet":
            pretrained_state = {k: v for k, v in pretrained_state.items() if "classifier.6" not in k}
        elif model_name == "MobileNetV2":
            pretrained_state = {k: v for k, v in pretrained_state.items() if "classifier.1" not in k}
        else:  # ResNet50 and others
            pretrained_state = {k: v for k, v in pretrained_state.items() if "fc" not in k}

        # Load only feature extractor weights
        model.load_state_dict(pretrained_state, strict=False)
    else:
        st.error(f"⚠️ Pretrained model not found: {pretrained_path}")
        return None



    # Load retrained weights
    if os.path.exists(retrained_path):
        retrained_state = torch.load(retrained_path, map_location=torch.device('cpu'), weights_only=False)
        new_state_dict = {k.replace("module.", ""): v for k, v in retrained_state.items()}  # Remove 'module.' prefix if needed
        model.load_state_dict(new_state_dict, strict=False)
    else:
        st.warning(f"⚠️ Retrained model not found. Using base pretrained model: {model_name}")

    model.eval()
    return model

# === Classify an image ===
def classify_image(image, model_name):
    """Predicts the class of an image using the selected model."""
    
    model = load_model(model_name)
    if model is None:
        return "Error loading model!", None

    # Preprocess the image
    image = image.convert("RGB")
    image = TRANSFORM(image).unsqueeze(0)

    # Perform classification
    with torch.no_grad():
        output = model(image)
        predicted_class = torch.argmax(output, 1).item()

    return CLASS_NAMES[predicted_class], predicted_class

# === Streamlit UI ===
st.title("🌿 Leaf Disease Classification")
st.write("Upload an image of a plant leaf to classify its disease using deep learning models.")

use_camera = st.checkbox("Use Camera")
picture = None
if use_camera:
   picture = st.camera_input("Take a photo")

   if picture and st.button("Flip Image"):
       image = Image.open(picture)
       flipped_image = ImageOps.mirror(image)
       st.image(flipped_image, caption="Flip", width=300)
       
# Upload image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image or picture:
   image = picture if picture else uploaded_image
   image = Image.open(image)
   st.image(image, caption="Your Image", width=300)

       # Choose a model
selected_model = st.selectbox("Select a Model", list(MODEL_FILES.keys()))

if st.button("Classify Image"):
        # Perform classification
        predicted_class, _ = classify_image(image, selected_model)

        # Display prediction
        st.write(f"### 🎯 Plant Status: `{predicted_class}`")
