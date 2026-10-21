"""Streamlit interface for trying a locally trained model on one patch."""
from pathlib import Path
import streamlit as st
import torch
from PIL import Image
from src.pcam_classifier.data import image_transform
from src.pcam_classifier.model import PcamCNN
ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="PCam Patch Classifier", page_icon="🔬")
st.title("PCam Histopathology Patch Classifier")
st.caption("Educational research demonstration. Not a diagnostic tool.")
model_path = Path(st.text_input("Checkpoint path", str(ROOT / "models" / "pcam_cnn.pt"))).expanduser()
uploaded = st.file_uploader("Upload a histology patch", type=["png", "jpg", "jpeg", "tif", "tiff"])

@st.cache_resource
def load_model(path_text):
    path = Path(path_text)
    if not path.exists():
        return None, f"Checkpoint not found: {path}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = PcamCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return (model, device), None

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Input patch", width=320)
    loaded, error = load_model(str(model_path))
    if error:
        st.info(error + " Train a model using the README instructions.")
    else:
        model, device = loaded
        tensor = image_transform()(image).unsqueeze(0).to(device)
        with torch.inference_mode():
            probabilities = model(tensor).softmax(1)[0]
        tumor_probability = float(probabilities[1])
        st.metric("Predicted tumor probability", f"{tumor_probability:.1%}")
        st.write("Predicted patch label:", "tumor present" if tumor_probability >= 0.5 else "no tumor detected")
        st.caption("Patch-level prediction only. Do not use for patient care.")
else:
    st.write("Upload a patch to view a model prediction.")
