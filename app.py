import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import time

########################
#     MODEL DEFINITION
########################
class MelanomaModel(nn.Module):
    def __init__(self, out_size, dropout_prob=0.5):
        super(MelanomaModel, self).__init__()
        from efficientnet_pytorch import EfficientNet
        self.efficient_net = EfficientNet.from_pretrained('efficientnet-b0')
        # Remove the original FC layer
        self.efficient_net._fc = nn.Identity()
        
        self.fc1 = nn.Linear(1280, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, out_size)
        
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.efficient_net(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


########################
#     DIAGNOSIS MAP
########################
DIAGNOSIS_MAP = {
    0: 'Melanoma',
    1: 'Melanocytic nevus',
    2: 'Basal cell carcinoma',
    3: 'Actinic keratosis',
    4: 'Benign keratosis',
    5: 'Dermatofibroma',
    6: 'Vascular lesion',
    7: 'Squamous cell carcinoma',
    8: 'Unknown'
}

########################
#   LOAD MODEL FUNCTION
########################
@st.cache_resource
def load_model():
    """
    Loads the model checkpoint.
    Using weights_only=False (if you trust the .pth file).
    If you prefer a more secure approach, re-save your checkpoint 
    to only contain raw state_dict and set weights_only=True.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MelanomaModel(out_size=9)

    # Path to your model file
    model_path = os.path.join("model", "multi_weight.pth")

    # If you trust the checkpoint file, set weights_only=False
    checkpoint = torch.load(
        model_path, 
        map_location=device,
        weights_only=False  # if you have a purely raw state_dict, you can use True
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()
    
    return model, device

########################
#   IMAGE TRANSFORM
########################
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

########################
#   PREDICTION UTILS
########################
def predict_skin_lesion(img: Image.Image, model: nn.Module, device: torch.device):
    # Transform and move image to device
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        top_probs, top_idxs = torch.topk(probs, 3, dim=1)  # top 3 predictions

    predictions = []
    for prob, idx in zip(top_probs[0], top_idxs[0]):
        label = DIAGNOSIS_MAP.get(idx.item(), "Unknown")
        confidence = prob.item() * 100
        predictions.append((label, confidence))

    return predictions

########################
#   PAGE CONFIG & STYLE
########################
st.set_page_config(
    page_title="Skin Lesion Classifier",
    page_icon=":microscope:",
    layout="centered",
    initial_sidebar_state="expanded"
)

def set_background_color():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #FDEAE0;  /* A pale peach/light skin tone */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_background_color()

########################
#   STREAMLIT APP
########################
def main():
    st.title("Skin Lesion Classifier")
    st.write("Upload an image of a skin lesion to see the top-3 predicted diagnoses.")

    # Create a stylish sidebar
    st.sidebar.title("Possible Diagnoses")
    st.sidebar.markdown("Here are the categories the model can distinguish:")
    for idx, diag in DIAGNOSIS_MAP.items():
        st.sidebar.markdown(f"- **{diag}**")

    # Add the names to the sidebar in a new section
    st.sidebar.title("Team Members")
    st.sidebar.markdown(
        """
        - **PRATHUSH MON**
        - **PRATIK J**
        - **RAYAN NASAR**
        - **R HARIMURALI**
        - **WASEEM AHAMMED**
        """
    )

    # Load the model once (cached)
    model, device = load_model()

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Display the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Predict on button click
        if st.button("Classify"):
            with st.spinner("Analyzing..."):
                time.sleep(3)  # 3-second spinner
                results = predict_skin_lesion(image, model, device)
            
            st.subheader("Top-3 Predictions")
            for i, (diagnosis, confidence) in enumerate(results, start=1):
                st.write(f"{i}. **{diagnosis}**: {confidence:.2f}%")

if __name__ == "__main__":
    main()
