import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import blob_log
from skimage.filters import gaussian
import warnings

st.set_page_config(layout="wide")
st.title("🔬 Bravais Lattice Classifier")

# ---------------- CONSTANTS ----------------
BRAVAIS_LABELS = {
    'cP': 0,  'cI': 1,  'cF': 2,
    'tP': 3,  'tI': 4,
    'oP': 5,  'oI': 6,  'oF': 7,  'oC': 8,
    'hP': 9,  'hR': 10,
    'mP': 11, 'mC': 12,
    'aP': 13,
}
LABEL_TO_BRAVAIS = {v: k for k, v in BRAVAIS_LABELS.items()}

PHYS_DIM = 128
TOP_N = 8
N_CLASSES = 14

DEVICE = "cpu"

# ---------------- MODEL ----------------
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(2, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.GELU(), nn.MaxPool2d(2),
        )
        self.layer1 = nn.Sequential(ResBlock(64), ResBlock(64))
        self.down1 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU())
        self.layer2 = nn.Sequential(ResBlock(128), ResBlock(128))
        self.down2 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.GELU())
        self.layer3 = nn.Sequential(ResBlock(256), ResBlock(256))
        self.down3 = nn.Sequential(nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.GELU())
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.down1(x)
        x = self.layer2(x); x = self.down2(x)
        x = self.layer3(x); x = self.down3(x)
        return self.pool(x).view(x.size(0), -1)

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.phys_net = nn.Sequential(
            nn.Linear(PHYS_DIM, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(512 + 128, 256), nn.GELU(),
            nn.Linear(256, N_CLASSES),
        )

    def forward(self, x, phys):
        return self.head(torch.cat([self.encoder(x), self.phys_net(phys)], dim=1))


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    encoder = Encoder(in_channels=2, feat_dim=512).to(DEVICE)
    model = Classifier(encoder).to(DEVICE)

    state_dict = torch.load("best_clf.pt", map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.eval()
    return model

# ---------------- HELPERS ----------------
def preprocess(image):
    img = image.astype(np.float32)
    if img.ndim == 3:
        img = img.mean(axis=2)
    img = gaussian(img, sigma=1.0, preserve_range=True)
    img = np.log(img + 1e-6)
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    return img

def extract_physics_features(image):
    return np.zeros(128, dtype=np.float32)  # ⚠️ replace with full function if needed

def add_fft(x):
    fft_mag = torch.log(torch.abs(torch.fft.fft2(x)) + 1e-6)
    fft_mag = (fft_mag - fft_mag.min()) / (fft_mag.max() - fft_mag.min() + 1e-6)
    return torch.cat([x, fft_mag], dim=0)

# ---------------- UI ----------------
uploaded_file = st.file_uploader("Upload .pt file", type=["pt"])

if uploaded_file:
    d = torch.load(uploaded_file, map_location=DEVICE)
    img = d['image'].float().numpy()

    img_tensor = add_fft(torch.from_numpy(img).unsqueeze(0)).unsqueeze(0)
    phys = extract_physics_features(img)

    with torch.no_grad():
        logits = model(img_tensor, torch.from_numpy(phys).unsqueeze(0))
        probs = F.softmax(logits, dim=1).squeeze()

    pred_idx = probs.argmax().item()
    pred_name = LABEL_TO_BRAVAIS[pred_idx]
    confidence = probs[pred_idx].item()

    st.success(f"Prediction: {pred_name} ({confidence:.2%})")

    # ---- Visualization ----
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        st.image(img, clamp=True)

    with col2:
        st.subheader("Class Probabilities")
        fig, ax = plt.subplots()
        ax.barh(list(BRAVAIS_LABELS.keys()), probs.numpy())
        st.pyplot(fig)
