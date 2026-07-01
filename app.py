"""
RetinaScan AI — Retinal Disease Classification Interface
Upload a fundus (retina) photo and get an instant multi-label risk assessment.
"""

import os
import io
import numpy as np
import cv2
import streamlit as st
from PIL import Image

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="RetinaScan AI",
    page_icon="👁️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
IMG_SIZE = 380

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "retina_model.h5")

# ضع رابط المودل هنا (HuggingFace / Drive / أي رابط مباشر)
MODEL_URL = os.environ.get("MODEL_URL", "")

LABELS = [
    {
        "key": "Disease_Risk",
        "name": "Overall Disease Risk",
        "desc": "General likelihood that the retina shows disease-related abnormalities.",
        "icon": "⚠️",
    },
    {
        "key": "DR",
        "name": "Diabetic Retinopathy",
        "desc": "Damage to retinal blood vessels caused by diabetes.",
        "icon": "🩸",
    },
    {
        "key": "ODC",
        "name": "Optic Disc Cupping",
        "desc": "Enlargement of the optic cup, often associated with glaucoma.",
        "icon": "⭕",
    },
    {
        "key": "TSLN",
        "name": "Tessellation",
        "desc": "A tigroid / tessellated pattern in the retina, often linked to myopia.",
        "icon": "🔶",
    },
]

THRESHOLD = 0.7

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #f6fbfb 0%, #ffffff 220px);
}
#MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():

    import tensorflow as tf
    import urllib.request

    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):

        if MODEL_URL:
            st.info("📥 Downloading model for first run...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        else:
            return None

    return tf.keras.models.load_model(MODEL_PATH, compile=False)


# ----------------------------------------------------------------------------
# Preprocess
# ----------------------------------------------------------------------------
def preprocess(image: Image.Image) -> np.ndarray:
    from tensorflow.keras.applications.efficientnet import preprocess_input

    img = np.array(image.convert("RGB"))
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.expand_dims(img, axis=0).astype("float32")
    img = preprocess_input(img)
    return img


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("👁️ RetinaScan AI")
st.write("Upload a retinal image to get AI-based screening results.")

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.write("AI model for retinal disease screening using EfficientNetB4.")

    st.subheader("Conditions")
    for l in LABELS:
        st.write(f"{l['icon']} {l['name']}")
        st.caption(l["desc"])

# ----------------------------------------------------------------------------
# Load model
# ----------------------------------------------------------------------------
model = load_model()

if model is None:
    st.error(
        "❌ Model not found.\n"
        "Please upload model.h5 to HuggingFace or set MODEL_URL."
    )

else:
    uploaded_file = st.file_uploader(
        "Upload retinal image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file:
        image = Image.open(io.BytesIO(uploaded_file.read()))

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Input Image", use_container_width=True)

        with st.spinner("Analyzing..."):
            input_tensor = preprocess(image)
            predictions = model.predict(input_tensor, verbose=0)[0]

        with col2:
            st.subheader("Results")

            for label, score in zip(LABELS, predictions):
                flagged = score > THRESHOLD

                st.markdown(f"""
                **{label['icon']} {label['name']}**  
                Score: `{score*100:.2f}%`  
                Status: {"🚨 Detected" if flagged else "✅ Normal"}
                ---
                """)

    else:
        st.info("Upload an image to start prediction.")
