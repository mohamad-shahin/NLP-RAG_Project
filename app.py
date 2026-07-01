"""
RetinaScan AI — Retinal Disease Classification Interface
Upload a fundus (retina) photo and get an instant multi-label risk assessment
for Diabetic Retinopathy, Optic Disc Cupping, and Tessellation, plus an
overall disease-risk flag.
"""

import os
import io
import numpy as np
import cv2
import streamlit as st
from PIL import Image

# ----------------------------------------------------------------------------
# Page config — must be first Streamlit call
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
MODEL_PATH = os.path.join("model", "retinal_model.h5")
# Optional: if you host the weights externally (e.g. Hugging Face Hub) instead
# of committing a large binary to GitHub, set this URL and the app will
# download the file automatically on first run.
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

THRESHOLD = 0.7  # matches the threshold used during model development

# ----------------------------------------------------------------------------
# Styling — distinctive, clinical-but-warm theme
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --ink: #0f2e35;
        --teal: #0e7c86;
        --teal-light: #e6f5f6;
        --amber: #d97706;
        --amber-light: #fff4e5;
        --danger: #c0392b;
        --danger-light: #fdecea;
    }
    .stApp {
        background: linear-gradient(180deg, #f6fbfb 0%, #ffffff 220px);
    }
    #MainMenu, footer, header {visibility: hidden;}

    .hero {
        text-align: center;
        padding: 1.2rem 0 0.4rem 0;
    }
    .hero h1 {
        font-size: 2.1rem;
        font-weight: 800;
        color: var(--ink);
        margin-bottom: 0.1rem;
        letter-spacing: -0.02em;
    }
    .hero p {
        color: #4b6a70;
        font-size: 1.02rem;
        margin-top: 0;
    }
    .badge-row {
        display: flex;
        justify-content: center;
        gap: 0.5rem;
        margin-bottom: 1.2rem;
        flex-wrap: wrap;
    }
    .pill {
        background: var(--teal-light);
        color: var(--teal);
        border-radius: 999px;
        padding: 0.25rem 0.8rem;
        font-size: 0.78rem;
        font-weight: 600;
    }

    .result-card {
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.75rem;
        border: 1px solid rgba(15,46,53,0.08);
    }
    .result-card.flagged {
        background: var(--danger-light);
        border-color: rgba(192,57,43,0.25);
    }
    .result-card.clear {
        background: var(--teal-light);
        border-color: rgba(14,124,134,0.2);
    }
    .result-top {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
    }
    .result-name {
        font-weight: 700;
        font-size: 1.05rem;
        color: var(--ink);
    }
    .result-score {
        font-weight: 800;
        font-size: 1.05rem;
    }
    .result-score.flagged { color: var(--danger); }
    .result-score.clear { color: var(--teal); }
    .result-desc {
        color: #5a7278;
        font-size: 0.85rem;
        margin-top: 0.15rem;
    }
    .status-tag {
        display: inline-block;
        margin-top: 0.4rem;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        padding: 0.15rem 0.55rem;
        border-radius: 6px;
    }
    .status-tag.flagged { background: var(--danger); color: white; }
    .status-tag.clear { background: var(--teal); color: white; }

    .disclaimer {
        background: var(--amber-light);
        border: 1px solid rgba(217,119,6,0.25);
        border-radius: 12px;
        padding: 0.8rem 1rem;
        font-size: 0.82rem;
        color: #7a4a06;
        margin-top: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    import tensorflow as tf

    if not os.path.exists(MODEL_PATH):
        if MODEL_URL:
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            import urllib.request

            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        else:
            return None
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def preprocess(image: Image.Image) -> np.ndarray:
    """Match the exact preprocessing used during training."""
    from tensorflow.keras.applications.efficientnet import preprocess_input

    img = np.array(image.convert("RGB"))
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.expand_dims(img, axis=0).astype("float32")
    img = preprocess_input(img)
    return img


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>👁️ RetinaScan AI</h1>
        <p>Upload a retinal fundus photo for an instant, AI-assisted risk screening.</p>
    </div>
    <div class="badge-row">
        <span class="pill">EfficientNetB4</span>
        <span class="pill">4 conditions screened</span>
        <span class="pill">Runs in seconds</span>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### About")
    st.write(
        "RetinaScan AI screens fundus (retina) photographs for signs of "
        "diabetic retinopathy, optic disc cupping, tessellation, and "
        "overall disease risk, using a deep learning model built on "
        "EfficientNetB4."
    )
    st.markdown("### Conditions screened")
    for label in LABELS:
        st.markdown(f"**{label['icon']} {label['name']}**  \n{label['desc']}")
    st.markdown("---")
    st.caption(
        "This tool is for educational and research purposes only and is "
        "not a substitute for professional medical diagnosis."
    )

# ----------------------------------------------------------------------------
# Main interaction
# ----------------------------------------------------------------------------
model = load_model()

if model is None:
    st.error(
        "**Model weights not found.**\n\n"
        "This interface is fully built, but no trained model file was found at "
        f"`{MODEL_PATH}`. Add your trained `.h5`/`.keras` file to the `model/` "
        "folder (or set the `MODEL_URL` environment variable to auto-download "
        "it) and redeploy. See the README for step-by-step instructions."
    )
else:
    uploaded_file = st.file_uploader(
        "Upload a retinal fundus image",
        type=["png", "jpg", "jpeg"],
        help="Use a clear, centered fundus photograph for best results.",
    )

    if uploaded_file is not None:
        image = Image.open(io.BytesIO(uploaded_file.read()))

        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Uploaded image", use_container_width=True)

        with st.spinner("Analyzing image..."):
            input_tensor = preprocess(image)
            predictions = model.predict(input_tensor, verbose=0)[0]

        with col2:
            st.markdown("#### Results")
            for label, score in zip(LABELS, predictions):
                flagged = score > THRESHOLD
                state = "flagged" if flagged else "clear"
                st.markdown(
                    f"""
                    <div class="result-card {state}">
                        <div class="result-top">
                            <span class="result-name">{label['icon']} {label['name']}</span>
                            <span class="result-score {state}">{score*100:.1f}%</span>
                        </div>
                        <div class="result-desc">{label['desc']}</div>
                        <span class="status-tag {state}">{"Signs Detected" if flagged else "No Signs Detected"}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown(
            """
            <div class="disclaimer">
                ⚠️ <strong>Not a medical diagnosis.</strong> This tool provides an
                AI-generated screening estimate only. Please consult an
                ophthalmologist or qualified healthcare provider for an
                accurate diagnosis and treatment plan.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("👆 Upload a retinal image above to get started.")
