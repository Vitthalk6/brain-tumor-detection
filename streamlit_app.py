import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from datetime import datetime
from zoneinfo import ZoneInfo
import matplotlib.pyplot as plt
import io
import zipfile
from huggingface_hub import hf_hub_download


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Brain Tumor Detection by Deep Learning",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CONSTANTS
# =========================================================

CLASSES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

IMG_SIZE = (224, 224)

MODEL_REPO = "Vitthalk/brain-tumor-efficientnet"
MODEL_FILE = "efficientnet_best.keras"


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background:
            radial-gradient(circle at 10% 10%, rgba(80, 120, 255, 0.10), transparent 30%),
            radial-gradient(circle at 90% 20%, rgba(180, 80, 255, 0.10), transparent 30%),
            linear-gradient(135deg, #07111f 0%, #0b1424 50%, #101827 100%);
    }

    /* Main container */
    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Hide Streamlit menu/footer */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* Hero */
    .hero {
        padding: 35px 30px;
        border-radius: 24px;
        margin-bottom: 25px;
        background: linear-gradient(
            135deg,
            rgba(20, 35, 65, 0.95),
            rgba(25, 20, 55, 0.95)
        );
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow: 0 15px 45px rgba(0,0,0,0.30);
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        margin: 0;
        letter-spacing: -1px;
    }

    .hero-subtitle {
        color: #aebbd0;
        font-size: 17px;
        margin-top: 10px;
    }

    .badge {
        display: inline-block;
        padding: 7px 14px;
        border-radius: 30px;
        background: rgba(100,130,255,0.15);
        border: 1px solid rgba(120,150,255,0.30);
        color: #b9c8ff;
        font-size: 13px;
        margin-top: 18px;
    }

    /* Cards */
    .card {
        background: rgba(18, 29, 48, 0.78);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 20px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.18);
    }

    .card-title {
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 7px;
    }

    .card-subtitle {
        color: #9eacc0;
        font-size: 14px;
        margin-bottom: 15px;
    }

    /* Result cards */
    .result-card {
        padding: 25px;
        border-radius: 20px;
        background: rgba(18, 29, 48, 0.85);
        border: 1px solid rgba(255,255,255,0.10);
        margin-bottom: 20px;
    }

    .result-label {
        color: #9eacc0;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .result-value {
        font-size: 30px;
        font-weight: 800;
        margin-top: 5px;
    }

    /* Info box */
    .info-box {
        padding: 18px 20px;
        border-radius: 15px;
        background: rgba(40, 70, 110, 0.22);
        border: 1px solid rgba(100,150,220,0.20);
        color: #c9d5e6;
        margin: 15px 0;
    }

    /* Footer */
    .custom-footer {
        text-align: center;
        color: #75839a;
        padding: 30px 10px 10px 10px;
        font-size: 13px;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(12, 22, 38, 0.65);
        border-radius: 18px;
        padding: 8px;
    }

    /* Metric styling */
    [data-testid="stMetric"] {
        background: rgba(18, 29, 48, 0.80);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 18px;
    }

    /* Buttons */
    .stButton > button,
    .stDownloadButton > button {
        border-radius: 12px;
        font-weight: 700;
        min-height: 48px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE
    )

    model = tf.keras.models.load_model(
        path,
        compile=False
    )

    return model


# =========================================================
# LOAD GRAD-CAM COMPONENTS
# IMPORTANT:
# This is the same structure used in your old working code.
# =========================================================

@st.cache_resource
def load_gradcam_parts():

    model = load_model()

    # Data augmentation layer
    augmentation = model.get_layer("data_augmentation")

    # EfficientNet backbone
    backbone = model.get_layer("efficientnetb0")

    # Classification layers
    gap = model.get_layer("global_average_pooling2d_1")
    dropout = model.get_layer("dropout_1")
    classifier = model.get_layer("dense_2")

    # EfficientNet Grad-CAM target layer
    top_conv = backbone.get_layer("top_conv")

    # Grad-CAM backbone model
    grad_backbone = tf.keras.Model(
        inputs=backbone.input,
        outputs=[
            top_conv.output,
            backbone.output
        ]
    )

    return (
        model,
        augmentation,
        grad_backbone,
        gap,
        dropout,
        classifier
    )


# =========================================================
# INITIALIZE
# =========================================================

try:

    (
        model,
        augmentation,
        grad_backbone,
        gap,
        dropout,
        classifier
    ) = load_gradcam_parts()

    MODEL_READY = True

except Exception as e:

    MODEL_READY = False
    MODEL_ERROR = str(e)


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict(image):

    image = image.convert("RGB")

    resized = image.resize(IMG_SIZE)

    x = np.expand_dims(
        np.asarray(resized, dtype=np.float32),
        axis=0
    )

    probs = model.predict(
        x,
        verbose=0
    )[0]

    idx = int(np.argmax(probs))

    return resized, x, probs, idx


# =========================================================
# GRAD-CAM FUNCTION
# =========================================================

def gradcam(x, class_index):

    try:

        x = tf.convert_to_tensor(
            x,
            dtype=tf.float32
        )

        with tf.GradientTape() as tape:

            # Same augmentation approach as old working code
            augmented = augmentation(
                x,
                training=False
            )

            # Get convolution output and backbone features
            conv, features = grad_backbone(
                augmented,
                training=False
            )

            # Pass backbone features through original classifier
            pooled = gap(features)

            dropped = dropout(
                pooled,
                training=False
            )

            predictions = classifier(
                dropped,
                training=False
            )

            score = predictions[
                :,
                class_index
            ]

        # Calculate gradients
        grads = tape.gradient(
            score,
            conv
        )

        if grads is None:
            return None

        # Average gradients across spatial dimensions
        weights = tf.reduce_mean(
            grads,
            axis=(1, 2)
        )[0]

        # First image in batch
        conv_output = conv[0]

        # Weighted combination of feature maps
        heat = tf.reduce_sum(
            conv_output * weights,
            axis=-1
        )

        # ReLU
        heat = tf.maximum(
            heat,
            0
        )

        # Normalize
        maximum = tf.reduce_max(
            heat
        )

        if float(maximum) > 0:

            heat = heat / maximum

        return heat.numpy()

    except Exception:

        return None


# =========================================================
# CREATE GRAD-CAM OVERLAY
# =========================================================

def overlay(image, heatmap):

    if heatmap is None:
        return None

    original = np.asarray(
        image.convert("RGB")
    ).astype(np.float32)

    height, width = original.shape[:2]

    heat = np.asarray(
        Image.fromarray(
            np.uint8(
                heatmap * 255
            )
        ).resize(
            (width, height)
        )
    ).astype(np.float32) / 255.0

    # Jet colormap
    colored = plt.get_cmap("jet")(
        heat
    )[..., :3] * 255

    # Blend original + heatmap
    result = (
        0.60 * original +
        0.40 * colored
    )

    result = np.clip(
        result,
        0,
        255
    )

    return Image.fromarray(
        np.uint8(result)
    )


# =========================================================
# REPORT GENERATION
# =========================================================

def report_file(
    resized,
    cam,
    predicted,
    confidence,
    probs
):

    # Indian Standard Time
    indian_time = datetime.now(
        ZoneInfo("Asia/Kolkata")
    )

    report = io.StringIO()

    report.write(
        "BRAIN TUMOR DETECTION BY DEEP LEARNING\n"
    )

    report.write(
        "=" * 55 + "\n"
    )

    report.write(
        f"Date/Time: "
        f"{indian_time.strftime('%Y-%m-%d %H:%M:%S IST')}\n"
    )

    report.write(
        f"Prediction: {predicted}\n"
    )

    report.write(
        f"Model Confidence: "
        f"{confidence:.2f}%\n\n"
    )

    report.write(
        "Class Probabilities:\n"
    )

    for name, probability in zip(
        CLASSES,
        probs
    ):

        report.write(
            f"{name}: "
            f"{float(probability) * 100:.2f}%\n"
        )

    report.write("\n")

    report.write(
        "DISCLAIMER:\n"
    )

    report.write(
        "This is a research/educational prototype "
        "and not a medical diagnosis. "
        "The output must not replace evaluation "
        "by a qualified medical professional.\n"
    )

    # ZIP file
    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        "w",
        zipfile.ZIP_DEFLATED
    ) as archive:

        # Text report
        archive.writestr(
            "prediction_report.txt",
            report.getvalue()
        )

        # Resized MRI
        image_buffer = io.BytesIO()

        resized.save(
            image_buffer,
            format="JPEG",
            quality=95
        )

        archive.writestr(
            "mri_224x224.jpg",
            image_buffer.getvalue()
        )

        # Grad-CAM
        if cam is not None:

            cam_buffer = io.BytesIO()

            cam.save(
                cam_buffer,
                format="JPEG",
                quality=95
            )

            archive.writestr(
                "gradcam.jpg",
                cam_buffer.getvalue()
            )

    return buffer.getvalue()


# =========================================================
# HERO SECTION
# =========================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            🧠 Brain Tumor Detection
        </div>

        <div class="hero-subtitle">
            Deep Learning Based MRI Classification System
        </div>

        <div class="badge">
            EfficientNetB0 • MRI Image Analysis • Grad-CAM
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# MODEL STATUS
# =========================================================

if not MODEL_READY:

    st.error(
        "⚠️ The AI model could not be loaded."
    )

    st.code(
        MODEL_ERROR
    )

    st.stop()


# =========================================================
# INTRODUCTION
# =========================================================

st.markdown(
    """
    <div class="info-box">
        Upload a brain MRI image and click
        <b>Analyze MRI</b> to run the deep learning model.
        The system provides a predicted class,
        model confidence, class probabilities and
        a Grad-CAM visualization.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# UPLOAD SECTION
# =========================================================

st.markdown(
    """
    <div class="card">

        <div class="card-title">
            📤 Upload MRI
        </div>

        <div class="card-subtitle">
            Select a JPG, JPEG or PNG brain MRI image.
            Analysis starts only after pressing the Analyze button.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# MANUAL ANALYSIS FORM
# =========================================================

with st.form(
    "mri_analysis_form",
    clear_on_submit=False
):

    uploaded = st.file_uploader(
        "Choose Brain MRI Image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        label_visibility="visible"
    )

    analyze = st.form_submit_button(
        "🔍 Analyze MRI",
        type="primary",
        use_container_width=True
    )


# =========================================================
# ANALYSIS
# =========================================================

if analyze:

    if uploaded is None:

        st.warning(
            "Please upload an MRI image first."
        )

        st.stop()

    try:

        image = Image.open(
            uploaded
        ).convert("RGB")

    except Exception:

        st.error(
            "Unable to read this image. "
            "Please upload a valid JPG, JPEG or PNG file."
        )

        st.stop()


    # =====================================================
    # DISPLAY ORIGINAL IMAGE
    # =====================================================

    st.markdown(
        """
        <div class="card-title">
            🖼️ Uploaded MRI
        </div>
        """,
        unsafe_allow_html=True
    )

    st.image(
        image,
        caption="Uploaded MRI",
        use_container_width=True
    )


    # =====================================================
    # RUN AI
    # =====================================================

    with st.spinner(
        "🧠 AI is analyzing the MRI..."
    ):

        resized, x, probs, idx = predict(
            image
        )

        predicted = CLASSES[idx]

        confidence = (
            float(probs[idx]) * 100
        )

        # Confidence description
        if confidence >= 80:

            level = (
                "High model confidence"
            )

        elif confidence >= 60:

            level = (
                "Moderate model confidence"
            )

        else:

            level = (
                "Low model confidence — "
                "interpret cautiously"
            )

        # Grad-CAM
        heatmap = gradcam(
            x,
            idx
        )

        cam = overlay(
            resized,
            heatmap
        )


    # =====================================================
    # PREDICTION RESULT
    # =====================================================

    if predicted == "No Tumor":

        st.success(
            "🟢 Model prediction: No Tumor"
        )

    else:

        st.warning(
            f"🟠 Model prediction: {predicted}"
        )


    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )


    # =====================================================
    # RESULT METRICS
    # =====================================================

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Prediction",
            predicted
        )

    with col2:

        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )

    with col3:

        st.metric(
            "Model",
            "EfficientNetB0"
        )


    st.info(
        f"Confidence level: **{level}**"
    )


    # =====================================================
    # PROBABILITIES
    # =====================================================

    st.markdown(
        """
        <div class="card-title">
            📊 Class Probabilities
        </div>
        """,
        unsafe_allow_html=True
    )

    for name, probability in zip(
        CLASSES,
        probs
    ):

        percentage = (
            float(probability) * 100
        )

        st.write(
            f"**{name} — {percentage:.2f}%**"
        )

        st.progress(
            float(probability)
        )


    # =====================================================
    # GRAD-CAM
    # =====================================================

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card-title">
            🔥 Grad-CAM Visualization
        </div>

        <div class="card-subtitle">
            Visualization of image regions that influenced
            the model's prediction.
        </div>
        """,
        unsafe_allow_html=True
    )


    if cam is not None:

        st.image(
            cam,
            caption="Grad-CAM — model attention visualization",
            use_container_width=True
        )

        st.caption(
            "Grad-CAM highlights regions that contributed "
            "to the model's prediction. It does not indicate "
            "a medically confirmed tumor location."
        )

    else:

        st.warning(
            "Grad-CAM visualization could not be generated "
            "for this image."
        )


    # =====================================================
    # DOWNLOAD REPORT
    # =====================================================

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card-title">
            📦 Analysis Report
        </div>

        <div class="card-subtitle">
            Download the prediction, probabilities,
            processed MRI and Grad-CAM visualization
            as a ZIP file.
        </div>
        """,
        unsafe_allow_html=True
    )


    report_data = report_file(
        resized,
        cam,
        predicted,
        confidence,
        probs
    )


    st.download_button(
        label="⬇️ Download Prediction Report",
        data=report_data,
        file_name="brain_tumor_prediction_report.zip",
        mime="application/zip",
        use_container_width=True
    )


    # =====================================================
    # MEDICAL DISCLAIMER
    # =====================================================

    st.warning(
        "⚠️ Medical disclaimer: This model is a "
        "research/educational prototype. Model confidence "
        "is not medical certainty. The output must not be "
        "used as a substitute for a qualified radiologist "
        "or doctor."
    )


# =========================================================
# ABOUT SECTION
# =========================================================

st.divider()

st.markdown(
    """
    ## 🔬 About This Project

    **Brain Tumor Detection by Deep Learning** is a
    research/educational prototype designed to demonstrate
    how deep learning can be applied to brain MRI image
    classification.

    ### Model

    - Architecture: **EfficientNetB0**
    - Input size: **224 × 224 pixels**
    - Classes: **4**
    - Grad-CAM: **Enabled**
    - Deployment: **Streamlit + Hugging Face**

    ### Supported Classes

    1. Glioma
    2. Meningioma
    3. No Tumor
    4. Pituitary

    ### Important

    This application is intended for **research and
    educational purposes only**.

    It is **not a medical diagnostic device** and its
    predictions should not be used to make medical
    decisions.
    """
)


# =========================================================
# FOOTER
# =========================================================

indian_time = datetime.now(
    ZoneInfo("Asia/Kolkata")
)

st.markdown(
    f"""
    <div class="custom-footer">

        Brain Tumor Detection by Deep Learning
        <br>
        Research / Educational Prototype
        <br>
        Last session time:
        {indian_time.strftime("%Y-%m-%d %H:%M:%S IST")}

    </div>
    """,
    unsafe_allow_html=True
)
