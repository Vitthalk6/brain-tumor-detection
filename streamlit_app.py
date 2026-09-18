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


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Brain Tumor Detection by Deep Learning",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CONSTANTS
# ============================================================

CLASSES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

IMG_SIZE = (224, 224)

MODEL_REPO = "Vitthalk/brain-tumor-efficientnet"
MODEL_FILE = "efficientnet_best.keras"


# ============================================================
# CREATIVE UI CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main application */
    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(80, 120, 255, 0.12),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 15%,
                rgba(170, 80, 255, 0.10),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #07111f 0%,
                #0b1424 50%,
                #101827 100%
            );
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Hero */
    .hero {
        padding: 38px 35px;
        border-radius: 26px;
        margin-bottom: 25px;
        background:
            linear-gradient(
                135deg,
                rgba(24, 42, 76, 0.96),
                rgba(35, 23, 67, 0.96)
            );
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow:
            0 20px 50px rgba(0,0,0,0.30);
    }

    .hero-title {
        font-size: 44px;
        font-weight: 800;
        letter-spacing: -1.5px;
        line-height: 1.15;
    }

    .hero-subtitle {
        margin-top: 10px;
        font-size: 18px;
        color: #aebbd0;
    }

    .hero-badge {
        display: inline-block;
        margin-top: 18px;
        padding: 8px 16px;
        border-radius: 30px;
        background: rgba(100,130,255,0.15);
        border: 1px solid rgba(130,155,255,0.30);
        color: #bdcaff;
        font-size: 13px;
        font-weight: 600;
    }

    /* Cards */
    .creative-card {
        background: rgba(17, 29, 48, 0.80);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 22px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 12px 35px rgba(0,0,0,0.18);
    }

    .card-heading {
        font-size: 23px;
        font-weight: 750;
        margin-bottom: 6px;
    }

    .card-description {
        color: #98a8bd;
        font-size: 14px;
        margin-bottom: 15px;
    }

    /* Section title */
    .section-title {
        font-size: 25px;
        font-weight: 750;
        margin-top: 28px;
        margin-bottom: 12px;
    }

    /* Status badge */
    .status {
        padding: 14px 18px;
        border-radius: 15px;
        background: rgba(40, 70, 110, 0.20);
        border: 1px solid rgba(100,150,220,0.20);
        margin: 15px 0;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #718096;
        font-size: 13px;
        padding: 30px 10px 10px 10px;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(10, 20, 35, 0.60);
        border-radius: 18px;
        padding: 8px;
    }

    /* Buttons */
    .stButton > button,
    .stDownloadButton > button {
        min-height: 48px;
        border-radius: 13px;
        font-weight: 700;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background: rgba(18, 30, 50, 0.78);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 18px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE
    )

    return tf.keras.models.load_model(
        model_path,
        compile=False
    )


# ============================================================
# LOAD GRAD-CAM PARTS
#
# THIS IS THE ORIGINAL WORKING STRUCTURE
# FROM YOUR PREVIOUS STREAMLIT CODE.
# ============================================================

@st.cache_resource
def load_gradcam_parts():

    model = load_model()

    # Original augmentation layer
    augmentation = model.get_layer(
        "data_augmentation"
    )

    # Original EfficientNet backbone
    backbone = model.get_layer(
        "efficientnetb0"
    )

    # Original classification layers
    gap = model.get_layer(
        "global_average_pooling2d_1"
    )

    dropout = model.get_layer(
        "dropout_1"
    )

    classifier = model.get_layer(
        "dense_2"
    )

    # Original Grad-CAM target layer
    top_conv = backbone.get_layer(
        "top_conv"
    )

    # IMPORTANT:
    # This is the same Grad-CAM model structure
    # from your old working application.
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


# ============================================================
# INITIALIZE MODEL
# ============================================================

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

except Exception as error:

    MODEL_READY = False
    MODEL_ERROR = str(error)


if not MODEL_READY:

    st.error(
        "⚠️ The AI model could not be loaded."
    )

    st.code(
        MODEL_ERROR
    )

    st.stop()


# ============================================================
# PREDICTION
# ============================================================

def predict(image):

    image = image.convert("RGB")

    resized = image.resize(
        IMG_SIZE
    )

    x = np.expand_dims(
        np.asarray(
            resized,
            dtype=np.float32
        ),
        axis=0
    )

    probabilities = model.predict(
        x,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    return (
        resized,
        x,
        probabilities,
        predicted_index
    )


# ============================================================
# ORIGINAL WORKING GRAD-CAM
# ============================================================

def gradcam(
    x,
    class_index
):

    try:

        x = tf.convert_to_tensor(
            x,
            dtype=tf.float32
        )

        with tf.GradientTape() as tape:

            # Original augmentation
            augmented = augmentation(
                x,
                training=False
            )

            # Original Grad-CAM backbone
            conv, features = grad_backbone(
                augmented,
                training=False
            )

            # Original classifier path
            pooled = gap(
                features
            )

            dropped = dropout(
                pooled,
                training=False
            )

            score = classifier(
                dropped,
                training=False
            )[:, class_index]

        # Gradient
        gradients = tape.gradient(
            score,
            conv
        )

        if gradients is None:
            return None

        # Channel weights
        weights = tf.reduce_mean(
            gradients,
            axis=(1, 2)
        )[0]

        # Feature maps
        conv_output = conv[0]

        # Weighted feature maps
        heatmap = tf.reduce_sum(
            conv_output * weights,
            axis=-1
        )

        # ReLU
        heatmap = tf.maximum(
            heatmap,
            0
        )

        # Normalize
        maximum = tf.reduce_max(
            heatmap
        )

        if float(maximum) > 0:

            heatmap = (
                heatmap / maximum
            )

        return heatmap.numpy()

    except Exception:

        return None


# ============================================================
# CREATE GRAD-CAM OVERLAY
# ============================================================

def create_overlay(
    image,
    heatmap
):

    if heatmap is None:
        return None

    original = np.asarray(
        image.convert("RGB")
    ).astype(np.float32)

    height, width = original.shape[:2]

    heatmap_image = Image.fromarray(
        np.uint8(
            heatmap * 255
        )
    ).resize(
        (width, height)
    )

    heat = (
        np.asarray(
            heatmap_image
        ).astype(np.float32)
        / 255.0
    )

    colored_heatmap = (
        plt.get_cmap("jet")(
            heat
        )[..., :3] * 255
    )

    result = (
        0.60 * original
        +
        0.40 * colored_heatmap
    )

    result = np.clip(
        result,
        0,
        255
    )

    return Image.fromarray(
        np.uint8(result)
    )


# ============================================================
# CREATE REPORT
# ============================================================

def create_report(
    resized,
    cam,
    predicted,
    confidence,
    probabilities
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
        "Date/Time: "
        + indian_time.strftime(
            "%Y-%m-%d %H:%M:%S IST"
        )
        + "\n"
    )

    report.write(
        f"Prediction: {predicted}\n"
    )

    report.write(
        f"Model Confidence: {confidence:.2f}%\n\n"
    )

    report.write(
        "Class Probabilities:\n"
    )

    for name, probability in zip(
        CLASSES,
        probabilities
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

        # Resized image
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

        # Grad-CAM image
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


# ============================================================
# HERO HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-title">
            🧠 Brain Tumor Detection
        </div>

        <div class="hero-subtitle">
            Deep Learning Based MRI Classification System
        </div>

        <div class="hero-badge">
            EfficientNetB0 &nbsp;•&nbsp;
            MRI Analysis &nbsp;•&nbsp;
            Grad-CAM
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# INTRODUCTION
# ============================================================

st.markdown(
    """
    <div class="creative-card">

        <div class="card-heading">
            🔬 AI-Powered MRI Analysis
        </div>

        <div class="card-description">
            Upload a brain MRI image and manually start the
            analysis. The system provides a model prediction,
            confidence score, class probabilities and Grad-CAM
            visualization.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# UPLOAD SECTION
# ============================================================

st.markdown(
    """
    <div class="section-title">
        📤 Upload Brain MRI
    </div>
    """,
    unsafe_allow_html=True
)

st.caption(
    "Supported formats: JPG, JPEG and PNG"
)


# ============================================================
# FORM
#
# IMPORTANT:
# Uploading an image DOES NOT automatically analyze it.
# Analysis happens only after pressing the button.
# ============================================================

with st.form(
    "mri_analysis_form",
    clear_on_submit=False
):

    uploaded = st.file_uploader(
        "Select your MRI image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    analyze = st.form_submit_button(
        "🔍 Analyze MRI",
        type="primary",
        use_container_width=True
    )


# ============================================================
# ANALYSIS
# ============================================================

if analyze:

    if uploaded is None:

        st.warning(
            "Please upload an MRI image before starting analysis."
        )

        st.stop()


    # --------------------------------------------------------
    # OPEN IMAGE
    # --------------------------------------------------------

    try:

        image = Image.open(
            uploaded
        ).convert("RGB")

    except Exception:

        st.error(
            "Unable to read this image. "
            "Please upload a valid JPG, JPEG or PNG image."
        )

        st.stop()


    # --------------------------------------------------------
    # IMAGE PREVIEW
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="section-title">
            🖼️ MRI Image
        </div>
        """,
        unsafe_allow_html=True
    )

    st.image(
        image,
        caption="Uploaded Brain MRI",
        use_container_width=True
    )


    # --------------------------------------------------------
    # AI ANALYSIS
    # --------------------------------------------------------

    with st.spinner(
        "🧠 AI is analyzing the MRI..."
    ):

        resized, x, probabilities, predicted_index = predict(
            image
        )

        predicted = CLASSES[
            predicted_index
        ]

        confidence = (
            float(
                probabilities[
                    predicted_index
                ]
            ) * 100
        )


        # Confidence level
        if confidence >= 80:

            confidence_level = (
                "High model confidence"
            )

        elif confidence >= 60:

            confidence_level = (
                "Moderate model confidence"
            )

        else:

            confidence_level = (
                "Low model confidence — "
                "interpret cautiously"
            )


        # ----------------------------------------------------
        # GRAD-CAM
        # ----------------------------------------------------

        heatmap = gradcam(
            x,
            predicted_index
        )

        cam = create_overlay(
            resized,
            heatmap
        )


    # ========================================================
    # RESULT
    # ========================================================

    st.markdown(
        """
        <div class="section-title">
            🎯 Analysis Result
        </div>
        """,
        unsafe_allow_html=True
    )


    if predicted == "No Tumor":

        st.success(
            "🟢 Model Prediction: No Tumor"
        )

    else:

        st.warning(
            f"🟠 Model Prediction: {predicted}"
        )


    # --------------------------------------------------------
    # RESULT METRICS
    # --------------------------------------------------------

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
            "AI Model",
            "EfficientNetB0"
        )


    st.info(
        f"**Confidence level:** {confidence_level}"
    )


    # ========================================================
    # CLASS PROBABILITIES
    # ========================================================

    st.markdown(
        """
        <div class="section-title">
            📊 Class Probabilities
        </div>
        """,
        unsafe_allow_html=True
    )


    for name, probability in zip(
        CLASSES,
        probabilities
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


    # ========================================================
    # GRAD-CAM
    # ========================================================

    st.markdown(
        """
        <div class="section-title">
            🔥 Grad-CAM Visualization
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        "Grad-CAM highlights image regions that influenced "
        "the model's prediction."
    )


    if cam is not None:

        st.image(
            cam,
            caption=(
                "Grad-CAM — Model Attention Visualization"
            ),
            use_container_width=True
        )

        st.success(
            "Grad-CAM visualization generated successfully."
        )

        st.caption(
            "The highlighted regions represent areas that "
            "influenced the neural network prediction. "
            "They are not a medically confirmed tumor location."
        )

    else:

        st.warning(
            "Grad-CAM visualization could not be generated "
            "for this image."
        )


    # ========================================================
    # DOWNLOAD REPORT
    # ========================================================

    st.markdown(
        """
        <div class="section-title">
            📦 Download Analysis
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        "The ZIP report contains the prediction report, "
        "processed MRI image and Grad-CAM image."
    )


    report_data = create_report(
        resized,
        cam,
        predicted,
        confidence,
        probabilities
    )


    st.download_button(
        label="⬇️ Download Prediction Report",
        data=report_data,
        file_name="brain_tumor_prediction_report.zip",
        mime="application/zip",
        use_container_width=True
    )


    # ==================================================
