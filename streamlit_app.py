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
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Brain Tumor Detection by Deep Learning",
    page_icon="🧠",
    layout="wide"
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
# SIMPLE PROFESSIONAL CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #07111f 0%,
        #0b1424 50%,
        #101827 100%
    );
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero-box {
    padding: 35px;
    border-radius: 24px;
    background: rgba(20, 35, 65, 0.90);
    border: 1px solid rgba(255,255,255,0.10);
    margin-bottom: 25px;
}

.hero-title {
    font-size: 40px;
    font-weight: 800;
}

.hero-subtitle {
    font-size: 17px;
    margin-top: 10px;
    opacity: 0.75;
}

.result-box {
    padding: 25px;
    border-radius: 20px;
    background: rgba(20, 35, 55, 0.85);
    border: 1px solid rgba(255,255,255,0.10);
    margin-top: 20px;
}

.section-title {
    font-size: 24px;
    font-weight: 700;
    margin-top: 25px;
    margin-bottom: 12px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL
# =========================================================

@st.cache_resource
def load_model():

    path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE
    )

    return tf.keras.models.load_model(
        path,
        compile=False
    )


# =========================================================
# GRAD-CAM
# =========================================================

@st.cache_resource
def load_gradcam_parts():

    model = load_model()

    augmentation = model.get_layer(
        "data_augmentation"
    )

    backbone = model.get_layer(
        "efficientnetb0"
    )

    gap = model.get_layer(
        "global_average_pooling2d_1"
    )

    dropout = model.get_layer(
        "dropout_1"
    )

    classifier = model.get_layer(
        "dense_2"
    )

    top_conv = backbone.get_layer(
        "top_conv"
    )

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
# LOAD EVERYTHING
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

    model_ready = True

except Exception as e:

    model_ready = False
    model_error = str(e)


if not model_ready:

    st.error(
        "The AI model could not be loaded."
    )

    st.code(model_error)

    st.stop()


# =========================================================
# PREDICTION
# =========================================================

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

    probs = model.predict(
        x,
        verbose=0
    )[0]

    index = int(
        np.argmax(probs)
    )

    return (
        resized,
        x,
        probs,
        index
    )


# =========================================================
# GRAD-CAM CALCULATION
# =========================================================

def make_gradcam(
    x,
    class_index
):

    try:

        x = tf.convert_to_tensor(
            x,
            dtype=tf.float32
        )

        with tf.GradientTape() as tape:

            augmented = augmentation(
                x,
                training=False
            )

            conv, features = grad_backbone(
                augmented,
                training=False
            )

            pooled = gap(
                features
            )

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

        gradients = tape.gradient(
            score,
            conv
        )

        if gradients is None:
            return None

        weights = tf.reduce_mean(
            gradients,
            axis=(1, 2)
        )[0]

        conv_output = conv[0]

        heatmap = tf.reduce_sum(
            conv_output * weights,
            axis=-1
        )

        heatmap = tf.maximum(
            heatmap,
            0
        )

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


# =========================================================
# GRAD-CAM OVERLAY
# =========================================================

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

    heat = np.asarray(
        heatmap_image
    ).astype(np.float32) / 255.0

    colored = (
        plt.get_cmap("jet")(
            heat
        )[..., :3] * 255
    )

    combined = (
        0.60 * original +
        0.40 * colored
    )

    combined = np.clip(
        combined,
        0,
        255
    )

    return Image.fromarray(
        np.uint8(combined)
    )


# =========================================================
# REPORT
# =========================================================

def create_report(
    resized,
    cam,
    predicted,
    confidence,
    probs
):

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

    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        "w",
        zipfile.ZIP_DEFLATED
    ) as archive:

        archive.writestr(
            "prediction_report.txt",
            report.getvalue()
        )

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
# HERO
# =========================================================

st.markdown(
    '<div class="hero-box">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-title">🧠 Brain Tumor Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-subtitle">'
    'Deep Learning Based MRI Classification System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<p>EfficientNetB0 • MRI Image Analysis • Grad-CAM</p>',
    unsafe_allow_html=True
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# INTRODUCTION
# =========================================================

st.info(
    "Upload a brain MRI image and click "
    "**Analyze MRI** to run the deep learning model. "
    "The system provides a predicted class, model "
    "confidence, class probabilities and Grad-CAM."
)


# =========================================================
# UPLOAD
# =========================================================

st.markdown(
    '<div class="section-title">📤 Upload Brain MRI</div>',
    unsafe_allow_html=True
)

st.write(
    "Select a JPG, JPEG or PNG MRI image."
)


# =========================================================
# FORM
# This prevents automatic analysis after upload.
# =========================================================

with st.form(
    "mri_form",
    clear_on_submit=False
):

    uploaded = st.file_uploader(
        "Choose MRI image",
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


# =========================================================
# ANALYZE
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
            "Unable to read the uploaded image."
        )

        st.stop()


    # -----------------------------------------------------
    # IMAGE
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">🖼️ Uploaded MRI</div>',
        unsafe_allow_html=True
    )

    st.image(
        image,
        use_container_width=True
    )


    # -----------------------------------------------------
    # ANALYSIS
    # -----------------------------------------------------

    with st.spinner(
        "🧠 AI is analyzing the MRI..."
    ):

        resized, x, probs, index = predict(
            image
        )

        predicted = CLASSES[index]

        confidence = (
            float(probs[index]) * 100
        )

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

        heatmap = make_gradcam(
            x,
            index
        )

        cam = create_overlay(
            resized,
            heatmap
        )


    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">🎯 Analysis Result</div>',
        unsafe_allow_html=True
    )


    if predicted == "No Tumor":

        st.success(
            "🟢 Model prediction: No Tumor"
        )

    else:

        st.warning(
            f"🟠 Model prediction: {predicted}"
        )


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
            "Architecture",
            "EfficientNetB0"
        )


    st.write(
        f"**Confidence level:** "
        f"{confidence_level}"
    )


    # -----------------------------------------------------
    # PROBABILITIES
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">📊 Class Probabilities</div>',
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
            f"**{name}: {percentage:.2f}%**"
        )

        st.progress(
            float(probability)
        )


    # -----------------------------------------------------
    # GRAD-CAM
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        '🔥 Grad-CAM Visualization'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "The visualization highlights regions that "
        "influenced the model prediction."
    )


    if cam is not None:

        st.image(
            cam,
            caption=(
                "Grad-CAM — model attention visualization"
            ),
            use_container_width=True
        )

        st.caption(
            "Grad-CAM is an AI visualization technique. "
            "It does not represent a medically confirmed "
            "tumor location."
        )

    else:

        st.warning(
            "Grad-CAM visualization could not be generated."
        )


    # -----------------------------------------------------
    # REPORT
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">📦 Analysis Report</div>',
        unsafe_allow_html=True
    )

    report_data = create_report(
        resized,
        cam,
        predicted,
        confidence,
        probs
    )

    st.download_button(
        "⬇️ Download Prediction Report",
        data=report_data,
        file_name="brain_tumor_prediction_report.zip",
        mime="application/zip",
        use_container_width=True
    )


    # -----------------------------------------------------
    # DISCLAIMER
    # -----------------------------------------------------

    st.warning(
        "⚠️ Medical disclaimer: This is a "
        "research/educational prototype. Model confidence "
        "is not medical certainty. This application must "
        "not replace evaluation by a qualified radiologist "
        "or doctor."
    )


# =========================================================
# ABOUT
# =========================================================

st.divider()

st.markdown(
    "## 🔬 About This Project"
)

st.write(
    "**Brain Tumor Detection by Deep Learning** is a "
    "research/educational prototype demonstrating the "
    "application of deep learning to brain MRI "
    "classification."
)

st.write("### Model")

st.write(
    """
    - Architecture: **EfficientNetB0**
    - Input size: **224 × 224 pixels**
    - Classes: **4**
    - Grad-CAM visualization: **Enabled**
    """
)

st.write("### Supported Classes")

st.write(
    """
    1. Glioma
    2. Meningioma
    3. No Tumor
    4. Pituitary
    """
)

st.warning(
    "This application is intended only for research "
    "and educational purposes. It is not a medical "
    "diagnostic device."
)


# =========================================================
# FOOTER
# =========================================================

indian_time = datetime.now(
    ZoneInfo("Asia/Kolkata")
)

st.markdown(
    "---"
)

st.caption(
    "🧠 Brain Tumor Detection by Deep Learning | "
    "Research / Educational Prototype | "
    f"{indian_time.strftime('%Y-%m-%d %H:%M:%S IST')}"
    )
