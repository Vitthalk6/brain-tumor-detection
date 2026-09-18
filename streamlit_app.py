import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
import io
from datetime import datetime
from zoneinfo import ZoneInfo


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Brain Tumor Detection AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background:
            radial-gradient(circle at 10% 10%, rgba(80,120,255,0.15), transparent 30%),
            radial-gradient(circle at 90% 20%, rgba(170,80,255,0.12), transparent 30%),
            linear-gradient(135deg, #070b17 0%, #0b1020 50%, #080d19 100%);
        color: white;
    }

    /* Hide Streamlit branding */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }

    /* Main title */
    .hero-title {
        font-size: 48px;
        font-weight: 800;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 5px;
        background: linear-gradient(
            90deg,
            #ffffff,
            #9db7ff,
            #d0a8ff
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        text-align: center;
        color: #aeb8d0;
        font-size: 18px;
        margin-bottom: 30px;
    }

    /* Cards */
    .card {
        background: rgba(18, 25, 45, 0.82);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.25);
    }

    .prediction-card {
        background: linear-gradient(
            145deg,
            rgba(27,39,72,0.95),
            rgba(20,25,50,0.95)
        );
        border: 1px solid rgba(120,150,255,0.25);
        border-radius: 24px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 15px 50px rgba(0,0,0,0.35);
    }

    .prediction-label {
        color: #aeb8d0;
        font-size: 15px;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    .prediction-value {
        font-size: 38px;
        font-weight: 800;
        margin-top: 8px;
        margin-bottom: 10px;
    }

    .confidence {
        font-size: 20px;
        color: #aabfff;
    }

    /* Section headings */
    .section-title {
        font-size: 25px;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    /* Information boxes */
    .info-box {
        background: rgba(255,255,255,0.045);
        border-radius: 15px;
        padding: 18px;
        border: 1px solid rgba(255,255,255,0.07);
        margin-bottom: 12px;
    }

    .info-title {
        font-weight: 700;
        font-size: 17px;
        margin-bottom: 5px;
    }

    .info-text {
        color: #aeb8d0;
        font-size: 14px;
        line-height: 1.6;
    }

    /* Disclaimer */
    .disclaimer {
        background: rgba(120,80,20,0.15);
        border: 1px solid rgba(255,190,80,0.25);
        border-radius: 16px;
        padding: 18px;
        color: #d7c9a4;
        font-size: 13px;
        line-height: 1.6;
        margin-top: 25px;
    }

    /* Footer */
    .custom-footer {
        text-align: center;
        color: #69748f;
        font-size: 13px;
        margin-top: 40px;
        padding: 20px;
    }

    /* Upload area */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.035);
        border-radius: 18px;
        padding: 15px;
        border: 1px dashed rgba(150,170,255,0.3);
    }

    /* Buttons */
    .stButton > button {
        border-radius: 12px;
        border: 1px solid rgba(150,170,255,0.25);
        background: rgba(70,90,170,0.25);
        color: white;
        font-weight: 600;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL SETTINGS
# ============================================================

MODEL_REPO = "Vitthalk/brain-tumor-efficientnet"
MODEL_FILE = "efficientnet_best.keras"

CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE
    )

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    return model


# ============================================================
# GRAD-CAM MODEL
#
# THIS PART IS FROM YOUR PREVIOUS WORKING CODE
# ============================================================

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


# ============================================================
# LOAD MODEL + GRAD-CAM PARTS
# ============================================================

try:

    with st.spinner("🔄 Loading AI model..."):

        (
            model,
            augmentation,
            grad_backbone,
            gap,
            dropout,
            classifier
        ) = load_gradcam_parts()

except Exception as e:

    st.error(
        "Unable to load the AI model."
    )

    st.code(
        str(e)
    )

    st.stop()


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(model, image):

    image = image.convert("RGB")

    resized = image.resize(
        (224, 224)
    )

    img_array = np.array(
        resized
    ).astype("float32")

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    predictions = model.predict(
        img_array,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = (
        float(
            predictions[
                predicted_index
            ]
        ) * 100
    )

    return (
        predicted_class,
        confidence,
        predictions,
        resized,
        img_array,
        predicted_index
    )


# ============================================================
# GRAD-CAM FUNCTION
#
# THIS IS THE SAME LOGIC FROM YOUR OLD WORKING CODE
# ============================================================

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

            score = classifier(
                dropout(
                    gap(features),
                    training=False
                )
            )[:, class_index]

        gradients = tape.gradient(
            score,
            conv
        )

        if gradients is None:
            return None

        pooled_gradients = tf.reduce_mean(
            gradients,
            axis=(1, 2)
        )[0]

        conv_outputs = conv[0]

        heatmap = tf.reduce_sum(
            conv_outputs *
            pooled_gradients,
            axis=-1
        )

        heatmap = tf.maximum(
            heatmap,
            0
        )

        max_value = tf.reduce_max(
            heatmap
        )

        if float(max_value) > 0:

            heatmap = (
                heatmap /
                max_value
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

    image = image.convert("RGB")

    image_array = np.array(
        image
    )

    heatmap_resized = Image.fromarray(
        np.uint8(
            heatmap * 255
        )
    ).resize(
        image.size
    )

    heatmap_array = np.array(
        heatmap_resized
    )

    cmap = plt.get_cmap(
        "jet"
    )

    colored_heatmap = cmap(
        heatmap_array
    )

    colored_heatmap = np.uint8(
        colored_heatmap[:, :, :3] * 255
    )

    overlay = (
        0.6 * image_array +
        0.4 * colored_heatmap
    )

    overlay = np.clip(
        overlay,
        0,
        255
    ).astype("uint8")

    return Image.fromarray(
        overlay
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="hero-title">'
    '🧠 Brain Tumor Detection AI'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-subtitle">'
    'Deep Learning based Brain MRI Classification • EfficientNetB0'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# BADGES
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.markdown(
        '<div class="info-box">'
        '<div class="info-title">🤖 AI Model</div>'
        '<div class="info-text">EfficientNetB0</div>'
        '</div>',
        unsafe_allow_html=True
    )

with col2:

    st.markdown(
        '<div class="info-box">'
        '<div class="info-title">🧠 Classes</div>'
        '<div class="info-text">4 MRI Categories</div>'
        '</div>',
        unsafe_allow_html=True
    )

with col3:

    st.markdown(
        '<div class="info-box">'
        '<div class="info-title">📐 Input</div>'
        '<div class="info-text">224 × 224 MRI</div>'
        '</div>',
        unsafe_allow_html=True
    )

with col4:

    st.markdown(
        '<div class="info-box">'
        '<div class="info-title">🔬 Technology</div>'
        '<div class="info-text">TensorFlow / Keras</div>'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🧠 Brain Tumor AI"
    )

    st.markdown("---")

    st.markdown(
        "### 📌 Model"
    )

    st.write(
        "EfficientNetB0"
    )

    st.markdown(
        "### 🎯 Detectable Classes"
    )

    st.write("• Glioma")
    st.write("• Meningioma")
    st.write("• No Tumor")
    st.write("• Pituitary")

    st.markdown("---")

    st.markdown(
        "### ℹ️ About"
    )

    st.write(
        "This application demonstrates how deep learning "
        "can be used for experimental classification of "
        "brain MRI images."
    )

    st.markdown("---")

    st.caption(
        "Research / Educational Prototype"
    )


# ============================================================
# UPLOAD SECTION
# ============================================================

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">'
    '📤 Upload Brain MRI'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload a brain MRI image in JPG, JPEG or PNG format."
)


# ============================================================
# MANUAL ANALYSIS FORM
#
# IMAGE WILL NOT BE ANALYZED UNTIL BUTTON IS PRESSED
# ============================================================

with st.form(
    "mri_analysis_form",
    clear_on_submit=False
):

    uploaded_file = st.file_uploader(
        "Choose an MRI image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        label_visibility="collapsed"
    )

    analyze_button = st.form_submit_button(
        "🔍 Analyze MRI",
        use_container_width=True
    )


st.markdown(
    "</div>",
    unsafe_allow_html=True
)


# ============================================================
# ============================================================
# ANALYSIS
# ============================================================

if analyze_button:

    # --------------------------------------------------------
    # OPEN IMAGE
    # --------------------------------------------------------

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

    except Exception:

        st.error(
            "Unable to read the uploaded image."
        )

        st.stop()


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    with st.spinner(
        "🧠 AI is analyzing the MRI..."
    ):

        (
            predicted_class,
            confidence,
            predictions,
            resized,
            img_array,
            predicted_index
        ) = predict_image(
            model,
            image
        )


    # ========================================================
    # IMAGE + PREDICTION
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🔍 MRI Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(2)


    # --------------------------------------------------------
    # ORIGINAL IMAGE
    # --------------------------------------------------------

    with left:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.markdown(
            "### 🖼️ Uploaded MRI"
        )

        st.image(
            image,
            use_container_width=True
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # PREDICTION CARD
    # --------------------------------------------------------

    with right:

        st.markdown(
            '<div class="prediction-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="prediction-label">'
            'AI Prediction'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="prediction-value">'
            f'{predicted_class}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="confidence">'
            f'Confidence: {confidence:.2f}%'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

        st.write("")

        if confidence >= 80:

            st.success(
                "High model confidence"
            )

        elif confidence >= 60:

            st.warning(
                "Moderate model confidence"
            )

        else:

            st.info(
                "Low model confidence"
            )


    # ========================================================
    # PROBABILITY DISTRIBUTION
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '📊 Prediction Probability'
        '</div>',
        unsafe_allow_html=True
    )

    probability_cols = st.columns(4)

    for i, class_name in enumerate(
        CLASS_NAMES
    ):

        probability = (
            float(
                predictions[i]
            ) * 100
        )

        with probability_cols[i]:

            st.metric(
                class_name,
                f"{probability:.2f}%"
            )

            st.progress(
                min(
                    max(
                        probability / 100,
                        0
                    ),
                    1
                )
            )


    # ========================================================
    # GRAD-CAM
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '🔥 AI Attention Map — Grad-CAM'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "The Grad-CAM visualization highlights regions "
        "that influenced the model prediction."
    )


    with st.spinner(
        "🔥 Generating Grad-CAM..."
    ):

        heatmap = make_gradcam(
            img_array,
            predicted_index
        )

        overlay = create_overlay(
            resized,
            heatmap
        )


    if overlay is not None:

        cam_col1, cam_col2 = st.columns(2)

        with cam_col1:

            st.markdown(
                '<div class="card">',
                unsafe_allow_html=True
            )

            st.markdown(
                "### Original MRI"
            )

            st.image(
                resized,
                use_container_width=True
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

        with cam_col2:

            st.markdown(
                '<div class="card">',
                unsafe_allow_html=True
            )

            st.markdown(
                "### 🔥 Grad-CAM"
            )

            st.image(
                overlay,
                use_container_width=True
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

        st.success(
            "Grad-CAM visualization generated successfully."
        )

        st.caption(
            "Grad-CAM highlights image regions that "
            "influenced the neural network prediction. "
            "It does not represent a medically confirmed "
            "tumor location."
        )

    else:

        st.warning(
            "Grad-CAM visualization could not be generated."
        )


    # ========================================================
    # REPORT
    # ========================================================

    st.markdown(
        '<div class="section-title">'
        '📄 Prediction Report'
        '</div>',
        unsafe_allow_html=True
    )


    # IST TIME

    indian_time = datetime.now(
        ZoneInfo("Asia/Kolkata")
    )


    report = f"""
BRAIN TUMOR DETECTION BY DEEP LEARNING
======================================

Analysis Time:
{indian_time.strftime("%Y-%m-%d %H:%M:%S IST")}

Model:
EfficientNetB0

Prediction:
{predicted_class}

Model Confidence:
{confidence:.2f}%

Class Probabilities:
--------------------
"""


    for name, probability in zip(
        CLASS_NAMES,
        predictions
    ):

        report += (
            f"{name}: "
            f"{float(probability) * 100:.2f}%\n"
        )


    report += """

Grad-CAM:
---------
Grad-CAM visualization was generated to highlight
image regions that influenced the model prediction.

IMPORTANT DISCLAIMER:
---------------------
This application is a research and educational prototype.
The prediction and confidence score are not a medical diagnosis.
This system must not be used for clinical decision-making.
Please consult a qualified radiologist or doctor for medical evaluation.
"""


    st.download_button(
        label="📥 Download Prediction Report",
        data=report,
        file_name="brain_tumor_prediction_report.txt",
        mime="text/plain",
        use_container_width=True
    )

# ============================================================
# ABOUT THIS PROJECT
# ============================================================

st.markdown(
    '<div class="section-title">🧠 About This Project</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-box">

        <div class="info-title">
            Brain Tumor Detection by Deep Learning
        </div>

        <div class="info-text">
            This application is an EfficientNetB0-based
            research and educational prototype for brain MRI
            image classification.

            <br><br>

            The model classifies MRI images into four categories:
            <b>Glioma</b>, <b>Meningioma</b>,
            <b>No Tumor</b>, and <b>Pituitary</b>.

            <br><br>

            The application also provides class probability
            information and Grad-CAM visualization to help
            visualize regions that influenced the model prediction.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MEDICAL DISCLAIMER
# ============================================================

st.markdown(
    """
    <div class="disclaimer">

        <b>⚠️ Medical Disclaimer</b>

        <br><br>

        This application is intended for research and educational
        purposes only. It is not a medical diagnostic device.

        <br><br>

        Model predictions and confidence scores should not be
        considered medical advice or a confirmed diagnosis.

        <br><br>

        Always consult a qualified radiologist or medical
        professional for clinical evaluation and diagnosis.

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="custom-footer">

        <b>Brain Tumor Detection by Deep Learning</b>

        <br><br>

        EfficientNetB0 • Research/Educational Prototype

        <br>

        AI-powered Brain MRI Classification

        <br><br>

        <span>Not a medical diagnostic device</span>

    </div>
    """,
    unsafe_allow_html=True
)
