import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt
import io, zipfile
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Brain Tumor Detection by Deep Learning", page_icon="🧠", layout="wide")

CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
IMG_SIZE = (224, 224)

@st.cache_resource
def load_model():
    path = hf_hub_download(
        repo_id="Vitthalk/brain-tumor-efficientnet",
        filename="efficientnet_best.keras"
    )
    return tf.keras.models.load_model(path)

@st.cache_resource
def load_gradcam_parts():
    model = load_model()
    aug = model.get_layer("data_augmentation")
    backbone = model.get_layer("efficientnetb0")
    gap = model.get_layer("global_average_pooling2d_1")
    dropout = model.get_layer("dropout_1")
    classifier = model.get_layer("dense_2")
    top_conv = backbone.get_layer("top_conv")
    grad_backbone = tf.keras.Model(
        inputs=backbone.input,
        outputs=[top_conv.output, backbone.output]
    )
    return model, aug, grad_backbone, gap, dropout, classifier

model, augmentation, grad_backbone, gap, dropout, classifier = load_gradcam_parts()

def predict(image):
    image = image.convert("RGB")
    resized = image.resize(IMG_SIZE)
    x = np.expand_dims(np.asarray(resized, dtype=np.float32), 0)
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return resized, x, probs, idx

def gradcam(x, class_index):
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    with tf.GradientTape() as tape:
        augmented = augmentation(x, training=False)
        conv, features = grad_backbone(augmented, training=False)
        score = classifier(dropout(gap(features), training=False))[:, class_index]
    grads = tape.gradient(score, conv)
    if grads is None: return None
    weights = tf.reduce_mean(grads, axis=(1, 2))[0]
    heat = tf.reduce_sum(conv[0] * weights, axis=-1)
    heat = tf.maximum(heat, 0)
    m = tf.reduce_max(heat)
    if float(m) > 0: heat = heat / m
    return heat.numpy()

def overlay(image, heatmap):
    if heatmap is None: return None
    original = np.asarray(image.convert("RGB")).astype(np.float32)
    h, w = original.shape[:2]
    heat = np.asarray(Image.fromarray(np.uint8(heatmap * 255)).resize((w, h))).astype(np.float32) / 255
    colored = plt.get_cmap("jet")(heat)[..., :3] * 255
    return Image.fromarray(np.uint8(np.clip(0.60 * original + 0.40 * colored, 0, 255)))

def report_file(resized, cam, predicted, confidence, probs):
    report = io.StringIO()
    report.write("BRAIN TUMOR DETECTION BY DEEP LEARNING\n")
    report.write("=" * 50 + "\n")
    report.write(f"Date/Time: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    report.write(f"Prediction: {predicted}\n")
    report.write(f"Model Confidence: {confidence:.2f}%\n\n")
    report.write("Class Probabilities:\n")
    for name, p in zip(CLASSES, probs):
        report.write(f"{name}: {float(p)*100:.2f}%\n")
    report.write("\nDISCLAIMER: Research/educational prototype; not a medical diagnosis.\n")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("prediction_report.txt", report.getvalue())
        b = io.BytesIO(); resized.save(b, "JPEG", quality=95); z.writestr("mri_224x224.jpg", b.getvalue())
        if cam is not None:
            b = io.BytesIO(); cam.save(b, "JPEG", quality=95); z.writestr("gradcam.jpg", b.getvalue())
    return buf.getvalue()

st.title("🧠 Brain Tumor Detection by Deep Learning")
st.caption("EfficientNetB0-based research/educational prototype")
st.info("Upload a brain MRI image. The model classifies it as Glioma, Meningioma, No Tumor, or Pituitary.")

uploaded = st.file_uploader("Upload Brain MRI", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.subheader("Uploaded MRI")
    st.image(image, use_container_width=True)

    if st.button("🔍 Analyze MRI", type="primary", use_container_width=True):
        with st.spinner("AI is analyzing the MRI..."):
            resized, x, probs, idx = predict(image)
            predicted = CLASSES[idx]
            confidence = float(probs[idx]) * 100
            level = "🟢 High model confidence" if confidence >= 80 else ("🟡 Moderate model confidence" if confidence >= 60 else "🟠 Low model confidence — interpret cautiously")
            cam = overlay(resized, gradcam(x, idx))

        if predicted == "No Tumor": st.success("🟢 NO TUMOR DETECTED")
        else: st.error("🔴 TUMOR DETECTED")

        c1, c2 = st.columns(2)
        c1.metric("Prediction", predicted)
        c2.metric("Model Confidence", f"{confidence:.2f}%")
        st.write("**Confidence level:**", level)

        st.subheader("Class Probabilities")
        for name, p in zip(CLASSES, probs):
            st.write(f"**{name}: {float(p)*100:.2f}%**")
            st.progress(float(p))

        if cam is not None:
            st.subheader("🔥 Grad-CAM Visualization")
            st.image(cam, caption="Regions influencing the model prediction", use_container_width=True)

        st.download_button(
            "📥 Download Prediction Report",
            data=report_file(resized, cam, predicted, confidence, probs),
            file_name="brain_tumor_prediction_report.zip",
            mime="application/zip",
            use_container_width=True
        )
        st.warning("Medical disclaimer: Model confidence is not medical certainty. This research/educational prototype must not replace a qualified radiologist or doctor.")

st.divider()
st.caption("Brain Tumor Detection by Deep Learning | Research/Educational Prototype | Not a medical diagnostic device")
