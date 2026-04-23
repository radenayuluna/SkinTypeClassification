import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Deteksi Kulit Wajah", layout="centered")

# =========================
# CUSTOM CSS (MODERN UI)
# =========================
st.markdown("""
<style>
body {
    background-color: #f5f7fb;
}

.main-title {
    text-align: center;
    font-size: 36px;
    font-weight: bold;
    color: #2c3e50;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #7f8c8d;
    margin-bottom: 30px;
}

.card {
    background: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.upload-box {
    border: 2px dashed #dcdde1;
    border-radius: 15px;
    padding: 50px;
    text-align: center;
    color: #7f8c8d;
    transition: 0.3s;
}

.upload-box:hover {
    border-color: #6c5ce7;
    background-color: #f8f9ff;
}

.result-card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    text-align: center;
}

.result-text {
    font-size: 22px;
    font-weight: bold;
    color: #2c3e50;
}

.confidence {
    color: #6c5ce7;
    font-size: 18px;
}

button[kind="primary"] {
    background-color: #6c5ce7;
    border-radius: 10px;
    height: 45px;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("k17.keras")

model = load_model()

# =========================
# PREPROCESSING
# =========================
def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# =========================
# PREDIKSI
# =========================
class_names = ["Normal", "Oily", "Dry"]

def predict(image):
    processed = preprocess_image(image)
    pred = model.predict(processed)
    idx = np.argmax(pred)
    return class_names[idx], np.max(pred)

# =========================
# HEADER
# =========================
st.markdown('<div class="main-title">Deteksi Kulit Wajah</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload foto wajah untuk mengetahui tipe kulit Anda</div>', unsafe_allow_html=True)

# =========================
# UPLOAD CARD
# =========================
st.markdown('<div class="card">', unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

st.markdown('<div class="upload-box">📤 Upload Foto Kulit Wajah</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# =========================
# BUTTON
# =========================
if st.button("🔍 Deteksi Sekarang"):

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        with st.spinner("Menganalisis gambar..."):
            label, confidence = predict(image)

        # =========================
        # HASIL
        # =========================
        st.markdown('<div class="main-title" style="font-size:28px;">Hasil Deteksi</div>', unsafe_allow_html=True)

        st.image(image, use_container_width=True)

        # Deskripsi hasil
        if label == "Oily":
            desc = "Kulit berminyak cenderung mengkilap dan rentan terhadap jerawat."
        elif label == "Dry":
            desc = "Kulit kering biasanya terasa kasar dan kurang kelembapan."
        else:
            desc = "Kulit normal memiliki keseimbangan antara minyak dan kelembapan."

        st.markdown(f"""
        <div class="result-card">
            <div class="result-text">{label}</div>
            <div class="confidence">Confidence: {confidence:.2f}</div>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.warning("Silakan upload gambar terlebih dahulu!")
