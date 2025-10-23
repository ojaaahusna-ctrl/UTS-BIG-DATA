import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import io
import base64
import requests
import re

# ================== KONFIGURASI HALAMAN ==================
st.set_page_config(
    page_title="VisionCraft — From Pixels to Insights | Final Version",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="auto"
)

# ================== INITIALIZE SESSION STATE ==================
defaults = {
    'page': 'home',
    'selected_image_bytes': None,
    'cnn_conf': 0.5
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ================== DESAIN TEMA & STYLE (CSS) ==================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&family=Playfair+Display:wght@700&display=swap');

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #E6FFFA 0%, #B2F5EA 100%);
    color: #2D3748; 
}

.stApp, .main, [data-testid="stSidebar"] {
    background: linear-gradient(135deg, #E6FFFA 0%, #B2F5EA 100%); 
    color: #2D3748;
}

h1, h2, h3, h4, h5, h6, p, li, label, .stMarkdown, .stText, 
[data-testid="stMarkdownContainer"],
.stRadio > label,
[data-testid="stMetricLabel"], 
[data-testid="stMetricValue"], 
[data-testid="stAlert"]
{
    color: #2D3748 !important; 
}

#home-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    width: 100%;
    margin-top: 1rem;
}

#home-container > div { max-width: 900px; width: 100%; }

[data-testid="stSidebar"] { background-color: #F0FFF4; }

.header {
    background-color: rgba(255, 255, 255, 0.6);
    backdrop-filter: blur(8px);
    padding: 2rem;
    border-radius: 16px;
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.8);
    margin-bottom: 1.5rem; 
    animation: fadeIn 1.2s ease-in-out;
}

.header h1 {
    font-family: 'Playfair Display', serif;
    color: #1F2937; 
    font-size: 2.6rem;
    animation: floatText 4s ease-in-out infinite;
    margin: 0;
}

.header p {
    margin-top: 0.5rem;
    color: #334155;
}

@keyframes fadeIn {
    0% { opacity: 0; transform: translateY(-8px); }
    100% { opacity: 1; transform: translateY(0); }
}

@keyframes floatText {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-4px); }
}

.menu-card {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    padding: 1.6rem 1.2rem;
    border-radius: 12px;
    text-align: center;
    transition: all 0.18s ease-in-out;
    height: 100%;
}

.menu-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 20px rgba(49, 151, 149, 0.08);
}

.stButton>button {
    background-color: #319795;
    color: white !important;
    border-radius: 8px;
    border: none;
    padding: 8px 16px;
    font-weight: 700;
}
.stButton>button:hover { background-color: #2C7A7B; }

div[data-baseweb="input"], div[data-baseweb="textarea"] {
    background-color: #FFFFFF !important;
    border-radius: 8px;
    border: 1px solid #B2F5EA;
    color: #2D3748 !important;
}

[data-testid="stFileUploader"] section {
    background-color: #FFFFFF !important;
    border: 2px dashed #B2F5EA !important;
}

[data-testid="stFileUploader"] section * {
    color: #2D3748 !important; 
}

.stFileUploader > div > label {
    color: #2D3748 !important; 
}

/* Hilangkan tombol fullscreen kecil */
button[title="View fullscreen"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# ================== LOAD MODEL ==================
@st.cache_resource(show_spinner="📦 Memuat model YOLO...")
def load_yolo_model():
    try:
        return YOLO("model/Raudhatul Husna_laporan4.pt")
    except Exception as e:
        st.error(f"❌ Gagal memuat model YOLO: {e}", icon="🔥")
        return None

@st.cache_resource(show_spinner="📦 Memuat model CNN...")
def load_cnn_model():
    try:
        return tf.keras.models.load_model("model/Raudhatul Husna_laporan2.h5", compile=False)
    except Exception as e:
        st.error(f"❌ Gagal memuat model CNN: {e}", icon="🔥")
        return None

# ================== UTILITAS ==================
def clear_image_state():
    st.session_state['selected_image_bytes'] = None

def reset_and_rerun():
    clear_image_state()
    st.session_state['page'] = 'home'
    try:
        st.toast("✅ Gambar dihapus dan halaman direset.", icon="🗑️")
    except Exception:
        pass

# ================== HALAMAN HOME ==================
def home_page():
    st.markdown('<div id="home-container">', unsafe_allow_html=True)
    st.markdown("""
    <div class="header">
        <h1>✨ VisionCraft — From Pixels to Insights ✨</h1>
        <p>Empowering AI to see the world, one pixel at a time.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Pilih Tugas yang Ingin Dilakukan:")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="menu-card"><h3>🌭 Deteksi Objek</h3><p>Menggunakan model YOLO untuk mendeteksi Hotdog vs Not-Hotdog.</p></div>', unsafe_allow_html=True)
        if st.button("Mulai Deteksi", use_container_width=True, key="yolo_nav"):
            st.session_state.page = 'yolo'
            clear_image_state()
    with col2:
        st.markdown('<div class="menu-card"><h3>🐆 Klasifikasi Gambar</h3><p>Menggunakan model CNN untuk mengklasifikasikan Cheetah dan Hyena.</p></div>', unsafe_allow_html=True)
        if st.button("Mulai Klasifikasi", use_container_width=True, key="cnn_nav"):
            st.session_state.page = 'cnn'
            clear_image_state()

    st.markdown("---")
    st.info("Proyek ini dibuat oleh **Raudhatul Husna** sebagai bagian dari Ujian Tengah Semester.", icon="🎓")
    st.markdown('</div>', unsafe_allow_html=True)

# ================== HALAMAN MODEL ==================
def run_model_page(page_type):
    if page_type == 'yolo':
        title = "🌭 Deteksi Objek: Hotdog vs Not-Hotdog"
        model_loader = load_yolo_model
        button_text = "🔍 Mulai Deteksi"
    else:
        title = "🐆 Klasifikasi Gambar: Cheetah vs Hyena"
        model_loader = load_cnn_model
        button_text = "🔮 Lakukan Prediksi"

    if st.button("⬅️ Kembali ke Menu Utama"):
        st.session_state.page = 'home'
        clear_image_state()
        st.rerun()

    st.header(title)

    if page_type == 'cnn':
        st.info("Model ini hanya mengenali **Cheetah** dan **Hyena**.", icon="💡")
    if page_type == 'yolo':
        st.info("⚠️ Model ini hanya dilatih untuk mendeteksi **Hotdog**.", icon="🌭")

    model = model_loader()
    if not model:
        return

    image_bytes = st.session_state.get('selected_image_bytes')

    if image_bytes:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except:
            st.error("❌ Format gambar tidak didukung.")
            return

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🖼️ Gambar Asli")
            st.image(image, use_container_width=True)
            if st.button("🗑️ Hapus Gambar & Reset", use_container_width=True):
                reset_and_rerun()

        placeholder = col2.empty()
        placeholder.info("Tekan tombol di bawah untuk memproses gambar.")

        if st.button(button_text, use_container_width=True):
            with st.spinner("🧠 Menganalisis gambar..."):
                if page_type == 'yolo':
                    results = model(image, conf=0.5)
                    plot_result = results[0].plot(show=False)

                    if plot_result is not None:
                        orig_w, orig_h = image.size
                        plot_result = cv2.resize(plot_result, (orig_w, orig_h))
                        result_img_rgb = cv2.cvtColor(plot_result, cv2.COLOR_BGR2RGB)

                        with placeholder.container():
                            st.subheader("🎯 Hasil Deteksi")
                            st.image(result_img_rgb, use_container_width=True)

                            boxes = results[0].boxes
                            if len(boxes) > 0:
                                for i, box in enumerate(boxes):
                                    cls_name = model.names.get(int(box.cls), str(int(box.cls)))
                                    st.success(f"Objek {i+1}: `{cls_name}` | Keyakinan: `{float(box.conf[0]):.2%}`")
                            else:
                                st.success("✅ Tidak ditemukan objek 'Hotdog' → **Not-Hotdog**", icon="👍")
                    else:
                        st.warning("Tidak ada hasil deteksi.")
                else:
                    CLASS_NAMES_CNN = {0: "Cheetah 🐆", 1: "Hyena 🐕"}
                    input_shape = model.input_shape[1:3]
                    img_array = np.expand_dims(np.array(image.resize(input_shape)) / 255.0, axis=0)
                    preds_output = model.predict(img_array, verbose=0)[0]
                    pred_idx = int(np.argmax(preds_output))
                    pred_prob = float(np.max(preds_output))

                    with placeholder.container():
                        st.subheader("🎯 Hasil Prediksi")
                        st.metric("Prediksi:", CLASS_NAMES_CNN[pred_idx])
                        st.metric("Keyakinan:", f"{pred_prob:.2%}")

# ================== ROUTER ==================
if st.session_state.page == 'home':
    home_page()
elif st.session_state.page == 'yolo':
    run_model_page('yolo')
elif st.session_state.page == 'cnn':
    run_model_page('cnn')

# ================== FOOTER ==================
st.markdown("""
<hr>
<div style='text-align:center; font-size:0.9em; color:gray; margin-top:16px;'>
    © 2025 VisionCraft — Made with ❤️ by Raudhatul Husna
</div>
""", unsafe_allow_html=True)
