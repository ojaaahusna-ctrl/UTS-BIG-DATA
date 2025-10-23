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

button[title="View fullscreen"] {
    visibility: hidden;
}

/* ================== PERBAIKAN DI SINI (1/3) ================== */
/* CSS untuk kotak hasil deteksi, kini teks sejajar vertikal & horizontal */
.detection-result-box {
    background-color: #319795;
    color: white !important;
    border-radius: 8px;
    border: none;
    padding: 8px 16px;
    font-weight: 700;
    font-family: 'Inter', sans-serif;
    display: flex;
    justify-content: center;   /* sejajar horizontal */
    align-items: center;       /* sejajar vertikal */
    text-align: center;
    margin-top: 1rem;
    min-height: 45px;          /* tinggi minimum biar teks tidak menempel */
}
/* ================== AKHIR PERBAIKAN CSS ================== */
</style>
""", unsafe_allow_html=True)
