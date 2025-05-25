import streamlit as st
import pandas as pd
import numpy as np
import cv2
import easyocr
from pdf2image import convert_from_bytes
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from PIL import Image
import openai
import os

# ✅ THIS MUST BE THE FIRST Streamlit COMMAND
st.set_page_config(page_title="🧪 Medical Report Interpreter", layout="centered")

# ----- API Key Management -----
openai.api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

# ----- EasyOCR Initialization -----
@st.cache_resource
def get_easyocr_reader():
    return easyocr.Reader(['en'])

reader = get_easyocr_reader()

# ----- Preprocess Uploaded Image -----
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
    _, binarized = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binarized

# ----- Extract Text Using OCR -----
def extract_text_from_image(img):
    results = reader.readtext(img)
    text = "\n".join([res[1] for res in results])
    return text

# ----- Parse Text into Structured Table -----
def parse_medical_data(text):
    rows = []
    for line in text.split('\n'):
        tokens = line.split()
        if len(tokens) >= 4:
            name = " ".join(tokens[:-3])
            try:
                value = float(tokens[-3])
                unit = tokens[-2]
                range_parts = tokens[-1].replace("–", "-").split('-')
