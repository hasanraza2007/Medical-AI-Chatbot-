import streamlit as st
import cv2
import pytesseract
from PIL import Image
import numpy as np

st.title("Medical Lab Report OCR and Analysis")

uploaded_file = st.file_uploader("Upload a lab report image or PDF", type=["png", "jpg", "jpeg", "pdf"])

if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        st.warning("PDF support not implemented in this example.")
    else:
        # Load image with PIL
        image = Image.open(uploaded_file).convert("RGB")
        
        # Convert PIL Image to OpenCV image format
        img_cv = np.array(image)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert to grayscale
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

        # Optional: thresholding to improve OCR accuracy
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        # OCR with pytesseract
        text = pytesseract.image_to_string(thresh)

        st.subheader("Extracted Text")
        st.text_area("OCR Output", value=text, height=300)
