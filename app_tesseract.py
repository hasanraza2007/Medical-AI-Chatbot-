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

# ✅ Must be the first Streamlit command
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

# ✅ FIXED: Parse Text into Structured Table with proper exception handling
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
                if len(range_parts) == 2:
                    normal_min = float(range_parts[0])
                    normal_max = float(range_parts[1])
                    status = (
                        "Critical" if value < normal_min * 0.8 or value > normal_max * 1.2 else
                        "Borderline" if value < normal_min or value > normal_max else
                        "Normal"
                    )
                    rows.append([name, value, unit, normal_min, normal_max, status])
            except Exception:
                continue
    return pd.DataFrame(rows, columns=["Test", "Value", "Unit", "Normal Min", "Normal Max", "Status"])

# ----- GPT-Based Explanation -----
def explain_with_openai(row):
    prompt = (
        f"Explain in simple language what it means if the patient’s {row['Test']} is "
        f"{row['Value']} {row['Unit']}, given the normal range is {row['Normal Min']}-{row['Normal Max']} {row['Unit']}."
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"⚠️ Failed to get explanation: {str(e)}"

# ----- Convert Uploaded File -----
def process_uploaded_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        images = convert_from_bytes(uploaded_file.read())
        image = np.array(images[0])
    else:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
    return image

# ----- Generate PDF Report -----
def generate_pdf_report(df, explanations):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    elements = [Paragraph("AI-Powered Medical Report Summary", styles['Title']), Spacer(1, 12)]

    for i, row in df.iterrows():
        elements.append(Paragraph(f"<b>{row['Test']} ({row['Status']})</b>", styles['Heading3']))
        elements.append(Paragraph(f"Value: {row['Value']} {row['Unit']}, Normal: {row['Normal Min']}-{row['Normal Max']} {row['Unit']}", styles['Normal']))
        elements.append(Paragraph(f"Explanation: {explanations[i]}", styles['Normal']))
        elements.append(Spacer(1, 12))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# ----- Streamlit App UI -----
st.title("🩺 AI-Powered Medical Report Assistant")
st.markdown("Upload a **PDF or image** of your lab report. The app will extract results, analyze them, and explain in plain language.")

uploaded_file = st.file_uploader("📄 Upload a medical report", type=["pdf", "jpg", "jpeg", "png"])

if uploaded_file:
    st.info("Processing the uploaded file...")
    image = process_uploaded_file(uploaded_file)
    processed_img = preprocess_image(image)
    st.image(processed_img, caption="Preprocessed Image", use_column_width=True)

    text = extract_text_from_image(processed_img)
    st.subheader("📜 OCR Extracted Text")
    st.text(text)

    data = parse_medical_data(text)
    st.subheader("📊 Lab Results")
    st.dataframe(data)

    st.subheader("🧠 AI-Powered Explanations")
    explanations = []
    for _, row in data.iterrows():
        with st.expander(f"{row['Test']} - {row['Status']}"):
            explanation = explain_with_openai(row)
            st.write(explanation)
            explanations.append(explanation)

    if st.button("📄 Generate PDF Summary"):
        pdf_file = generate_pdf_report(data, explanations)
        st.download_button("📥 Download PDF", data=pdf_file, file_name="lab_report_summary.pdf", mime="application/pdf")
