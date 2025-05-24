
import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
import re
import pytesseract
import openai
import streamlit as st
import os
import tempfile

# ✅ Set your OpenAI API Key
openai.api_key = "your-openai-api-key"  # Replace with your real key

def load_report(file_path):
    if file_path.lower().endswith(".pdf"):
        images = convert_from_path(file_path, dpi=300)
    else:
        images = [Image.open(file_path)]
    return images

def preprocess_image(pil_img):
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    img = cv2.bilateralFilter(img, 11, 17, 17)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img

def extract_text_from_image(cv2_img):
    return pytesseract.image_to_string(cv2_img)

def structure_medical_data(text):
    results = []
    lines = text.split('\n')
    for line in lines:
        match = re.match(r"([A-Za-z\s]+)\s+([\d.]+)\s+([a-zA-Z%/]+)?\s*(\(?[\d\-\u2013]+[\s\u2013-]*[\d\-\u2013]+\)?)?", line)
        if match:
            test_name = match.group(1).strip()
            value = float(match.group(2))
            unit = match.group(3)
            normal_range = match.group(4)
            status = "Unknown"
            try:
                if normal_range:
                    range_clean = re.sub(r"[^\d\-\u2013]", "", normal_range)
                    range_parts = re.split(r"[-\u2013]", range_clean)
                    if len(range_parts) == 2:
                        low = float(range_parts[0])
                        high = float(range_parts[1])
                        range_span = high - low
                        buffer = range_span * 0.1
                        if low <= value <= high:
                            status = "Normal"
                        elif (low - buffer) <= value < low or high < value <= (high + buffer):
                            status = "Borderline"
                        else:
                            status = "Critical"
            except:
                status = "Unknown"
            results.append({
                "Test": test_name,
                "Value": value,
                "Unit": unit,
                "Normal Range": normal_range,
                "Status": status
            })
    return results

def explain_results(structured_data):
    explanations = {}
    for row in structured_data:
        status = row.get("Status")
        if status not in ["Critical", "Borderline"]:
            continue
        prompt = (
            f"Explain in simple language what it means if the patient's {row['Test']} is {row['Value']} {row['Unit']}, "
            f"given the normal range is {row['Normal Range']}. This value is considered {status}."
        )
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150
            )
            explanation = response['choices'][0]['message']['content']
        except Exception as e:
            explanation = f"Error generating explanation: {e}"
        explanations[row['Test']] = explanation
        row["Explanation"] = explanation
    return explanations

def generate_follow_up_suggestions(structured_data):
    suggestions = []
    for row in structured_data:
        test = row.get("Test")
        status = row.get("Status")
        if status == "Critical":
            suggestions.append(f"{test}: Consult a specialist immediately.")
        elif status == "Borderline":
            suggestions.append(f"{test}: Monitor this value and consider retesting soon.")
    return suggestions

# Streamlit App
st.title("🧠 AI Medical Report Assistant")
st.write("Upload a scanned medical report (PDF or image), and let the AI explain it.")

uploaded_file = st.file_uploader("Upload PDF or Image", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.success("✅ File uploaded successfully!")

    try:
        images = load_report(tmp_path)
        st.image(images[0], caption="First Page of Report", use_column_width=True)

        processed_img = preprocess_image(images[0])
        text = extract_text_from_image(processed_img)

        st.subheader("📄 Extracted Text")
        st.text(text)

        structured = structure_medical_data(text)
        st.subheader("📊 Structured Results")
        st.dataframe(structured)

        explanations = explain_results(structured)
        st.subheader("💬 AI Explanations")
        for row in structured:
            if "Explanation" in row:
                with st.expander(f"{row['Test']} ({row['Status']})"):
                    st.write(row['Explanation'])

        st.subheader("📋 Follow-Up Suggestions")
        suggestions = generate_follow_up_suggestions(structured)
        for s in suggestions:
            st.write(f"- {s}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
