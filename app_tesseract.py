import streamlit as st
import requests
import tempfile
import io
from PIL import Image
import re
import openai

# 🔐 Set your OpenAI and OCR.space API Keys
openai.api_key = "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"          # Replace with your OpenAI API key
ocr_space_api_key = "K87494185088957"    # Replace with your OCR.space key (get from https://ocr.space/OCRAPI)

# OCR.space API Function
def ocr_space_api(image_bytes):
    url = 'https://api.ocr.space/parse/image'
    payload = {
        'isOverlayRequired': False,
        'apikey': ocr_space_api_key,
        'language': 'eng'
    }
    files = {
        'filename': image_bytes
    }
    response = requests.post(url, files=files, data=payload)
    result = response.json()
    return result['ParsedResults'][0]['ParsedText'] if result.get('ParsedResults') else ""

# Extract structure from OCR text
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

# Use OpenAI to explain abnormal results
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

# Follow-up suggestions based on results
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

# Streamlit UI
st.set_page_config(page_title="🧠 AI Medical Report Assistant", layout="centered")
st.title("🧠 AI Medical Report Assistant")
st.write("Upload a scanned medical report (image only), and let the AI extract, analyze, and explain it.")

uploaded_file = st.file_uploader("📎 Upload Report (PNG, JPG, JPEG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.success("✅ File uploaded successfully!")

    # Convert to bytes for API
    image = Image.open(uploaded_file).convert("RGB")
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    image_bytes = buffered.getvalue()

    st.image(image, caption="Uploaded Report", use_column_width=True)

    with st.spinner("🔍 Extracting text..."):
        extracted_text = ocr_space_api(image_bytes)

    st.subheader("📄 Extracted Text")
    st.text_area("OCR Result", extracted_text, height=300)

    structured = structure_medical_data(extracted_text)
    if structured:
        st.subheader("📊 Structured Test Results")
        st.dataframe(structured)

        st.subheader("💬 AI Explanations")
        explain_results(structured)
        for row in structured:
            if "Explanation" in row:
                with st.expander(f"{row['Test']} ({row['Status']})"):
                    st.write(row['Explanation'])

        st.subheader("📋 Follow-Up Suggestions")
        suggestions = generate_follow_up_suggestions(structured)
        for s in suggestions:
            st.write(f"- {s}")
    else:
        st.warning("⚠️ No structured lab test data could be detected.")
