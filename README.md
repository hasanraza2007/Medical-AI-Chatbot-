# Medical-AI-Chatbot-
Medical lab reports—such as blood tests, lipid profiles, and diagnostic summaries—are often filled with technical jargon, abbreviations, and reference values that the average person cannot interpret without a doctor.
❓The Challenge:
Design and develop an AI-powered assistant that can:
● Extract data (text, numbers, tables) from scanned medical reports or PDFs
● Use NLP to analyze and structure the content
● Apply Generative AI to explain test results in simple, human-understandable
language
● Optionally, suggest follow-up actions or flag values that are out of range
🎯 Project Objectives:
1. Input Handling
○ Allow users to upload medical report files in image format (JPEG/PNG) or
scanned PDFs.
○ Preprocess the input (denoising, binarization) using OpenCV to improve
accuracy.
2. Text Extraction (OCR)
○ Use Tesseract or EasyOCR to extract content from reports.
○ Extract structured data like:
■ Test Name
■ Measured Value
■ Normal Range
■ Unit (mg/dL, etc.)
3. NLP-based Structuring
○ Use rule-based or ML-based logic to:
■ Map extracted rows into structured format (dictionary or table).
■ Identify values outside the normal reference range.
■ Categorize values (e.g., Critical, Borderline, Normal).
4. Generative AI Explanation
○ Use GPT-3.5 or Gemini Pro via API to explain each test result using a prompt
like:
“Explain in simple language what it means if the patient’s Hemoglobin is 9.5
g/dL, given the normal range is 13–17 g/dL.”
○ Return explanations for each abnormal result or all if time allows.
5. Optional Risk Summary / Follow-up Suggestion
○ Based on extracted values and explanations, optionally generate:
■ A summary paragraph
■ A list of suggested actions like “Consult a cardiologist” or “Increase iron
intake.”
6. User Interface (Streamlit / Flask)
○ File upload box
○ OCR result viewer
○ Explanations in expandable/collapsible sections
○ Downloadable PDF summary of results
