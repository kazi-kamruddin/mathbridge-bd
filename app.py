import os
import tempfile
from pathlib import Path

import streamlit as st
from google import genai
from PIL import Image


MODEL_NAME = "gemma-4-26b-a4b-it"

st.set_page_config(
    page_title="MathBridge BD",
    page_icon="🧮",
    layout="wide",
)

st.title("🧮 MathBridge BD")
st.subheader("Handwritten expression → LaTeX → Bangla study guide")

st.write(
    "Upload a cropped handwritten mathematics or chemistry expression. "
    "Gemma will recognize it, explain it in Bangla, and generate practice material."
)


def get_client():
    api_key = st.secrets.get("GEMMA_API_KEY")

    if not api_key:
        st.error("GEMMA_API_KEY is not configured.")
        st.stop()

    return genai.Client(api_key=api_key)


def analyze_with_gemma(
    image_file,
    student_level: str,
) -> str:
    client = get_client()

    suffix = Path(image_file.name).suffix or ".png"

    with tempfile.NamedTemporaryFile(
        suffix=suffix,
        delete=False,
    ) as temp_file:
        temp_file.write(image_file.getbuffer())
        image_path = temp_file.name

    try:
        uploaded_image = client.files.upload(
            file=image_path,
        )

        prompt = f"""
You are MathBridge BD, an educational assistant for students
in Bangladesh.

Student level: {student_level}

Analyze the handwritten mathematical or chemical expression
in the uploaded image.

Return clean Markdown with these sections:

# Gemma Recognition Report

## LATEX
Write the recognized expression in valid LaTeX.

## CONFIDENCE
Write high, medium, or low.

## UNCLEAR
Mention uncertain symbols, or write none.

# শনাক্ত করা সমীকরণ

# সহজ ব্যাখ্যা

# প্রতীকগুলোর অর্থ

# সম্ভাব্য বিষয়

# সাধারণ ভুল

# অনুশীলনী

# উত্তর

Rules:
- Transcribe only what is visible.
- Preserve fractions, powers, roots, subscripts, brackets,
  arrows, and chemical notation.
- Do not invent missing information.
- Clearly state uncertainty.
- Do not treat chemistry as ordinary algebra.
- Explain in concise Bangla suitable for the selected level.
"""

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                uploaded_image,
                prompt,
            ],
        )

        return response.text or "No response was returned."

    finally:
        Path(image_path).unlink(missing_ok=True)


left, right = st.columns(2)

with left:
    uploaded_file = st.file_uploader(
        "Upload handwritten expression",
        type=["png", "jpg", "jpeg"],
    )

    student_level = st.selectbox(
        "Student level",
        [
            "Class 6–8",
            "SSC",
            "HSC",
            "University",
        ],
        index=1,
    )

    analyze_button = st.button(
        "Analyze with Gemma",
        type="primary",
        use_container_width=True,
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(
            image,
            caption="Uploaded expression",
            use_container_width=True,
        )

with right:
    if analyze_button:
        if uploaded_file is None:
            st.warning("Please upload an image first.")
        else:
            try:
                with st.spinner("Gemma is analyzing the expression..."):
                    result = analyze_with_gemma(
                        uploaded_file,
                        student_level,
                    )

                st.markdown(result)

                st.download_button(
                    "Download Markdown report",
                    data=result,
                    file_name="mathbridge_report.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

            except Exception as error:
                st.error(
                    f"{type(error).__name__}: {error}"
                )
    else:
        st.info("Your Bangla study guide will appear here.")