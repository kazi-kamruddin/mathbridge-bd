import os
import re
import tempfile
from pathlib import Path

import requests
import streamlit as st
from fpdf import FPDF
from google import genai
from PIL import Image


# =========================================================
# Configuration
# =========================================================

MODEL_NAME = "gemma-4-26b-a4b-it"

st.set_page_config(
    page_title="MathBridge BD",
    page_icon="🧮",
    layout="wide",
)

st.title("🧮 MathBridge BD")
st.subheader("Handwritten expression → LaTeX → Bangla study guide")

st.write(
    "Upload handwritten mathematics or chemistry. Gemma will recognize "
    "the expression, explain it according to your education level, and "
    "answer follow-up questions about the same topic."
)


# =========================================================
# Session state
# =========================================================

DEFAULT_STATE = {
    "analysis": None,
    "recognized_context": None,
    "chat_messages": [],
    "markdown_bytes": None,
    "pdf_bytes": None,
    "report_name": "mathbridge_report",
}

for key, default_value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


def clear_current_work() -> None:
    """Remove the analysis and chat while keeping the app usable."""
    st.session_state.analysis = None
    st.session_state.recognized_context = None
    st.session_state.chat_messages = []
    st.session_state.markdown_bytes = None
    st.session_state.pdf_bytes = None
    st.session_state.report_name = "mathbridge_report"


# =========================================================
# API client
# =========================================================

@st.cache_resource
def get_client():
    api_key = st.secrets.get("GEMMA_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMMA_API_KEY is missing from Streamlit secrets."
        )

    return genai.Client(api_key=api_key)


# =========================================================
# Education-level behavior
# =========================================================

LEVEL_GUIDANCE = {
    "Class 6–8": """
Target a learner aged approximately 11–14.

- Use very simple Bangla.
- Avoid advanced terminology unless immediately explained.
- Use short sentences.
- Explain the basic purpose of every symbol.
- Use one familiar real-life analogy where appropriate.
- Give one very easy practice question.
- Do not introduce concepts beyond the visible expression.
""",
    "SSC": """
Target a Bangladesh SSC-level learner.

- Use clear textbook-style Bangla.
- Name the relevant chapter or curriculum concept.
- Explain the main mathematical or scientific rule.
- Show one short reasoning step.
- Mention one common examination mistake.
- Give one SSC-style practice question.
""",
    "HSC": """
Target a Bangladesh HSC-level learner.

- Use precise academic Bangla.
- Explain the underlying principle and structural relationships.
- Show intermediate reasoning where relevant.
- Mention assumptions, domain restrictions, balancing rules,
  notation conventions, or conceptual conditions.
- Give one conceptual and one calculation-oriented question.
""",
    "University": """
Target an undergraduate university learner.

- Use technical terminology in Bangla with English terms in parentheses.
- Explain formal definitions, notation, and assumptions.
- Discuss structural interpretation and possible generalization.
- Mention ambiguity or alternative interpretations.
- Provide a more advanced extension or related concept.
- Give one analytical follow-up problem.
""",
}


# =========================================================
# Initial Gemma image analysis
# =========================================================

def analyze_with_gemma(
    uploaded_file,
    student_level: str,
) -> str:
    client = get_client()

    suffix = Path(uploaded_file.name).suffix.lower() or ".png"

    with tempfile.NamedTemporaryFile(
        suffix=suffix,
        delete=False,
    ) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        image_path = temp_file.name

    try:
        uploaded_image = client.files.upload(file=image_path)

        level_instruction = LEVEL_GUIDANCE[student_level]

        prompt = f"""
You are MathBridge BD, an educational application for
students in Bangladesh.

STUDENT LEVEL:
{student_level}

LEVEL-SPECIFIC INSTRUCTIONS:
{level_instruction}

Analyze the handwritten mathematics or chemistry expression
in the uploaded image.

Return clean Markdown with exactly these major sections:

# Gemma Recognition Report

## LATEX
Write the recognized expression in valid LaTeX.

## CONFIDENCE
Write high, medium, or low.

## UNCLEAR
Identify uncertain handwriting, or write "none".

# শনাক্ত করা সমীকরণ

Show the recognized expression clearly.

# সহজ ব্যাখ্যা

Explain it according to the selected education level.

# ধাপে ধাপে বিশ্লেষণ

Show an appropriate amount of reasoning for the selected level.

# প্রতীকগুলোর অর্থ

Explain important variables, operators, functions, chemical
symbols, arrows, subscripts, superscripts, and constants.

# সম্ভাব্য বিষয়

Identify the chapter, topic, or scientific concept.

# সাধারণ ভুল

Mention likely learner or examination mistakes.

# অনুশীলনী

Create practice material appropriate to the selected level.

# উত্তর

Give the answer with suitable explanation.

# পরবর্তী প্রশ্নের প্রস্তাব

Suggest three short follow-up questions that the student
could ask about this exact topic.

Rules:
- Transcribe only what is visible.
- Preserve fractions, roots, powers, subscripts, brackets,
  arrows, matrices, functions, and chemical notation.
- Do not invent missing symbols.
- State uncertainty clearly.
- Distinguish mathematics from chemistry.
- Do not claim that an ambiguous expression has only one
  interpretation.
- The educational depth must visibly differ according to
  the selected student level.
"""

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[uploaded_image, prompt],
        )

        result = response.text

        if not result:
            raise RuntimeError("Gemma returned an empty response.")

        return result

    finally:
        Path(image_path).unlink(missing_ok=True)


# =========================================================
# Follow-up conversation
# =========================================================

def answer_follow_up(question: str) -> str:
    client = get_client()

    if not st.session_state.recognized_context:
        raise RuntimeError("Analyze an image before asking questions.")

    recent_history = st.session_state.chat_messages[-8:]

    conversation = "\n\n".join(
        f"{message['role'].upper()}: {message['content']}"
        for message in recent_history
    )

    prompt = f"""
You are the follow-up tutor inside MathBridge BD.

The student has already uploaded a handwritten expression.
The original Gemma analysis is below.

ORIGINAL ANALYSIS:
{st.session_state.recognized_context}

EDUCATION LEVEL:
{st.session_state.get("selected_level", "SSC")}

LEVEL REQUIREMENTS:
{LEVEL_GUIDANCE[
    st.session_state.get("selected_level", "SSC")
]}

RECENT CONVERSATION:
{conversation}

NEW STUDENT QUESTION:
{question}

Instructions:
- Answer specifically about the recognized expression or its
  directly related topic.
- Maintain continuity with the original analysis.
- Answer in clear Bangla unless the student requests English.
- Preserve correct LaTeX notation.
- Do not pretend that unclear handwriting is certain.
- If the question is unrelated, briefly say that the chat is
  currently focused on the uploaded expression.
- Do not repeat the entire original report.
- Give a focused, educational answer.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    if not response.text:
        raise RuntimeError("Gemma returned an empty follow-up response.")

    return response.text


# =========================================================
# Export helpers
# =========================================================

def create_full_markdown() -> str:
    analysis = st.session_state.analysis or ""

    sections = [
        "# MathBridge BD Study Report",
        "",
        analysis,
    ]

    if st.session_state.chat_messages:
        sections.extend(
            [
                "",
                "---",
                "",
                "# Follow-up Questions",
                "",
            ]
        )

        for message in st.session_state.chat_messages:
            speaker = (
                "Student"
                if message["role"] == "user"
                else "MathBridge BD"
            )

            sections.extend(
                [
                    f"## {speaker}",
                    "",
                    message["content"],
                    "",
                ]
            )

    return "\n".join(sections)


def markdown_to_plain_text(markdown_text: str) -> str:
    """Make Markdown safer for a simple PDF text renderer."""
    text = markdown_text

    # Convert headings into plain lines.
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)

    # Remove emphasis and code markers.
    text = text.replace("**", "")
    text = text.replace("__", "")
    text = text.replace("`", "")

    # Keep LaTeX content but remove display delimiters.
    text = text.replace("$$", "")
    text = text.replace("$", "")

    # Convert Markdown links to readable text.
    text = re.sub(
        r"\[([^\]]+)\]\([^)]+\)",
        r"\1",
        text,
    )

    # Convert bullets to a PDF-friendly bullet substitute.
    text = re.sub(
        r"^\s*[-*+]\s+",
        "• ",
        text,
        flags=re.MULTILINE,
    )

    return text.strip()


@st.cache_resource
def get_bengali_font_path() -> str:
    """
    Download a Bangla-capable Google Font once per app process.

    Google Fonts returns a WOFF2 font, which current fpdf2 can
    embed as a Unicode font.
    """
    font_directory = Path(tempfile.gettempdir()) / "mathbridge_fonts"
    font_directory.mkdir(parents=True, exist_ok=True)

    font_path = font_directory / "NotoSansBengali.woff2"

    if font_path.exists() and font_path.stat().st_size > 10_000:
        return str(font_path)

    css_url = (
        "https://fonts.googleapis.com/css2?"
        "family=Noto+Sans+Bengali:wght@400&display=swap"
    )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 AppleWebKit/537.36 "
            "Chrome/120 Safari/537.36"
        )
    }

    css_response = requests.get(
        css_url,
        headers=headers,
        timeout=30,
    )
    css_response.raise_for_status()

    font_urls = re.findall(
        r"url\((https://[^)]+)\)",
        css_response.text,
    )

    if not font_urls:
        raise RuntimeError(
            "Could not locate the Bangla font download URL."
        )

    font_response = requests.get(
        font_urls[-1],
        headers=headers,
        timeout=60,
    )
    font_response.raise_for_status()

    font_path.write_bytes(font_response.content)

    return str(font_path)


def create_pdf_bytes(markdown_text: str) -> bytes:
    plain_text = markdown_to_plain_text(markdown_text)
    font_path = get_bengali_font_path()

    pdf = FPDF(
        orientation="P",
        unit="mm",
        format="A4",
    )

    pdf.set_auto_page_break(
        auto=True,
        margin=15,
    )

    pdf.add_page()

    pdf.add_font(
        family="NotoBangla",
        style="",
        fname=font_path,
    )

    # Bangla is an Indic script, so text shaping is important.
    pdf.set_text_shaping(
        use_shaping_engine=True,
        direction="ltr",
    )

    pdf.set_font(
        "NotoBangla",
        size=11,
    )

    pdf.multi_cell(
        w=0,
        h=7,
        text=plain_text,
        new_x="LMARGIN",
        new_y="NEXT",
    )

    return bytes(pdf.output())


def refresh_exports() -> None:
    if not st.session_state.analysis:
        st.session_state.markdown_bytes = None
        st.session_state.pdf_bytes = None
        return

    full_markdown = create_full_markdown()

    st.session_state.markdown_bytes = full_markdown.encode("utf-8")

    try:
        st.session_state.pdf_bytes = create_pdf_bytes(
            full_markdown
        )
    except Exception as error:
        st.session_state.pdf_bytes = None
        st.session_state.pdf_error = str(error)


# =========================================================
# Main interface
# =========================================================

left_column, right_column = st.columns(
    [1, 1],
    gap="large",
)

with left_column:
    uploaded_file = st.file_uploader(
        "Upload handwritten expression",
        type=["png", "jpg", "jpeg", "webp"],
        key="expression_upload",
    )

    selected_level = st.selectbox(
        "Student level",
        list(LEVEL_GUIDANCE.keys()),
        index=1,
    )

    st.session_state.selected_level = selected_level

    analyze_button = st.button(
        "Analyze with Gemma",
        type="primary",
        use_container_width=True,
    )

    clear_button = st.button(
        "Start a new expression",
        use_container_width=True,
    )

    if clear_button:
        clear_current_work()
        st.rerun()

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded expression",
            use_container_width=True,
        )

with right_column:
    if analyze_button:
        if uploaded_file is None:
            st.warning("Please upload an image first.")
        else:
            try:
                with st.spinner(
                    "Gemma is analyzing the expression..."
                ):
                    analysis = analyze_with_gemma(
                        uploaded_file,
                        selected_level,
                    )

                st.session_state.analysis = analysis
                st.session_state.recognized_context = analysis
                st.session_state.chat_messages = []
                st.session_state.report_name = (
                    f"mathbridge_{selected_level.lower().replace(' ', '_')}"
                )

                refresh_exports()

            except Exception as error:
                st.error(
                    f"{type(error).__name__}: {error}"
                )

    # This is intentionally outside the button block.
    # Therefore, it remains visible after reruns.
    if st.session_state.analysis:
        st.markdown(st.session_state.analysis)
    else:
        st.info(
            "Your Gemma-generated Bangla study guide "
            "will appear here."
        )


# =========================================================
# Persistent downloads
# =========================================================

if st.session_state.analysis:
    st.divider()
    st.subheader("⬇️ Export your complete study report")

    # Regenerate exports after follow-up responses.
    refresh_exports()

    markdown_column, pdf_column = st.columns(2)

    with markdown_column:
        st.download_button(
            label="Download Markdown",
            data=st.session_state.markdown_bytes,
            file_name=(
                f"{st.session_state.report_name}.md"
            ),
            mime="text/markdown",
            use_container_width=True,
            on_click="ignore",
        )

    with pdf_column:
        if st.session_state.pdf_bytes:
            st.download_button(
                label="Download PDF",
                data=st.session_state.pdf_bytes,
                file_name=(
                    f"{st.session_state.report_name}.pdf"
                ),
                mime="application/pdf",
                use_container_width=True,
                on_click="ignore",
            )
        else:
            st.warning(
                "PDF generation is temporarily unavailable. "
                "Markdown download is still ready."
            )

            if st.session_state.get("pdf_error"):
                with st.expander("PDF error details"):
                    st.code(st.session_state.pdf_error)


# =========================================================
# Follow-up tutor chat
# =========================================================

if st.session_state.analysis:
    st.divider()
    st.subheader("💬 Ask follow-up questions")

    st.caption(
        "The tutor remembers the recognized expression and "
        "the current study guide."
    )

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    student_question = st.chat_input(
        "Ask about the expression, explanation, symbols, or solution..."
    )

    if student_question:
        st.session_state.chat_messages.append(
            {
                "role": "user",
                "content": student_question,
            }
        )

        with st.chat_message("user"):
            st.markdown(student_question)

        try:
            with st.chat_message("assistant"):
                with st.spinner("Gemma is preparing an answer..."):
                    assistant_answer = answer_follow_up(
                        student_question
                    )

                st.markdown(assistant_answer)

            st.session_state.chat_messages.append(
                {
                    "role": "assistant",
                    "content": assistant_answer,
                }
            )

            # Include the new chat in both download formats.
            refresh_exports()

        except Exception as error:
            st.error(
                f"{type(error).__name__}: {error}"
            )


# =========================================================
# Safety note
# =========================================================

st.divider()

st.caption(
    "Gemma may misread unclear handwriting. Students should review "
    "the recognized LaTeX before relying on the explanation."
)