import json
import re
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st
from google import genai

from .config import ERROR_CATEGORIES, LEVEL_GUIDANCE, MODEL_NAME, STATUS_OPTIONS


@st.cache_resource
def get_client():
    api_key = st.secrets.get("GEMMA_API_KEY")
    if not api_key:
        raise RuntimeError("GEMMA_API_KEY is missing from Streamlit secrets.")
    return genai.Client(api_key=api_key)


def upload_to_gemma(uploaded_file):
    if uploaded_file is None:
        return None
    suffix = Path(uploaded_file.name).suffix.lower() or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_path = Path(temp_file.name)
    try:
        return get_client().files.upload(file=str(temp_path))
    finally:
        temp_path.unlink(missing_ok=True)


def extract_json(text: str) -> Any:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for pattern in [r"\{.*\}", r"\[.*\]"]:
        match = re.search(pattern, cleaned, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
    raise ValueError("Gemma did not return valid JSON. Please retry.")


def normalize_student_result(data: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "status": "Incomplete or Unclear",
        "confidence": "low",
        "question_transcription": "Not available",
        "answer_transcription": "Not available",
        "first_error_step": "Could not determine",
        "first_error_explanation": "The available work is insufficient or unclear.",
        "error_category": "Other",
        "what_was_done_well": [],
        "corrected_solution_steps": [],
        "remediation_lesson": "Review the relevant rule and retry the problem.",
        "retry_problem": "No retry problem generated.",
        "retry_answer": "Not available",
        "copyable_solution": "No corrected solution available.",
        "follow_up_suggestions": [],
        "teacher_note": "",
    }
    for key, value in defaults.items():
        data.setdefault(key, value)
    if data["status"] not in STATUS_OPTIONS:
        data["status"] = "Incomplete or Unclear"
    if data["error_category"] not in ERROR_CATEGORIES:
        data["error_category"] = "Other"
    for key in ["what_was_done_well", "corrected_solution_steps", "follow_up_suggestions"]:
        if not isinstance(data[key], list):
            data[key] = [str(data[key])]
    return data


def analyze_student_submission(question_text, answer_text, question_upload, answer_upload, level):
    question_image = upload_to_gemma(question_upload)
    answer_image = upload_to_gemma(answer_upload)
    prompt = f"""
You are MathBridge, a Bangla-first diagnostic STEM tutor for Bangladesh.

STUDENT LEVEL: {level}
LEVEL INSTRUCTIONS:
{LEVEL_GUIDANCE[level]}

TYPED QUESTION:
{question_text or "[No typed question provided]"}

TYPED STUDENT ANSWER:
{answer_text or "[No typed answer provided]"}

The answer may be partial, fully correct, partly correct, fully incorrect, or unclear.
Transcribe all available inputs, find the FIRST incorrect or missing step, preserve
correct work, classify the mistake, explain how to repair it, and generate a
level-appropriate corrected solution and retry task.

Return ONLY valid JSON with this schema:
{{
  "status": "Fully Correct | Partially Correct | Fully Incorrect | Incomplete or Unclear",
  "confidence": "high | medium | low",
  "question_transcription": "plain readable transcription",
  "answer_transcription": "plain readable transcription preserving steps",
  "first_error_step": "the exact first wrong or missing step, or No mathematical mistake detected",
  "first_error_explanation": "Bangla explanation suited to the level",
  "error_category": "one of {ERROR_CATEGORIES}",
  "what_was_done_well": ["specific correct observations"],
  "corrected_solution_steps": ["step 1", "step 2"],
  "remediation_lesson": "targeted Bangla mini-lesson",
  "retry_problem": "one related problem",
  "retry_answer": "answer with brief reasoning",
  "copyable_solution": "clean corrected solution in plain text, one step per line",
  "follow_up_suggestions": ["question 1", "question 2", "question 3"],
  "teacher_note": "one concise diagnostic note"
}}
"""
    contents: list[Any] = [prompt]
    if question_image is not None:
        contents.append(question_image)
    if answer_image is not None:
        contents.append(answer_image)
    response = get_client().models.generate_content(model=MODEL_NAME, contents=contents)
    if not response.text:
        raise RuntimeError("Gemma returned an empty response.")
    return normalize_student_result(extract_json(response.text))


def answer_student_follow_up(question: str, level: str, analysis: dict, history: list[dict]) -> str:
    conversation = "\n\n".join(
        f"{item['role'].upper()}: {item['content']}" for item in history[-8:]
    )
    prompt = f"""
You are the continuing tutor inside MathBridge.

STUDENT LEVEL: {level}
LEVEL INSTRUCTIONS:
{LEVEL_GUIDANCE[level]}

DIAGNOSTIC RESULT:
{json.dumps(analysis, ensure_ascii=False, indent=2)}

RECENT CONVERSATION:
{conversation or "[No previous chat]"}

NEW QUESTION:
{question}

Answer in clear Bangla unless the user asks for English. Stay grounded in the
original question, student work, first mistake, and remediation. If the student
asks for a solution, provide the steps in a fenced code block so Streamlit shows
a one-click copy button.
"""
    response = get_client().models.generate_content(model=MODEL_NAME, contents=prompt)
    if not response.text:
        raise RuntimeError("Gemma returned an empty follow-up response.")
    return response.text


def prepare_student_pdf_content(level: str, analysis: dict, history: list[dict]) -> dict:
    prompt = f"""
Create a polished English-only student diagnostic report from the JSON and chat.
The final text will be placed into a PDF.

Requirements:
- Use readable English only.
- Do not output Markdown headings, JSON, or LaTeX commands.
- Write equations in plain readable notation, for example: (a+b)^2 = a^2 + 2ab + b^2.
- Preserve the logical sequence of the corrected solution.
- Summarize follow-up conversation faithfully.
- Return ONLY valid JSON with string fields:
  title, verdict, transcriptions, first_mistake, what_was_correct,
  corrected_solution, remediation, retry_practice, conversation_summary.

LEVEL: {level}
ANALYSIS: {json.dumps(analysis, ensure_ascii=False)}
CHAT: {json.dumps(history, ensure_ascii=False)}
"""
    response = get_client().models.generate_content(model=MODEL_NAME, contents=prompt)
    if not response.text:
        raise RuntimeError("Gemma returned an empty PDF export response.")
    return extract_json(response.text)
