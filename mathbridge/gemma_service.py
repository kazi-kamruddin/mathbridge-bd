import json
import re
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st
from google import genai

from .config import ERROR_CATEGORIES, LEVEL_GUIDANCE, MODEL_NAME, STATUS_OPTIONS


STUDENT_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "status": {"type": "string", "enum": STATUS_OPTIONS},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "question_transcription": {"type": "string"},
        "answer_transcription": {"type": "string"},
        "first_error_step": {"type": "string"},
        "first_error_explanation": {"type": "string"},
        "error_category": {"type": "string", "enum": ERROR_CATEGORIES},
        "what_was_done_well": {"type": "array", "items": {"type": "string"}},
        "corrected_solution_steps": {"type": "array", "items": {"type": "string"}},
        "remediation_lesson": {"type": "string"},
        "retry_problem": {"type": "string"},
        "retry_answer": {"type": "string"},
        "copyable_solution": {"type": "string"},
        "follow_up_suggestions": {"type": "array", "items": {"type": "string"}},
        "teacher_note": {"type": "string"},
    },
    "required": [
        "status",
        "confidence",
        "question_transcription",
        "answer_transcription",
        "first_error_step",
        "first_error_explanation",
        "error_category",
        "what_was_done_well",
        "corrected_solution_steps",
        "remediation_lesson",
        "retry_problem",
        "retry_answer",
        "copyable_solution",
        "follow_up_suggestions",
        "teacher_note",
    ],
}


PDF_EXPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "title": {"type": "string"},
        "verdict": {"type": "string"},
        "transcriptions": {"type": "string"},
        "first_mistake": {"type": "string"},
        "what_was_correct": {"type": "string"},
        "corrected_solution": {"type": "string"},
        "remediation": {"type": "string"},
        "retry_practice": {"type": "string"},
        "conversation_summary": {"type": "string"},
    },
    "required": [
        "title",
        "verdict",
        "transcriptions",
        "first_mistake",
        "what_was_correct",
        "corrected_solution",
        "remediation",
        "retry_practice",
        "conversation_summary",
    ],
}


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
    """Best-effort parser retained as a fallback for models/endpoints without schema support."""
    cleaned = (text or "").strip().lstrip("\ufeff")
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for start_char in ("{", "["):
        start = cleaned.find(start_char)
        while start != -1:
            try:
                value, _ = decoder.raw_decode(cleaned[start:])
                return value
            except json.JSONDecodeError:
                start = cleaned.find(start_char, start + 1)

    raise ValueError("Gemma did not return valid JSON.")


def _generate_structured(contents: list[Any] | str, schema: dict[str, Any], purpose: str) -> Any:
    """Request schema-constrained JSON, with a repair fallback for API/model variance."""
    client = get_client()

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": schema,
                "temperature": 0.1,
            },
        )
        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            if hasattr(parsed, "model_dump"):
                return parsed.model_dump()
            return parsed
        return extract_json(response.text or "")
    except Exception as structured_error:
        # Some model/API combinations may not accept response_json_schema.
        # Make one ordinary call, then repair only if its JSON is malformed.
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config={"temperature": 0.1},
        )
        raw = response.text or ""
        try:
            return extract_json(raw)
        except ValueError:
            repair_prompt = f"""
Convert the following malformed model output into ONE valid JSON value for this purpose:
{purpose}

Required JSON Schema:
{json.dumps(schema, ensure_ascii=False)}

Rules:
- Preserve the original mathematical meaning.
- Do not add commentary or Markdown fences.
- Return JSON only.

Malformed output:
{raw}
"""
            repaired = client.models.generate_content(
                model=MODEL_NAME,
                contents=repair_prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": schema,
                    "temperature": 0,
                },
            )
            try:
                parsed = getattr(repaired, "parsed", None)
                if parsed is not None:
                    if hasattr(parsed, "model_dump"):
                        return parsed.model_dump()
                    return parsed
                return extract_json(repaired.text or "")
            except Exception as repair_error:
                raise ValueError(
                    "Gemma could not produce the required structured result. "
                    "Please retry once with clearer/cropped images."
                ) from repair_error


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
    if not isinstance(data, dict):
        data = {}
    for key, value in defaults.items():
        data.setdefault(key, value)
    if data["status"] not in STATUS_OPTIONS:
        data["status"] = "Incomplete or Unclear"
    if data["confidence"] not in {"high", "medium", "low"}:
        data["confidence"] = "low"
    if data["error_category"] not in ERROR_CATEGORIES:
        data["error_category"] = "Other"
    for key in ["what_was_done_well", "corrected_solution_steps", "follow_up_suggestions"]:
        if not isinstance(data[key], list):
            data[key] = [str(data[key])] if data[key] else []
        data[key] = [str(item) for item in data[key]]
    for key in defaults:
        if key not in {"what_was_done_well", "corrected_solution_steps", "follow_up_suggestions"}:
            data[key] = str(data[key])
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

Analyze the question and the student's attempted solution. The attempt may be
partial, fully correct, partly correct, fully incorrect, or unclear.

Requirements:
- Transcribe the question and answer faithfully.
- Check the student's steps in their written order.
- Identify the FIRST incorrect or missing step, not merely the final answer.
- Preserve and praise correct work before that point.
- If the work is fully correct, use status Fully Correct, category No Error,
  and first_error_step exactly: No mathematical mistake detected.
- If handwriting is genuinely ambiguous, lower confidence and say what is unclear.
- Explain the diagnosis in Bangla at the selected level.
- Produce a clean corrected solution with one step per line.
- Return only the structured diagnostic object requested by the response schema.
"""

    contents: list[Any] = [prompt]
    if question_image is not None:
        contents.extend(["QUESTION IMAGE:", question_image])
    if answer_image is not None:
        contents.extend(["STUDENT ANSWER IMAGE:", answer_image])

    result = _generate_structured(
        contents,
        STUDENT_RESULT_SCHEMA,
        "a MathBridge student mistake-diagnosis object",
    )
    return normalize_student_result(result)


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
Create a polished English-only student diagnostic report from the diagnostic
object and follow-up chat.

Requirements:
- Use readable English only.
- Do not output Markdown headings, JSON, or LaTeX commands.
- Write equations in plain readable notation, for example: (a+b)^2 = a^2 + 2ab + b^2.
- Preserve the logical sequence of the corrected solution.
- Summarize follow-up conversation faithfully.
- Return only the structured report object requested by the response schema.

LEVEL: {level}
ANALYSIS: {json.dumps(analysis, ensure_ascii=False)}
CHAT: {json.dumps(history, ensure_ascii=False)}
"""
    result = _generate_structured(
        prompt,
        PDF_EXPORT_SCHEMA,
        "a clean English MathBridge PDF report object",
    )
    if not isinstance(result, dict):
        raise ValueError("Gemma returned an invalid PDF export object.")
    return result
