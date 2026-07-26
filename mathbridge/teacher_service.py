import csv
import io
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ERROR_CATEGORIES, LEVEL_GUIDANCE, MAX_TEACHER_ANSWERS, MODEL_NAME, STATUS_OPTIONS
from .gemma_service import extract_json, get_client, upload_to_gemma


def parse_answer_blocks(raw_text: str) -> list[dict[str, str]]:
    answers = []
    for index, block in enumerate(raw_text.split("---"), 1):
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        if lines and lines[0].lower().startswith("id:"):
            student_id = lines[0].split(":", 1)[1].strip() or f"Student {index}"
            answer = "\n".join(lines[1:]).strip()
        else:
            student_id, answer = f"Student {index}", block
        answers.append({"student_id": student_id, "answer": answer})
    return answers


def read_teacher_answers(csv_upload, answer_blocks: str) -> list[dict[str, str]]:
    answers: list[dict[str, str]] = []
    if csv_upload is not None:
        decoded = csv_upload.getvalue().decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(decoded))
        if not {"student_id", "answer"}.issubset(set(reader.fieldnames or [])):
            raise ValueError("CSV must contain columns named student_id and answer.")
        for row in reader:
            if str(row.get("answer", "")).strip():
                answers.append({"student_id": str(row.get("student_id") or f"Student {len(answers)+1}"), "answer": str(row.get("answer", ""))})
    answers.extend(parse_answer_blocks(answer_blocks))
    if len(answers) > MAX_TEACHER_ANSWERS:
        raise ValueError(f"Please evaluate at most {MAX_TEACHER_ANSWERS} answers per batch.")
    return answers


def evaluate_teacher_batch(question_text, question_upload, answers, answer_images, level) -> tuple[pd.DataFrame, str]:
    client = get_client()
    question_image = upload_to_gemma(question_upload)
    results: list[dict[str, Any]] = []
    if answers:
        prompt = f"""
You are the teacher analytics engine of MathBridge.
LEVEL: {level}
LEVEL INSTRUCTIONS: {LEVEL_GUIDANCE[level]}
QUESTION TEXT: {question_text or "[Question is supplied as an image]"}
STUDENT ANSWERS: {json.dumps(answers, ensure_ascii=False)}
For every student, determine one status from {STATUS_OPTIONS}, find the first
incorrect or missing step, and classify it using one category from
{ERROR_CATEGORIES}. Return ONLY a valid JSON array with fields student_id,
status, error_category, first_error_step, short_feedback, confidence.
"""
        contents: list[Any] = [prompt]
        if question_image is not None:
            contents.append(question_image)
        response = client.models.generate_content(model=MODEL_NAME, contents=contents)
        parsed = extract_json(response.text or "")
        if not isinstance(parsed, list):
            raise ValueError("Gemma did not return a list for the teacher evaluation.")
        results.extend(parsed)
    for image_file in answer_images or []:
        answer_image = upload_to_gemma(image_file)
        prompt = f"""
Evaluate this single handwritten student answer against the supplied question.
LEVEL: {level}
QUESTION TEXT: {question_text or "[Question is supplied as an image]"}
STUDENT ID: {Path(image_file.name).stem}
Return ONLY valid JSON with student_id, status, error_category,
first_error_step, short_feedback, confidence.
"""
        contents = [prompt]
        if question_image is not None:
            contents.append(question_image)
        contents.append(answer_image)
        response = client.models.generate_content(model=MODEL_NAME, contents=contents)
        results.append(extract_json(response.text or ""))
    normalized = []
    for index, item in enumerate(results, 1):
        normalized.append({
            "student_id": str(item.get("student_id", f"Student {index}")),
            "status": item.get("status") if item.get("status") in STATUS_OPTIONS else "Incomplete or Unclear",
            "error_category": item.get("error_category") if item.get("error_category") in ERROR_CATEGORIES else "Other",
            "first_error_step": str(item.get("first_error_step", "Could not determine")),
            "short_feedback": str(item.get("short_feedback", "")),
            "confidence": str(item.get("confidence", "low")),
        })
    df = pd.DataFrame(normalized)
    if df.empty:
        raise ValueError("No student answers were evaluated.")
    summary_prompt = f"""
Create a concise teacher-facing English summary. Identify the dominant
misconception, the most urgent reteaching priority, and one recommended
mini-lesson. STATUS COUNTS: {df['status'].value_counts().to_dict()}
ERROR COUNTS: {df['error_category'].value_counts().to_dict()}
SAMPLE FIRST ERRORS: {df['first_error_step'].head(10).tolist()}
"""
    summary = client.models.generate_content(model=MODEL_NAME, contents=summary_prompt).text or "Summary unavailable."
    return df, summary
