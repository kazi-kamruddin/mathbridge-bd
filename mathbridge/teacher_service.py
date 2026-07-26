import csv
import io
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .config import (
    ERROR_CATEGORIES,
    LEVEL_GUIDANCE,
    MAX_TEACHER_ANSWERS,
    MODEL_NAME,
    STATUS_OPTIONS,
)
from .gemma_service import _generate_structured, get_client, upload_to_gemma


TEACHER_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "student_id": {"type": "string"},
        "status": {"type": "string", "enum": STATUS_OPTIONS},
        "error_category": {"type": "string", "enum": ERROR_CATEGORIES},
        "first_error_step": {"type": "string"},
        "short_feedback": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": [
        "student_id",
        "status",
        "error_category",
        "first_error_step",
        "short_feedback",
        "confidence",
    ],
}

TEACHER_BATCH_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": TEACHER_ITEM_SCHEMA,
}


CORRECT_FEEDBACK_PATTERNS = (
    r"\bfully correct\b",
    r"\bcorrectly solved\b",
    r"\bsolution is correct\b",
    r"\ball steps (?:are|were) correct\b",
    r"\banswer is accurate\b",
    r"\bresult(?:ing values?|s)? (?:is|are) accurate\b",
    r"\bno mathematical (?:mistake|error)\b",
    r"\bno error detected\b",
)

INCORRECT_FEEDBACK_PATTERNS = (
    r"\bincorrect\b",
    r"\bwrong\b",
    r"\bmistake\b",
    r"\berror\b",
    r"\bshould have\b",
    r"\bneeds? to\b",
)

UNCLEAR_FEEDBACK_PATTERNS = (
    r"\bunclear\b",
    r"\billegible\b",
    r"\bcannot determine\b",
    r"\bnot enough (?:work|information)\b",
    r"\bincomplete\b",
)

NO_ERROR_VALUES = {
    "",
    "none",
    "n/a",
    "na",
    "no error",
    "no mathematical mistake detected",
    "no mistake detected",
    "not applicable",
}


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
                answers.append({
                    "student_id": str(row.get("student_id") or f"Student {len(answers)+1}"),
                    "answer": str(row.get("answer", "")),
                })
    answers.extend(parse_answer_blocks(answer_blocks))
    if len(answers) > MAX_TEACHER_ANSWERS:
        raise ValueError(f"Please evaluate at most {MAX_TEACHER_ANSWERS} answers per batch.")
    return answers


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _parse_step_number(value: str) -> int | None:
    match = re.search(r"\b(?:step\s*)?(\d+)\b", value or "", flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _normalize_confidence(value: Any) -> str:
    if isinstance(value, (int, float)):
        if float(value) >= 0.8:
            return "high"
        if float(value) >= 0.5:
            return "medium"
        return "low"
    normalized = str(value or "low").strip().lower()
    return normalized if normalized in {"high", "medium", "low"} else "low"


def reconcile_teacher_item(item: dict[str, Any], fallback_id: str) -> dict[str, str]:
    """Make verdict fields internally consistent before analytics are computed."""
    student_id = str(item.get("student_id") or fallback_id)
    feedback = str(item.get("short_feedback") or "").strip()
    first_error = str(item.get("first_error_step") or "").strip()
    status = str(item.get("status") or "Incomplete or Unclear").strip()
    category = str(item.get("error_category") or "Other").strip()

    feedback_lower = feedback.lower()
    first_error_lower = first_error.lower()
    explicit_correct = _contains_any(feedback_lower, CORRECT_FEEDBACK_PATTERNS)
    explicit_problem = _contains_any(feedback_lower, INCORRECT_FEEDBACK_PATTERNS)
    explicit_unclear = _contains_any(feedback_lower, UNCLEAR_FEEDBACK_PATTERNS)
    no_error_step = first_error_lower in NO_ERROR_VALUES
    step_number = _parse_step_number(first_error)

    # A positive explanation plus no actual error must never be counted as incorrect.
    if explicit_correct and not explicit_problem:
        status = "Fully Correct"
        category = "No Error"
        first_error = "No mathematical mistake detected"

    # Keep all fields aligned when the model explicitly chose Fully Correct.
    elif status == "Fully Correct":
        category = "No Error"
        first_error = "No mathematical mistake detected"

    # Reserve Incomplete/Unclear for genuine missing or unreadable work.
    elif explicit_unclear or category in {"Handwriting Unclear", "Incomplete Solution"}:
        status = "Incomplete or Unclear"
        if category not in ERROR_CATEGORIES:
            category = "Handwriting Unclear" if "unclear" in feedback_lower else "Incomplete Solution"

    # A definite diagnosed mistake means the work is gradable, not merely unclear.
    elif explicit_problem or (not no_error_step and first_error):
        if step_number is not None and step_number <= 1:
            status = "Fully Incorrect"
        else:
            status = "Partially Correct"
        if category == "No Error" or category not in ERROR_CATEGORIES:
            category = "Other"

    # If no error is present but the status is contradictory, prefer a correct verdict.
    elif no_error_step and status == "Incomplete or Unclear" and feedback:
        status = "Fully Correct"
        category = "No Error"
        first_error = "No mathematical mistake detected"

    if status not in STATUS_OPTIONS:
        status = "Incomplete or Unclear"
    if category not in ERROR_CATEGORIES:
        category = "Other"

    return {
        "student_id": student_id,
        "status": status,
        "error_category": category,
        "first_error_step": first_error or "Could not determine",
        "short_feedback": feedback,
        "confidence": _normalize_confidence(item.get("confidence")),
    }


def evaluate_teacher_batch(question_text, question_upload, answers, answer_images, level) -> tuple[pd.DataFrame, str]:
    client = get_client()
    question_image = upload_to_gemma(question_upload)
    results: list[dict[str, Any]] = []

    rubric = f"""
VERDICT CONTRACT — all fields must agree:
- Fully Correct: every meaningful step and final answer are correct. Use error_category
  "No Error" and first_error_step "No mathematical mistake detected".
- Partially Correct: at least one meaningful early step is correct, but a later step is
  incorrect or missing. State the first wrong/missing step.
- Fully Incorrect: the first meaningful step/method is wrong, or the answer shows no
  valid progress toward the solution.
- Incomplete or Unclear: use ONLY when the work is too incomplete or unreadable to
  judge. Never use this status when you can identify a definite mathematical mistake,
  and never use it for a fully correct solution.
Before returning, verify that status, error_category, first_error_step, and
short_feedback do not contradict one another.
"""

    if answers:
        prompt = f"""
You are the teacher analytics engine of MathBridge.
LEVEL: {level}
LEVEL INSTRUCTIONS: {LEVEL_GUIDANCE[level]}
QUESTION TEXT: {question_text or "[Question is supplied as an image]"}
STUDENT ANSWERS: {json.dumps(answers, ensure_ascii=False)}

{rubric}

For every student, check the work in order, determine the verdict, identify the
first incorrect or missing step, and classify the mistake. Preserve every supplied
student_id. Return only the structured array requested by the response schema.
"""
        contents: list[Any] = [prompt]
        if question_image is not None:
            contents.extend(["QUESTION IMAGE:", question_image])
        parsed = _generate_structured(
            contents,
            TEACHER_BATCH_SCHEMA,
            "an array of internally consistent teacher evaluation records",
        )
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

{rubric}

Read the complete solution before deciding. Return only the structured evaluation
object requested by the response schema.
"""
        contents: list[Any] = [prompt]
        if question_image is not None:
            contents.extend(["QUESTION IMAGE:", question_image])
        contents.extend(["STUDENT ANSWER IMAGE:", answer_image])
        item = _generate_structured(
            contents,
            TEACHER_ITEM_SCHEMA,
            "one internally consistent teacher evaluation record",
        )
        if isinstance(item, dict):
            results.append(item)

    normalized = [
        reconcile_teacher_item(item, f"Student {index}")
        for index, item in enumerate(results, 1)
    ]

    df = pd.DataFrame(normalized)
    if df.empty:
        raise ValueError("No student answers were evaluated.")

    summary_prompt = f"""
Create a concise teacher-facing English summary grounded only in these corrected
class analytics. Identify the dominant misconception, the most urgent reteaching
priority, and one recommended mini-lesson. Do not call correct work incomplete.
STATUS COUNTS: {df['status'].value_counts().to_dict()}
ERROR COUNTS: {df['error_category'].value_counts().to_dict()}
SAMPLE FIRST ERRORS: {df['first_error_step'].head(10).tolist()}
"""
    summary = client.models.generate_content(
        model=MODEL_NAME,
        contents=summary_prompt,
    ).text or "Summary unavailable."
    return df, summary
