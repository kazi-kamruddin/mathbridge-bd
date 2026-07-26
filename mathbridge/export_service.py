import io
import re
from typing import Any

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def student_markdown_report(data: dict[str, Any], chat: list[dict]) -> str:
    lines = [
        "# MathBridge Student Diagnostic Report",
        "",
        f"**Status:** {data.get('status', '')}",
        f"**Confidence:** {data.get('confidence', '')}",
        f"**Error category:** {data.get('error_category', '')}",
        "",
        "## Question transcription",
        data.get("question_transcription", ""),
        "",
        "## Student answer transcription",
        data.get("answer_transcription", ""),
        "",
        "## First mistake",
        data.get("first_error_step", ""),
        "",
        data.get("first_error_explanation", ""),
        "",
        "## What was done well",
    ]
    lines.extend(f"- {item}" for item in data.get("what_was_done_well", []))
    lines.extend(["", "## Corrected solution"])
    lines.extend(
        f"{index}. {step}" for index, step in enumerate(data.get("corrected_solution_steps", []), 1)
    )
    lines.extend([
        "", "## Remediation lesson", data.get("remediation_lesson", ""),
        "", "## Retry problem", data.get("retry_problem", ""),
        "", "## Retry answer", data.get("retry_answer", ""),
    ])
    if chat:
        lines.extend(["", "---", "", "# Follow-up conversation"])
        for message in chat:
            speaker = "Student" if message["role"] == "user" else "MathBridge"
            lines.extend(["", f"## {speaker}", message["content"]])
    return "\n".join(lines)


def clean_text_for_pdf(value: str) -> str:
    text = str(value).replace("**", "").replace("__", "").replace("`", "")
    text = re.sub(r"```(?:\w+)?", "", text).replace("```", "").replace("$", "")
    text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", text)
    text = text.replace(r"\cdot", " x ").replace(r"\times", " x ")
    text = text.replace(r"\sqrt", "sqrt").replace(r"\left", "").replace(r"\right", "")
    text = re.sub(r"\\text\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"\\[A-Za-z]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def build_pdf_bytes(title: str, sections: list[tuple[str, str]], table_df: pd.DataFrame | None = None) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=16 * mm, leftMargin=16 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm, title=title,
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=20, leading=24, alignment=TA_CENTER, spaceAfter=12))
    styles.add(ParagraphStyle(name="SectionTitle", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=13, leading=16, spaceBefore=10, spaceAfter=5))
    styles.add(ParagraphStyle(name="BodyReadable", parent=styles["BodyText"], fontName="Helvetica", fontSize=10.5, leading=15, alignment=TA_LEFT, spaceAfter=6))
    story = [Paragraph(title, styles["ReportTitle"]), Spacer(1, 4 * mm)]
    for heading, body in sections:
        story.append(Paragraph(clean_text_for_pdf(heading), styles["SectionTitle"]))
        parts = [part.strip() for part in str(body).split("\n") if part.strip()] or ["Not available."]
        for paragraph in parts:
            safe = clean_text_for_pdf(paragraph).replace("&", "&amp;")
            story.append(Paragraph(safe, styles["BodyReadable"]))
    if table_df is not None and not table_df.empty:
        story.extend([PageBreak(), Paragraph("Evaluation Table", styles["SectionTitle"])])
        display_df = table_df.astype(str)
        rows = [list(display_df.columns)] + display_df.values.tolist()
        table = Table(rows, repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 7.5),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FC")]),
        ]))
        story.append(table)
    doc.build(story)
    return buffer.getvalue()


def student_pdf_from_export(export: dict) -> bytes:
    sections = [
        ("Verdict", export.get("verdict", "")),
        ("Question and answer transcription", export.get("transcriptions", "")),
        ("First detected mistake", export.get("first_mistake", "")),
        ("What the student did correctly", export.get("what_was_correct", "")),
        ("Corrected solution", export.get("corrected_solution", "")),
        ("Targeted remediation", export.get("remediation", "")),
        ("Retry practice", export.get("retry_practice", "")),
        ("Follow-up conversation summary", export.get("conversation_summary", "")),
    ]
    return build_pdf_bytes(export.get("title", "MathBridge Student Report"), sections)
