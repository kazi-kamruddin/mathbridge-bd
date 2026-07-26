import json

import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image

from ..config import LEVEL_GUIDANCE, MAX_TEACHER_ANSWERS, STATUS_OPTIONS
from ..export_service import build_pdf_bytes
from ..state import reset_teacher_state
from ..teacher_service import evaluate_teacher_batch, read_teacher_answers


STATUS_ORDER = [
    "Fully Correct",
    "Partially Correct",
    "Fully Incorrect",
    "Incomplete or Unclear",
]

STATUS_COLORS = ["#2E8B57", "#E0A800", "#D9534F", "#6C757D"]


def _status_chart(df: pd.DataFrame) -> alt.Chart:
    counts = (
        df["status"]
        .value_counts()
        .reindex(STATUS_ORDER, fill_value=0)
        .rename_axis("status")
        .reset_index(name="students")
    )
    return (
        alt.Chart(counts)
        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
        .encode(
            x=alt.X("status:N", sort=STATUS_ORDER, title=None, axis=alt.Axis(labelAngle=-20)),
            y=alt.Y("students:Q", title="Number of students", axis=alt.Axis(tickMinStep=1)),
            color=alt.Color(
                "status:N",
                scale=alt.Scale(domain=STATUS_ORDER, range=STATUS_COLORS),
                legend=None,
            ),
            tooltip=[alt.Tooltip("status:N", title="Verdict"), alt.Tooltip("students:Q", title="Students")],
        )
        .properties(title="Correctness distribution", height=360)
    )


def _error_chart(df: pd.DataFrame) -> alt.Chart:
    errors = df.loc[df["error_category"] != "No Error", "error_category"].value_counts()
    if errors.empty:
        chart_data = pd.DataFrame({"error_category": ["No class-wide errors"], "students": [0]})
    else:
        chart_data = errors.rename_axis("error_category").reset_index(name="students")

    return (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5)
        .encode(
            y=alt.Y(
                "error_category:N",
                sort="-x",
                title=None,
                axis=alt.Axis(labelLimit=220),
            ),
            x=alt.X("students:Q", title="Number of students", axis=alt.Axis(tickMinStep=1)),
            tooltip=[
                alt.Tooltip("error_category:N", title="Error category"),
                alt.Tooltip("students:Q", title="Students"),
            ],
        )
        .properties(title="Most common first-error categories", height=360)
    )


def render_teacher_view(storage) -> None:
    st.subheader("Evaluate multiple answers to one question")
    st.write(
        "Upload one question, then provide several student answers. "
        "MathBridge shows correctness, first-error categories, and reteaching priorities."
    )
    level = st.selectbox("Class level", list(LEVEL_GUIDANCE), index=1, key="teacher_level")
    c1, c2 = st.columns(2)
    with c1:
        question_text = st.text_area("Question text", height=130, key="teacher_question_text")
    with c2:
        question_upload = st.file_uploader(
            "Question image (optional)",
            type=["png", "jpg", "jpeg", "webp"],
            key="teacher_question_image",
        )
        if question_upload:
            st.image(Image.open(question_upload), use_container_width=True)

    st.markdown("### Student answers")
    c1, c2 = st.columns(2)
    with c1:
        csv_upload = st.file_uploader(
            "Upload CSV with columns: student_id, answer",
            type=["csv"],
            key="teacher_csv",
        )
        blocks = st.text_area(
            "Or paste answers separated by ---",
            height=220,
            key="teacher_answer_blocks",
        )
    with c2:
        images = st.file_uploader(
            "Or upload multiple handwritten answer images",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key="teacher_answer_images",
        )
        st.info(f"For free-tier reliability, keep each batch at or below {MAX_TEACHER_ANSWERS} answers.")

    c1, c2 = st.columns([3, 1])
    with c1:
        run = st.button("Evaluate class with Gemma", type="primary", use_container_width=True)
    with c2:
        if st.button("Clear class report", use_container_width=True):
            reset_teacher_state()
            st.rerun()

    if run:
        try:
            answers = read_teacher_answers(csv_upload, blocks)
            total = len(answers) + len(images or [])
            if not (question_text.strip() or question_upload):
                st.warning("Provide the common question as text or an image.")
            elif total == 0:
                st.warning("Provide at least one student answer.")
            else:
                with st.spinner("Gemma is evaluating the class..."):
                    df, summary = evaluate_teacher_batch(
                        question_text,
                        question_upload,
                        answers,
                        images,
                        level,
                    )
                st.session_state.teacher_results = df
                st.session_state.teacher_summary = summary
                st.session_state.teacher_pdf = None
        except Exception as error:
            st.error(f"{type(error).__name__}: {error}")

    df = st.session_state.teacher_results
    if df is None or df.empty:
        return

    st.divider()
    st.subheader("Class overview")
    counts = df["status"].value_counts()
    metrics = st.columns(4)
    for index, status in enumerate(STATUS_OPTIONS):
        value = int(counts.get(status, 0))
        metrics[index].metric(status, value, f"{value / len(df):.0%}")

    st.markdown("#### Teacher summary")
    st.markdown(st.session_state.teacher_summary or "")

    chart_left, chart_right = st.columns(2, gap="large")
    with chart_left:
        st.altair_chart(_status_chart(df), use_container_width=True)
    with chart_right:
        st.altair_chart(_error_chart(df), use_container_width=True)

    st.markdown("#### Student-level evaluation")
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "student_id": st.column_config.TextColumn("Student"),
            "status": st.column_config.TextColumn("Verdict"),
            "error_category": st.column_config.TextColumn("First-error category"),
            "first_error_step": st.column_config.TextColumn("First error step", width="medium"),
            "short_feedback": st.column_config.TextColumn("Feedback", width="large"),
            "confidence": st.column_config.TextColumn("Confidence"),
        },
    )

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download evaluation CSV",
            csv_bytes,
            "mathbridge_teacher_evaluation.csv",
            "text/csv",
            use_container_width=True,
            on_click="ignore",
        )
    with c2:
        if st.button("Prepare teacher PDF", use_container_width=True):
            sections = [
                ("Class level", level),
                ("Overall summary", st.session_state.teacher_summary or ""),
                ("Status distribution", json.dumps(df["status"].value_counts().to_dict())),
                ("Error distribution", json.dumps(df["error_category"].value_counts().to_dict())),
            ]
            st.session_state.teacher_pdf = build_pdf_bytes(
                "MathBridge Teacher Evaluation Report",
                sections,
                df[["student_id", "status", "error_category", "first_error_step"]],
            )
        if st.session_state.teacher_pdf:
            st.download_button(
                "Download teacher PDF",
                st.session_state.teacher_pdf,
                "mathbridge_teacher_report.pdf",
                "application/pdf",
                use_container_width=True,
                on_click="ignore",
            )
