from datetime import datetime

import streamlit as st
from PIL import Image

from ..config import LEVEL_GUIDANCE
from ..export_service import student_markdown_report, student_pdf_from_export
from ..gemma_service import analyze_student_submission, answer_student_follow_up, prepare_student_pdf_content
from ..state import reset_student_state
from ..storage import workspace_ready


def _session_payload(level: str, question_text: str, answer_text: str) -> dict:
    return {
        "level": level,
        "question_text": question_text,
        "answer_text": answer_text,
        "analysis": st.session_state.student_analysis,
        "chat": st.session_state.student_chat,
        "saved_at": datetime.utcnow().isoformat(),
    }


def _save_current(storage, level: str, question_text: str, answer_text: str) -> None:
    if not workspace_ready() or not st.session_state.student_analysis:
        return
    title = (st.session_state.student_analysis.get("question_transcription") or "Math attempt")[:80]
    st.session_state.student_session_id = storage.save(
        st.session_state.workspace_id,
        st.session_state.workspace_pin,
        "student",
        title,
        _session_payload(level, question_text, answer_text),
        st.session_state.student_session_id,
    )


def render_student_view(storage) -> None:
    st.subheader("Upload the question and your attempted answer")
    st.write("MathBridge finds the first likely mistake, preserves correct work, and creates a targeted retry lesson.")
    level = st.selectbox("Student level", list(LEVEL_GUIDANCE), index=1, key="student_level")
    question_col, answer_col = st.columns(2, gap="large")
    with question_col:
        st.markdown("### 1. Question")
        question_text = st.text_area("Type or paste the question", height=150, key="student_question_text")
        question_upload = st.file_uploader("Or upload a question image", type=["png", "jpg", "jpeg", "webp"], key="student_question_image")
        if question_upload:
            st.image(Image.open(question_upload), caption="Question image", use_container_width=True)
    with answer_col:
        st.markdown("### 2. Your attempted answer")
        answer_text = st.text_area("Type or paste your steps", height=150, key="student_answer_text")
        answer_upload = st.file_uploader("Or upload your handwritten answer", type=["png", "jpg", "jpeg", "webp"], key="student_answer_image")
        if answer_upload:
            st.image(Image.open(answer_upload), caption="Answer image", use_container_width=True)
    action_col, reset_col = st.columns([3, 1])
    with action_col:
        analyze_button = st.button("Find my first mistake with Gemma", type="primary", use_container_width=True)
    with reset_col:
        if st.button("New attempt", use_container_width=True):
            reset_student_state()
            st.rerun()
    if analyze_button:
        if not (question_text.strip() or question_upload):
            st.warning("Please provide the question as text or an image.")
        elif not (answer_text.strip() or answer_upload):
            st.warning("Please provide your attempted answer as text or an image.")
        else:
            try:
                with st.spinner("Gemma is transcribing and checking each step..."):
                    result = analyze_student_submission(question_text, answer_text, question_upload, answer_upload, level)
                st.session_state.student_analysis = result
                st.session_state.student_context = result
                st.session_state.student_chat = []
                st.session_state.student_pdf = None
                if st.session_state.auto_save:
                    _save_current(storage, level, question_text, answer_text)
            except Exception as error:
                st.error(f"{type(error).__name__}: {error}")
    data = st.session_state.student_analysis
    if not data:
        return
    st.divider()
    if data["status"] == "Fully Correct":
        st.success(f"Verdict: {data['status']}")
    elif data["status"] == "Partially Correct":
        st.warning(f"Verdict: {data['status']}")
    else:
        st.error(f"Verdict: {data['status']}")
    metrics = st.columns(3)
    metrics[0].metric("Confidence", data["confidence"].title())
    metrics[1].metric("Error category", data["error_category"])
    metrics[2].metric("Correct steps found", len(data["what_was_done_well"]))
    st.subheader("First detected mistake")
    st.code(data["first_error_step"], language=None)
    st.markdown(data["first_error_explanation"])
    if data["what_was_done_well"]:
        with st.expander("What you did correctly", expanded=True):
            for item in data["what_was_done_well"]:
                st.write(f"- {item}")
    st.subheader("Corrected solution - one-click copy")
    st.code(data["copyable_solution"], language=None)
    with st.expander("Step-by-step corrected solution", expanded=True):
        for index, step in enumerate(data["corrected_solution_steps"], 1):
            st.write(f"**{index}.** {step}")
    remediation_col, retry_col = st.columns(2)
    with remediation_col:
        st.subheader("Targeted remediation")
        st.markdown(data["remediation_lesson"])
    with retry_col:
        st.subheader("Retry problem")
        st.markdown(data["retry_problem"])
        with st.expander("Show retry answer"):
            st.markdown(data["retry_answer"])
    st.divider()
    st.subheader("Export and save")
    markdown = student_markdown_report(data, st.session_state.student_chat).encode("utf-8")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("Download Markdown", markdown, "mathbridge_student_report.md", "text/markdown", use_container_width=True, on_click="ignore")
    with c2:
        if st.button("Prepare clean English PDF", use_container_width=True):
            try:
                with st.spinner("Preparing a readable PDF..."):
                    export = prepare_student_pdf_content(level, data, st.session_state.student_chat)
                    st.session_state.student_pdf = student_pdf_from_export(export)
                    st.session_state.student_pdf_error = None
            except Exception as error:
                st.session_state.student_pdf_error = str(error)
        if st.session_state.student_pdf:
            st.download_button("Download PDF", st.session_state.student_pdf, "mathbridge_student_report.pdf", "application/pdf", use_container_width=True, on_click="ignore")
    with c3:
        if st.button("Save this session", use_container_width=True, disabled=not workspace_ready()):
            _save_current(storage, level, question_text, answer_text)
            st.success("Session saved.")
    if st.session_state.student_pdf_error:
        st.error(st.session_state.student_pdf_error)
    st.divider()
    st.subheader("Follow-up tutor")
    for message in st.session_state.student_chat:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    chat_question = st.chat_input("Ask a follow-up question about this attempt...")
    if chat_question:
        st.session_state.student_chat.append({"role": "user", "content": chat_question})
        with st.chat_message("user"):
            st.markdown(chat_question)
        try:
            with st.chat_message("assistant"):
                with st.spinner("Gemma is preparing a focused answer..."):
                    reply = answer_student_follow_up(chat_question, level, data, st.session_state.student_chat)
                st.markdown(reply)
            st.session_state.student_chat.append({"role": "assistant", "content": reply})
            st.session_state.student_pdf = None
            if st.session_state.auto_save:
                _save_current(storage, level, question_text, answer_text)
        except Exception as error:
            st.error(f"{type(error).__name__}: {error}")
