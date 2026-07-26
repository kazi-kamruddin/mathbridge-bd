from typing import Any

import streamlit as st

DEFAULT_STATE: dict[str, Any] = {
    "student_analysis": None,
    "student_context": None,
    "student_chat": [],
    "student_pdf": None,
    "student_pdf_error": None,
    "student_session_id": None,
    "teacher_results": None,
    "teacher_summary": None,
    "teacher_pdf": None,
    "teacher_pdf_error": None,
    "workspace_id": "",
    "workspace_pin": "",
    "workspace_connected": False,
    "auto_save": True,
}


def initialize_state() -> None:
    for key, value in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = value.copy() if isinstance(value, list) else value


def reset_student_state() -> None:
    st.session_state.student_analysis = None
    st.session_state.student_context = None
    st.session_state.student_chat = []
    st.session_state.student_pdf = None
    st.session_state.student_pdf_error = None
    st.session_state.student_session_id = None


def reset_teacher_state() -> None:
    st.session_state.teacher_results = None
    st.session_state.teacher_summary = None
    st.session_state.teacher_pdf = None
    st.session_state.teacher_pdf_error = None
