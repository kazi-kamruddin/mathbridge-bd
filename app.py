import streamlit as st

from mathbridge.config import APP_CAPTION, APP_NAME
from mathbridge.state import initialize_state
from mathbridge.storage import get_storage
from mathbridge.views.history_view import render_history_view
from mathbridge.views.student_view import render_student_view
from mathbridge.views.teacher_view import render_teacher_view

st.set_page_config(page_title=APP_NAME, page_icon="🧮", layout="wide")
initialize_state()

st.title(f"🧮 {APP_NAME}")
st.caption(APP_CAPTION)

storage = get_storage()

student_tab, teacher_tab, history_tab = st.tabs(
    ["🎓 Student diagnosis", "👩‍🏫 Teacher evaluation", "🗂️ Saved sessions"]
)

with student_tab:
    render_student_view(storage)

with teacher_tab:
    render_teacher_view(storage)

with history_tab:
    render_history_view(storage)
