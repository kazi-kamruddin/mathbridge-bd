import streamlit as st

from ..storage import workspace_ready


def render_history_view(storage) -> None:
    st.subheader("Persistent workspace")
    st.write(
        "Use the same workspace ID and PIN whenever you return. With Supabase configured, "
        "saved sessions survive app restarts and redeployments. Without Supabase, the app "
        "uses a temporary SQLite fallback that survives browser refreshes but is not guaranteed after redeployment."
    )
    c1, c2 = st.columns(2)
    with c1:
        workspace_id = st.text_input("Workspace ID", value=st.session_state.workspace_id, placeholder="example: kazi-math")
    with c2:
        pin = st.text_input("Workspace PIN", value=st.session_state.workspace_pin, type="password", placeholder="Use a memorable private PIN")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Connect workspace", type="primary", use_container_width=True):
            if workspace_id.strip() and pin.strip():
                st.session_state.workspace_id = workspace_id.strip()
                st.session_state.workspace_pin = pin.strip()
                st.session_state.workspace_connected = True
                st.success("Workspace connected.")
            else:
                st.warning("Enter both a workspace ID and PIN.")
    with c2:
        st.session_state.auto_save = st.toggle("Auto-save student sessions", value=st.session_state.auto_save)
    if not workspace_ready():
        st.info("Connect a workspace to save and restore conversations.")
        return
    sessions = storage.list(st.session_state.workspace_id, st.session_state.workspace_pin)
    if not sessions:
        st.info("No saved sessions yet.")
        return
    labels = {
        f"{item['title']} · {item['updated_at'][:16].replace('T', ' ')}": item["id"]
        for item in sessions
    }
    selected_label = st.selectbox("Saved sessions", list(labels))
    session_id = labels[selected_label]
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Load selected session", use_container_width=True):
            payload = storage.load(st.session_state.workspace_id, st.session_state.workspace_pin, session_id)
            if payload:
                st.session_state.student_analysis = payload.get("analysis")
                st.session_state.student_context = payload.get("analysis")
                st.session_state.student_chat = payload.get("chat", [])
                st.session_state.student_session_id = session_id
                st.session_state.student_level = payload.get("level", "SSC")
                st.session_state.student_question_text = payload.get("question_text", "")
                st.session_state.student_answer_text = payload.get("answer_text", "")
                st.success("Session loaded. Open the Student diagnosis tab.")
    with c2:
        if st.button("Delete selected session", use_container_width=True):
            storage.delete(st.session_state.workspace_id, st.session_state.workspace_pin, session_id)
            st.success("Session deleted.")
            st.rerun()
