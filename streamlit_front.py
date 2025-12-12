# genai_aws/front.py
# .\myenv\Scripts\activate
# streamlit run streamlit_front.py
import streamlit as st
from rag_pipeline import ingestion_workflow_pdf
from deep_agent import agent_invoke
import os
from datetime import datetime, timezone
import html
import streamlit.components.v1 as components


def append_response_md(question: str, answer: str,
                       filename: str = "response.md") -> str:
    """Append approved Q/A to reports/response.md with timestamp header."""
    reports_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    filepath = os.path.join(reports_dir, filename)
    timestamp = datetime.now(timezone.utc).isoformat()
    qpart = f"**Question:** {question}\n\n" if question else ""
    header = f"## {timestamp}\n\n{qpart}---\n\n"
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(header)
        f.write(answer)
        f.write("\n\n")
    return filepath


def extract_answer_from_result(result) -> str:
    """Try multiple locations in the agent result to find a textual answer."""
    if not result:
        return ""
    # Common key used in deep_agent: 'answer'
    if isinstance(result, dict):
        answer = result.get("answer")
        if answer and isinstance(answer, str) and answer.strip():
            return answer
    # Some agents return messages list with last AI message
    msgs = result.get("messages") if isinstance(result, dict) else None
    if msgs and isinstance(msgs, list):
        for m in reversed(msgs):
            content = getattr(m, "content", None)
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, dict) and content.get("response"):
                return content.get("response")
    # Fallback: try top-level 'message' or 'text'
    if isinstance(result, dict):
        for key in ("message", "text", "answer_text"):
            if result.get(key):
                return result.get(key)
    return ""


# -----------------------------
# CONFIGURACIÓN DE PÁGINA
# -----------------------------
st.set_page_config(page_title="Assistant", layout="wide")

# -----------------------------
# ESTILOS PERSONALIZADOS
# -----------------------------
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #e0f7fa 0%, #fce4ec 100%);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.chat-bubble-user {
    background: #4a90e2;
    color: white;
    padding: 14px;
    margin: 8px 0;
    border-radius: 18px;
    max-width: 75%;
    align-self: flex-end;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.15);
}
.chat-bubble-bot {
    background: #ffffff;
    color: #333;
    padding: 14px;
    margin: 8px 0;
    border-radius: 18px;
    max-width: 75%;
    border-left: 5px solid #4a90e2;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
}
.stTextArea textarea {
    background: #ffffff;
    color: #111111;
    border-radius: 10px;
    padding: 12px;
    font-size: 16px;
    border: 2px solid #3b82f6;
    transition: border 0.3s ease;
}
div.stButton > button {
    /* Softer, formal button style */
    background: linear-gradient(180deg, #f7f7f8 0%, #eef0f2 100%);
    color: #1f2937; /* dark gray for formal text */
    font-weight: 600;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 12px;
    border: 1px solid #d1d5db;
    box-shadow: none;
    transition: transform 0.12s ease, box-shadow 0.12s ease;
}
div.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 1px 3px rgba(16, 24, 40, 0.06);
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# TÍTULO
# -----------------------------
st.markdown("""
<h1 style="text-align:left; color:#2c3e50; font-weight:900; font-family:'Segoe UI', sans-serif;">
   Company Analysis Assistant
</h1>
""", unsafe_allow_html=True)

# -----------------------------
# LAYOUT
# -----------------------------
chat_col, sidebar_col = st.columns([3, 1])

# -----------------------------
# BARRA LATERAL
# -----------------------------
with sidebar_col:
    st.subheader("Upload PDFs / URL")
    uploaded_file = st.file_uploader("Upload a PDF:", type=["pdf"])
    pdf_url = st.text_input("Or URL with a PDF:")

    # Inicializar flags en session_state
    if "pdf_url_processed" not in st.session_state:
        st.session_state.pdf_url_processed = None
    if "uploaded_file_processed" not in st.session_state:
        st.session_state.uploaded_file_processed = None

    vector_index = st.session_state.get("vector_index", None)

    # Procesar URL solo si cambió y no se ha procesado
    if pdf_url and pdf_url != st.session_state.pdf_url_processed:
        st.session_state.pdf_url_processed = pdf_url  # marcar como procesada
        with st.spinner("Getting PDF from URL..."):
            vector_index = ingestion_workflow_pdf(pdf_url)
            st.session_state.vector_index = vector_index
        st.success("PDF added successfully.")

    # Procesar archivo solo si cambió y no se ha procesado
    if uploaded_file and uploaded_file != st.session_state.uploaded_file_processed:
        st.session_state.uploaded_file_processed = uploaded_file
        with st.spinner("Processing uploaded PDF..."):
            vector_index = ingestion_workflow_pdf(uploaded_file)
            st.session_state.vector_index = vector_index
        st.success("PDF local added successfully.")

# -----------------------------
# CHAT PRINCIPAL
# -----------------------------
with chat_col:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Prefill input if a follow-up was requested from a message
    if "input_text_prefill" in st.session_state:
        prefill = st.session_state.pop("input_text_prefill")
    else:
        prefill = ""

    input_text = st.text_area(
        "Type your question here:",
        value=prefill,
        label_visibility="collapsed",
    )
    go_button = st.button("📌Send a question", type="primary")

    if (
        go_button
        and input_text.strip() != ""
    ):
        with st.spinner("Thinking..."):
            result = agent_invoke(input_text)

        answer = extract_answer_from_result(result)
        refs = result.get("refs", []) if isinstance(result, dict) else []

        st.session_state.messages.append({
            "user": input_text,
            "bot": answer,
            "refs": refs,
        })

    for i in range(len(st.session_state.messages) - 1, -1, -1):
        message = st.session_state.messages[i]
        # User message
        st.markdown(
            f'<div class="chat-bubble-user">{message["user"]}</div>',
            unsafe_allow_html=True,
        )

        # Bot message + compact action block to the right
        cols = st.columns([5, 1])
        with cols[0]:
            # Force inline style to guarantee white block rendering
            # Coerce to string, escape HTML and preserve newlines.
            bot_text = message.get("bot", "")
            bot_text = "" if bot_text is None else bot_text
            # Provide inline edit area when user toggles edit for this message
            edit_mode_key = f"edit_mode_{i}"
            edited_text_key = f"edited_bot_{i}"

            if st.session_state.get(edit_mode_key, False):
                initial = message.get("bot", "") or ""
                edited_val = st.text_area("Edit response:", value=initial, key=edited_text_key, height=200)
                safe_bot = html.escape(str(edited_val)).replace("\n", "<br>")
            else:
                safe_bot = html.escape(str(bot_text)).replace("\n", "<br>")

            if not safe_bot.strip():
                safe_bot = "<i>(no textual response)</i>"

            bot_html = (
                "<div style='background:#ffffff; color:#333; padding:14px; "
                "margin:8px 0; border-radius:18px; max-width:100%; "
                "border-left:5px solid #4a90e2; "
                "box-shadow:2px 2px 10px rgba(0,0,0,0.1);'>"
                + safe_bot
                + "</div>"
            )

            num_lines = safe_bot.count("<br>") + 1
            height = min(max(250, num_lines * 24), 1600)
            components.html(bot_html, height=height, scrolling=True)

            if message.get("refs"):
                st.markdown(
                    "<div style='margin-top:4px; font-weight:bold;'>"
                    "Sources used:</div>",
                    unsafe_allow_html=True,
                )
                for r in message["refs"]:
                    src = r.get("source", "")
                    page = r.get("page", "N/A")
                    st.markdown(
                        (
                            f"<div style='margin-left:12px;'>- `{src}` — "
                            f"page <b>{page}</b></div>"
                        ),
                        unsafe_allow_html=True,
                    )

        # Compact action buttons (emoji-only to reduce visual size)
        with cols[1]:
            st.markdown("**Actions**")
            status_key = f"status_{i}"

            # Approve: write either edited text (if present) or original
            if st.button("✔ Approve", key=f"approve_{i}"):
                edited_val = st.session_state.get(f"edited_bot_{i}", None)
                final_bot = edited_val if edited_val is not None else message.get("bot", "")
                filepath = append_response_md(message["user"], final_bot)
                try:
                    st.session_state["messages"][i]["bot"] = final_bot
                except Exception:
                    pass
                st.session_state[f"edit_mode_{i}"] = False
                st.session_state[status_key] = (
                    f"approved ({os.path.basename(filepath)})"
                )

            if st.button("✎ Edit", key=f"edit_{i}"):
                st.session_state[f"edit_mode_{i}"] = True
                if f"edited_bot_{i}" not in st.session_state:
                    st.session_state[f"edited_bot_{i}"] = message.get("bot", "")
                st.session_state[status_key] = "editing"

            if st.button("✖ Reject", key=f"reject_{i}"):
                # Mark as rejected and do not save
                st.session_state[status_key] = "rejected"

            # Show current status
            cur_status = st.session_state.get(status_key)
            if cur_status:
                st.markdown(f"**Status:** {cur_status}")
