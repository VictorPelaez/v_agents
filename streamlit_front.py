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


# =====================================================
# HELPERS
# =====================================================
def append_response_md(question: str, answer: str,
                       filename: str = "response.md") -> str:
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


def extract_last_assistant_message(result) -> str:
    """
    Robust extractor compatible with deepagents / tool messages
    """
    if not result:
        return ""

    messages = result.get("messages", [])
    for msg in reversed(messages):
        content = getattr(msg, "content", None)

        if isinstance(content, str) and content.strip():
            return content

        if isinstance(content, dict):
            if content.get("response"):
                return content.get("response")

    return ""


# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Assistant", layout="wide")


# =====================================================
# STYLES (UNCHANGED)
# =====================================================
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
}
div.stButton > button {
    background: #f8fafc;
    color: #374151;
    font-weight: 500;
    border-radius: 4px;
    padding: 4px 6px;
    font-size: 11px;
    line-height: 1.1;
    border: 1px solid #e5e7eb;
    box-shadow: none;
    min-height: unset;
}

div.stButton > button:hover {
    background: #eef2f7;
    border-color: #cbd5e1;
}
            
    section[data-testid="stVerticalBlock"] button {
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# TITLE
# =====================================================
st.markdown("""
<h1 style="text-align:left; color:#2c3e50; font-weight:900;">
   Company Analysis Assistant
</h1>
""", unsafe_allow_html=True)


# =====================================================
# LAYOUT
# =====================================================
chat_col, sidebar_col = st.columns([3, 1])


# =====================================================
# SIDEBAR (UNCHANGED)
# =====================================================
with sidebar_col:
    st.subheader("Upload PDFs / URL")
    uploaded_file = st.file_uploader("Upload a PDF:", type=["pdf"])
    pdf_url = st.text_input("Or URL with a PDF:")

    if "pdf_url_processed" not in st.session_state:
        st.session_state.pdf_url_processed = None
    if "uploaded_file_processed" not in st.session_state:
        st.session_state.uploaded_file_processed = None

    if pdf_url and pdf_url != st.session_state.pdf_url_processed:
        st.session_state.pdf_url_processed = pdf_url
        with st.spinner("Getting PDF from URL..."):
            st.session_state.vector_index = ingestion_workflow_pdf(pdf_url)
        st.success("PDF added successfully.")

    if uploaded_file and uploaded_file != st.session_state.uploaded_file_processed:
        st.session_state.uploaded_file_processed = uploaded_file
        with st.spinner("Processing uploaded PDF..."):
            st.session_state.vector_index = ingestion_workflow_pdf(uploaded_file)
        st.success("PDF local added successfully.")


# =====================================================
# CHAT
# =====================================================
with chat_col:

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = [
            {
                "role": "system",
                "content": "User name: Juan"
            }
        ]

    input_text = st.text_area(
        "Type your question here:",
        label_visibility="collapsed",
    )

    go_button = st.button("📌Send a question", type="primary")

    if go_button and input_text.strip():

        # USER → agent memory
        st.session_state.agent_messages.append({
            "role": "user",
            "content": input_text,
        })

        with st.spinner("Thinking..."):
            result = agent_invoke(st.session_state.agent_messages)

        answer = extract_last_assistant_message(result)

        # ASSISTANT → agent memory
        st.session_state.agent_messages.append({
            "role": "assistant",
            "content": answer,
        })

        # UI history (unchanged)
        st.session_state.messages.append({
            "user": input_text,
            "bot": answer,
            "refs": [],
        })

    # =================================================
    # RENDER CHAT (FULL, ORIGINAL BEHAVIOR)
    # =================================================
    for i in range(len(st.session_state.messages) - 1, -1, -1):
        message = st.session_state.messages[i]

        st.markdown(
            f'<div class="chat-bubble-user">{message["user"]}</div>',
            unsafe_allow_html=True,
        )

        cols = st.columns([5, 1])

        # ---------------- BOT MESSAGE ----------------
        with cols[0]:
            bot_text = message.get("bot", "") or ""

            edit_mode_key = f"edit_mode_{i}"
            edited_text_key = f"edited_bot_{i}"

            if st.session_state.get(edit_mode_key, False):
                initial = bot_text
                edited_val = st.text_area(
                    "Edit response:",
                    value=initial,
                    key=edited_text_key,
                    height=200,
                )
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
                + safe_bot +
                "</div>"
            )

            num_lines = safe_bot.count("<br>") + 1
            height = min(max(250, num_lines * 24), 1600)

            components.html(bot_html, height=height, scrolling=True)

        # ---------------- ACTIONS ----------------
        with cols[1]:
            st.markdown("**Actions**")
            status_key = f"status_{i}"

            if st.button("✔ Approve", key=f"approve_{i}"):
                edited_val = st.session_state.get(edited_text_key)
                final_bot = edited_val if edited_val is not None else bot_text
                filepath = append_response_md(message["user"], final_bot)
                st.session_state.messages[i]["bot"] = final_bot
                st.session_state[edit_mode_key] = False
                st.session_state[status_key] = f"approved ({os.path.basename(filepath)})"

            if st.button("✎ Edit", key=f"edit_{i}"):
                st.session_state[edit_mode_key] = True
                if edited_text_key not in st.session_state:
                    st.session_state[edited_text_key] = bot_text
                st.session_state[status_key] = "editing"

            if st.button("✖ Reject", key=f"reject_{i}"):
                st.session_state[status_key] = "rejected"

            if st.session_state.get(status_key):
                st.markdown(f"**Status:** {st.session_state[status_key]}")