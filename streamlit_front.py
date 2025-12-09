# genai_aws/front.py
# .\myenv\Scripts\activate
# streamlit run streamlit_front.py
import streamlit as st
from rag_pipeline import ingestion_workflow_pdf, rag_response, get_vector_index, list_sources_from_vector_index
from deep_agent import agent_invoke

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
    background: linear-gradient(90deg, #4a90e2 0%, #50e3c2 100%);
    color: white;
    font-weight: bold;
    border-radius: 10px;
    padding: 10px 20px;
    border: none;
    transition: all 0.3s ease;
}
div.stButton > button:hover {
    transform: scale(1.05);
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# TÍTULO A LA IZQUIERDA
# -----------------------------
st.markdown("""
<h1 style="text-align:left; color:#2c3e50; font-weight:900; font-family:'Segoe UI', sans-serif;">
   Company Analysis Assistant
</h1>
""", unsafe_allow_html=True)

# -----------------------------
# LAYOUT: Chat + Barra lateral
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

    # LIST EXISTING SOURCES IN INDEX
    st.subheader("Documents in Index")
    if "vector_index" in st.session_state:
        try:
            sources = list_sources_from_vector_index("vector_index")
            if sources:
                for src in sources:
                    st.markdown(f"- **Source:** `{src}`")
            else:
                st.info("No documents indexed yet.")
        except Exception as e:
            st.error(f"Error loading index: {e}")
    else:
        st.info("Index not loaded yet.")

# -----------------------------
# CHAT PRINCIPAL
# -----------------------------
with chat_col:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if 'vector_index' not in st.session_state:
        with st.spinner("Wait for magic..."):
            st.session_state.vector_index = get_vector_index()

    input_text = st.text_area("Type your question here:",
                              label_visibility="collapsed")
    go_button = st.button("📌Send a question", type="primary")

    if go_button and input_text.strip() != "":
        with st.spinner("Thinking..."):
            answer, refs = agent_invoke(input_text)
            st.session_state.messages.append({
                "user": input_text,
                "bot": answer,
                "refs": refs
            })

    for message in reversed(st.session_state.messages):
        # Mensaje del usuario
        st.markdown(f'<div class="chat-bubble-user">{message["user"]}</div>', unsafe_allow_html=True)
        # Mensaje del bot
        st.markdown(f'<div class="chat-bubble-bot">{message["bot"]}</div>', unsafe_allow_html=True)
        # Referencias
        if message.get("refs"):
            st.markdown("<div style='margin-top:4px; font-weight:bold;'>Sources used:</div>", unsafe_allow_html=True)
            for r in message["refs"]:
                st.markdown(f"<div style='margin-left:12px;'>- `{r['source']}` — page <b>{r.get('page','N/A')}</b></div>", unsafe_allow_html=True)
