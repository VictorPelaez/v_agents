#!/usr/bin/env python3

import os
from v_agents.config import OPENAI_API_KEY
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain.chat_models import init_chat_model
import time
from langchain_community.vectorstores.faiss import FAISS

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def ingestion_workflow_pdf(doc_url):
    """
    Load a PDF, split into chunks preserving metadata (source & page), 
    create embeddings and store/update FAISS vector index.
    """
    start = time.time()
    # 1. Define and load data with PDFLoader
    loader = PyPDFLoader(doc_url)
    docs_loader = loader.load()

    # 2. Split the Text based on Character - Tokens Recursively split
    data_split = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=500, 
        chunk_overlap=50)
    docs_chunks = data_split.split_documents(docs_loader)
    print(f"Split into {len(docs_chunks)} sub-documents.")

    # 3. Create Embeddings and index
    embeddings = BedrockEmbeddings(
        credentials_profile_name='default',
        model_id='amazon.titan-embed-text-v2:0')

    # 4. Create Vector Store
    if os.path.exists("vector_index"):
        vector_store = FAISS.load_local(
            "vector_index",  # load folder FAISS
            embeddings,      # embeddings
            allow_dangerous_deserialization=True
            )
        vector_store.add_documents(docs_chunks)
        # Guardar índice actualizado
        vector_store.save_local("vector_index")
    else:
        vector_store = FAISS.from_documents(docs_chunks, embeddings)
        vector_store.save_local("vector_index")

    end = time.time()
    print(end - start)
    return vector_store


def get_vector_index():
    if os.path.exists("vector_index"):
        embeddings = BedrockEmbeddings(
            credentials_profile_name='default',
            model_id='amazon.titan-embed-text-v2:0')
        vector_store = FAISS.load_local(
            "vector_index",  # load folder FAISS
            embeddings,      # embeddings
            allow_dangerous_deserialization=True
            )
    return vector_store


def retrieve_context(question, vector_store):
    """
    Retrieve the most relevant documents and return:
    - serialized text for the model
    - retrieved doc objects
    - metadata (source, page)
    """
    retrieved_docs = vector_store.similarity_search(question, k=1)
    # Serialized context for the LLM
    serialized = "\n\n".join(
        (
            f"Source: {doc.metadata.get('source', 'unknown')}, "
            f"Page: {doc.metadata.get('page', 'unknown')}\n"
            f"Content: {doc.page_content}"
        )
        for doc in retrieved_docs
    )
    # Reference metadata for UI
    ref_metadata = [
        {
            "source": doc.metadata.get("source"),
            "page": doc.metadata.get("page")
        }
        for doc in retrieved_docs
    ]
    return serialized, retrieved_docs, ref_metadata


def rag_response(index, question):
    model = init_chat_model("gpt-4o-mini", temperature=0.7, max_tokens=500)
    serialized, retrieved_docs, ref_metadata = retrieve_context(question, index)
    # print(serialized)
    prompt = (
        "Eres un analista financiero experto."
        "Responde con precisión usando únicamente la información proporcionada, "
        "los estados financieros, cuentas de resultados e informes cargados"
        "Explica cifras, variaciones y métricas como ingresos, EBITDA, "
        "margen, cashflow o deuda de manera clara y concisa, etc"
        "Si falta información, dilo explícitamente y no inventes datos.")
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": question},
        {"role": "assistant", "content": serialized}
        ]
    response = model.invoke(messages)  # langchain_core.messages.ai.AIMessage'
    # Return BOTH: model answer + reference metadata
    return response, ref_metadata


def list_sources_from_vector_index(index_path="vector_index"):
    """
    Load a FAISS vector index and return unique sources of the documents
    along with page information if available.

    Args:
        index_path (str): Path to the FAISS index folder.

    Returns:
        sources_info (list of dict): [{'source': ..., 'page': ...}, ...]
    """
    embeddings = BedrockEmbeddings(
        credentials_profile_name='default',
        model_id='amazon.titan-embed-text-v2:0')
    # Load the FAISS vector store
    vector_store = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True)
    docs = list(vector_store.docstore._dict.values())

    # Collect unique sources
    unique_sources = set()
    for doc in docs:
        source = doc.metadata.get("source")
        if source:
            unique_sources.add(source)

    return list(unique_sources)