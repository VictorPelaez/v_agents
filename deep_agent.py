# deep agent
import os
from rag_pipeline import retrieve_context, get_vector_index
from config import TAVILY_API_KEY, OPENAI_API_KEY
from tavily import TavilyClient # langchain.tools import DuckDuckGoSearchRun
from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool


os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def agent_invoke(question, index=None):
    model_simple = init_chat_model("gpt-4o", temperature=0.0, max_tokens=500)
    model_advanced = init_chat_model("gpt-5.1", temperature=0.0, max_tokens=500)

    # Web search tool
    @tool
    def internet_search_tool(query: str, max_results: int = 2,
                            topic: str = "finance",
                            include_raw_content: bool = False):
        """Realiza una búsqueda en internet para obtener datos financieros."""
        return tavily_client.search(query, max_results=max_results,
                                    include_raw_content=include_raw_content,
                                    topic=topic)

    # Retrieval tool
    @tool
    def retrieve_context_tool(query: str):
        """Recupera información del vector store (estados financieros)."""
        index = get_vector_index()
        serialized, retrieved_docs, ref_metadata = retrieve_context(query, index)
        if not retrieved_docs:
            return {"response": "No hay datos en el vector store.", "docs": [], "metadata": ref_metadata, "found": False} 
        return {"response": serialized,
                "docs": [{"content": doc.page_content, 
                          "source": doc.metadata.get("source", "")} for doc in retrieved_docs],
                "metadata": ref_metadata,
                "found": True}

    # Prompts
    research_instructions = (
        "Eres un analista experto en finanzas. "
        "Tu tarea es investigar en internet el BPA (beneficio por acción) de una empresa, "
        "o en su defecto, su número de acciones y beneficio neto. "
        "Usa herramientas de búsqueda online para obtener información precisa y actualizada. "
        "Luego redacta un informe claro y conciso con los datos encontrados junto con las fuentes consultadas. "
        "No preguntar si necesitas más información, simplemente realiza la búsqueda y proporciona los resultados."
    )

    retrieval_instructions = (
        "Actúas como un sistema RAG ESTRICTO de información financiera.\n"
        "REGLAS OBLIGATORIAS:\n"
        "1. Solo puedes responder usando EXACTAMENTE la información que venga en los documentos recuperados.\n"
        "2. Si los documentos pertenecen a una empresa distinta a la preguntada (por ejemplo BBVA vs Santander), "
        "debes responder: 'No hay datos en el vector store para esa empresa'.\n"
        "3. Está prohibido deducir, inferir o completar con conocimiento del modelo.\n"
        "4. Si falta información, debes decirlo literalmente.\n"
        "5. No puedes mezclar datos de diferentes bancos.\n"
        "6. Nunca uses datos de internet; eso es tarea del research-agent.\n"
    )

    # ----------------------------
    # Sub-agentes
    # ----------------------------
    research_subagent = {
        "name": "research-agent",
        "description": "Busca información financiera en internet.",
        "system_prompt": research_instructions,
        "tools": [internet_search_tool],
        "model": model_simple
    }

    retrieval_subagent = {
        "name": "retrieval-agent",
        "description": "Recupera datos del vector store.",
        "system_prompt": retrieval_instructions,
        "tools": [retrieve_context_tool],
        "model": model_simple
    }

    # ----------------------------
    # Crear Deep Agent
    # ----------------------------
    agent = create_deep_agent(
        model=model_advanced,  # 
        subagents=[retrieval_subagent, research_subagent],
        system_prompt=(
            "Responde de froma concisa a las preguntas financieras, "
            "delegando las tareas a los subagentes según corresponda. "
            "Primero intenta obtener la información con retrieval-agent "
            "usando los datos internos. Primero verifica el retrieval-agent. "
            "Si devuelve datos, ÚNICAMENTE usa eso y no llames a ningún otro subagente. "
            "Nunca inventes datos ni uses internet si hay información interna."
            "Si retrieval-agent no devuelve suficiente información "
            "(o devuelve vacío), "
            "entonces delega la tarea a research-agent para buscar en internet"
            "No termines pidiendo más información al usuario.  "
            "No pongas simbolos markdown, ni iconos en tus respuestas. "
            "Simplemente delega y responde con las fuentes consultadas." 
            "Termina siempre la respuesta con - Fuente: fuente1, fuente2 "
            "y los enlaces a los documentos o url"
        ))

    result = agent.invoke({
        "messages": [{"role": "user", "content": question}]
        })

    # Extraer respuesta y referencias
    answer_text = ""
    refs = []
    print("\n\n", len(result["messages"]))
    print(question)
    for m in result["messages"]:
        content = getattr(m, "content", "")
        if isinstance(content, dict) and "docs" in content:
            refs.extend(content["docs"])
            if content.get("response"):
                answer_text = content["response"]
        elif "AIMessage" in str(type(m)) and content.strip():
            answer_text = content

    # print subagent details
    for m in result["messages"]:
        # Identificar ToolMessage
        if "ToolMessage" in str(type(m)):
            print("Contenido completo:", m.content[:200], "...")  # preview
            # Buscar el tool_call asociado
            for t in result["messages"]:
                if hasattr(t, "tool_calls"):
                    for call in t.tool_calls:
                        if call["id"] == m.tool_call_id:
                            print("Sub-agente:", call["args"].get("subagent_type"))
                            print("Descripción de la task:", call["args"].get("description"))
                            print("\n")  
    return answer_text, refs
