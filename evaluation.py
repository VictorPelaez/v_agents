# https://docs.langchain.com/langsmith/evaluate-rag-tutorial

def print_subagent_tasks(result):
    """Imprime las tareas realizadas por los sub-agentes en una invocación del agente."""
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