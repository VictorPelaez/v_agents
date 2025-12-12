# deep agent
import os
from datetime import datetime, timezone
from rag_pipeline import retrieve_context, get_vector_index
from evaluation import print_subagent_tasks
from config import TAVILY_API_KEY, OPENAI_API_KEY
from tavily import TavilyClient  # langchain.tools import DuckDuckGoSearchRun
from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from deepagents.backends import FilesystemBackend

from prompts import (
    RESEARCH_INSTRUCTIONS,
    RETRIEVAL_INSTRUCTIONS,
    WRITE_INSTRUCTIONS,
    MAIN_SYSTEM_PROMPT,
)

# ============================
# ENVIRONMENT
# ============================
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


# ============================
# MAIN AGENT FUNCTION
# ============================
def agent_invoke(question, index=None):
    model_subagent = init_chat_model(
        model="gpt-4o",
        temperature=0.0,
        top_p=1.0,
        max_tokens=500
    )

    model_main = init_chat_model(
        model="gpt-5.1",
        temperature=0.0,
        top_p=1.0,
        max_tokens=500
    )

    # ------------------------------------
    # Web search tool
    # ------------------------------------
    @tool
    def internet_search_tool(query: str, max_results: int = 2,
                             topic: str = "finance",
                             include_raw_content: bool = False):
        """Web search tool using Tavily API"""
        return tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic
        )

    # ------------------------------------
    # Vector store retrieval tool
    # ------------------------------------
    @tool
    def retrieve_context_tool(query: str):
        """Vector store retrieval tool"""
        index = get_vector_index()
        serialized, retrieved_docs, ref_metadata = retrieve_context(query,
                                                                    index)

        if not retrieved_docs:
            return {
                "response": "Non data in vector store.",
                "docs": [],
                "metadata": ref_metadata,
                "found": False
            }

        return {
            "response": serialized,
            "docs": [
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "")
                } for doc in retrieved_docs
            ],
            "metadata": ref_metadata,
            "found": True
        }

    # ------------------------------------
    # File writer tool
    # ------------------------------------
    @tool
    def write_response_to_file(response: str,
                               filename: str = "response.md",
                               question: str | None = None):
        """
        Guarda la respuesta en /reports para persistencia (append).
        Añade un header con timestamp y la pregunta (si se proporciona).
        """
        try:
            # Root path
            reports_dir = os.path.join(os.path.dirname(__file__), "reports")
            os.makedirs(reports_dir, exist_ok=True)
            filepath = os.path.join(reports_dir, filename)
            # timestamp & header
            timestamp = datetime.now(timezone.utc).isoformat()
            qpart = f"**Question:** {question}\n\n" if question else ""
            header = "## " + timestamp + "\n\n" + qpart + "---\n\n"
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(header)
                f.write(response)
                f.write("\n\n")
            print(f"Response successfully appended to {filepath}")
            return f"Response successfully appended to {filepath}"
        except Exception as e:
            return f"Error saving file: {str(e)}"
    # ------------------------------------
    # Sub-agentes
    # ------------------------------------
    research_subagent = {
        "name": "research-agent",
        "description": ("""Conducts in-depth research using multiple web
                        searches to collect information.
                        Use when the answer cannot be resolved
                        from internal documents"""),
        "system_prompt": RESEARCH_INSTRUCTIONS,
        "tools": [internet_search_tool],
        "model": model_subagent
    }

    retrieval_subagent = {
        "name": "retrieval-agent",
        "description": ("""Extracts relevant information from the vector
                        database using semantic search.
                        Use when the required knowledge should come from
                        indexed internal documents"""),
        "system_prompt": RETRIEVAL_INSTRUCTIONS,
        "tools": [retrieve_context_tool],
        "model": model_subagent
    }

    file_writer_subagent = {
        "name": "file-writer",
        "description": ("""This subagent must always finalize the process by
                        appending the response to ‘response.md’, including
                        an ISO 8601 timestamp header"""),
        "system_prompt": WRITE_INSTRUCTIONS,
        "tools": [write_response_to_file],
        "model": model_subagent
    }
    # ------------------------------------
    # Deep Agent
    # ------------------------------------
    # reports_dir = os.path.join(os.path.dirname(__file__), "reports")
    agent = create_deep_agent(
        model=model_main,
        subagents=[retrieval_subagent,
                   research_subagent,
                   file_writer_subagent],
        system_prompt=MAIN_SYSTEM_PROMPT,
        # backend=FilesystemBackend(root_dir=reports_dir),
    )

    result = agent.invoke({
        "messages": [{"role": "user", "content": question}]
    })

    # ------------------------------------
    # Extraer respuesta
    # ------------------------------------
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
    print_subagent_tasks(result)
    return answer_text, refs


# ============================
# Test
# ============================
if __name__ == "__main__":
    resp, refs = agent_invoke(
        "¿Cuál sería la solvencia CET de BBVA en junio 2025?"
        # "¿Cuál sería el beneficio de BBVA en junio 2025?"
    )
    #print("\nRESPUESTA:\n", resp)
    #print("\nREFERENCIAS:\n", refs)
