# deep agent
import os
import json

from tavily import TavilyClient
from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

from rag_pipeline import (retrieve_context, get_vector_index,
                          query_reports_md)
from evaluation import print_subagent_tasks
from config import TAVILY_API_KEY, OPENAI_API_KEY
from prompts import (
    RESEARCH_INSTRUCTIONS,
    RETRIEVAL_INSTRUCTIONS,
    MAIN_SYSTEM_PROMPT,
)

# =====================================================
# ENVIRONMENT
# =====================================================
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
_agent_instance = None


# =====================================================
# MAIN AGENT FUNCTION
# =====================================================
def get_agent():
    """Get deep agent."""

    global _agent_instance
    if _agent_instance is not None:
        return _agent_instance

    # -------------------------------
    # Models
    # -------------------------------
    model_subagent = init_chat_model(
        model="gpt-4o", temperature=0.0, top_p=1.0, max_tokens=500
    )
    model_main = init_chat_model(
        model="gpt-5.1", temperature=0.0, top_p=1.0, max_tokens=500
    )

    # -------------------------------
    # Tools
    # -------------------------------
    @tool
    def local_cache_tool(query: str):
        """Tool wrapper for semantic search in response.md"""
        return query_reports_md(query)

    @tool
    def internet_search_tool(
        query: str,
        max_results: int = 2,
        topic: str = "finance",
        include_raw_content: bool = False,
    ):
        """Web search tool using Tavily API."""
        return tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )

    @tool
    def retrieve_context_tool(query: str):
        """Vector store retrieval tool."""
        vect_index = get_vector_index()
        serialized, retrieved_docs, ref_metadata = retrieve_context(
            query, vect_index
        )

        if not retrieved_docs:
            return {
                "response": "No data found in vector store.",
                "docs": [],
                "metadata": ref_metadata,
                "found": False,
            }

        return {
            "response": serialized,
            "docs": [
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", ""),
                }
                for doc in retrieved_docs
            ],
            "metadata": ref_metadata,
            "found": True,
        }

    # -------------------------------
    # Sub-agents
    # -------------------------------
    local_cache_subagent = {
        "name": "local-cache-agent",
        "description": ("""First checks response.md using semantic search
                        embeddings before other subagents."""),
        "system_prompt": ("""You are a local cache agent that
                          responds only using response.md file content."""),
        "tools": [local_cache_tool],
        "model": model_subagent,
    }
    research_subagent = {
        "name": "research-agent",
        "description": (
            "Conducts deep web research when information is not "
            "available locally."
        ),
        "system_prompt": RESEARCH_INSTRUCTIONS,
        "tools": [internet_search_tool],
        "model": model_subagent,
    }

    retrieval_subagent = {
        "name": "retrieval-agent",
        "description": (
            "Retrieves knowledge from the internal vector store "
            "using semantic search."
        ),
        "system_prompt": RETRIEVAL_INSTRUCTIONS,
        "tools": [retrieve_context_tool],
        "model": model_subagent,
    }

    # -------------------------------
    # Deep Agent
    # -------------------------------
    # memories_dir = os.path.join(os.path.dirname(__file__), "memories")
    agent = create_deep_agent(
        model=model_main,
        subagents=[
            local_cache_subagent,
            retrieval_subagent,
            research_subagent
        ],
        system_prompt=MAIN_SYSTEM_PROMPT,
        # backend=FilesystemBackend(root_dir=memories_dir),
    )
    _agent_instance = agent
    return agent


# -------------------------------
# Agent Invocation
# -------------------------------
def agent_invoke(messages: list):
    """
    messages = [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
    ]
    """

    agent = get_agent()

    result = agent.invoke({"messages": messages})

    print_subagent_tasks(result)
    return result

# =====================================================
# Manual Test
# =====================================================
if __name__ == "__main__":
    output = agent_invoke(
        "¿Cuál sería la solvencia CET1 de BBVA en junio 2025?"
    )