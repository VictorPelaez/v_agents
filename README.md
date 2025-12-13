# v_agents

Lightweight toolkit for building deep-agent workflows for research and retrieval.
It allows a main agent to coordinate specialized subagents while keeping strict control over data sources.

## Features
- Create and run a main deep agent with subagents for retrieval and web research
- Main deep agent with specialized subagents (local cache, vector retrieval, web research)
- Source selection follows a fixed hierarchy: local cache → vector store → web search
- Semantic search over approved answers stored in `reports/response.md`
- Vector-based retrieval from local document indexes
- Web search using Tavily
- Multi-turn conversation support
- Approved answers are saved with UTC timestamp and original question
- Easy integration with a Streamlit chat frontend

## Getting started

Prerequisites
- Python 3.10+ (project uses virtualenv)
- See `requirements.txt` for Python dependencies

Quick setup
1. Create and activate a virtual environment:

```powershell
python -m venv myenv
myenv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

Configuration
- Copy or edit `config.py` to provide `TAVILY_API_KEY` and `OPENAI_API_KEY`.

Running the agent
- Run the main script to invoke a test run:

```powershell
python .\deep_agent.py
```

- To call the agent programmatically, import and use `agent_invoke(question)` from `deep_agent.py`.

Files of interest
- `deep_agent.py`: Main agent orchestration and subagents (retrieval, research, file writer).
- `rag_pipeline.py`: Helpers for vector index retrieval.
- `reports/response.md`: Appended responses; each entry includes a timestamp and the question.
- `requirements.txt`: Dependency list.

