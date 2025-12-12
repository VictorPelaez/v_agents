# v_agents

Lightweight deep-agent toolkit for research and retrieval workflows.

Features
- Create and run a main deep agent with subagents for retrieval, web research, and file writing.
- Saves final answers to `reports/response.md` with a UTC timestamp and the original question.
- Uses Tavily for web search and a local vector index for retrieval.

Getting started

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

How responses are saved
- The file-writer subagent and the explicit end-of-run save both append to `reports/response.md`.
- Each appended entry begins with a Markdown header containing a UTC timestamp and, if available, the question.

