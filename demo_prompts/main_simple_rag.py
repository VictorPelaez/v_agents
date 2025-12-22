# .\myenv\Scripts\activate
import os
import time
import yaml
import json
from pathlib import Path
from jinja2 import Template
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from config import OPENAI_API_KEY

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def load_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def render_prompt(template_str, variables):
    return Template(template_str).render(**variables)


def load_all_field_paths(dir="fields"):
    return sorted(
        Path(dir).glob("*.yaml"),
        key=lambda p: int(p.stem.split("_")[0])
    )


def retrieval_with_answer(question, field_prompt):
    """
    Retrieve the most relevant documents and response
    """
    embeddings = BedrockEmbeddings(
        credentials_profile_name='default',
        model_id='amazon.titan-embed-text-v2:0')

    vector_store = FAISS.load_local(
        "vector_index",  # load folder FAISS
        embeddings,      # embeddings
        allow_dangerous_deserialization=True
        )

    retrieved_docs = vector_store.similarity_search(question, k=5)

    chunks = "\n\n".join(
        (
            f"Source: {doc.metadata.get('source', 'unknown')}, "
            f"Page: {doc.metadata.get('page', 'unknown')}\n"
            f"Content: {doc.page_content}"
        )
        for doc in retrieved_docs
    )

    global_cfg = load_yaml("configs/global.yaml")
    output_contract = load_yaml(
        "configs/output_contract.yaml")["output_contract"]
    extract_prompt_cfg = load_yaml("prompts/extract.yaml")

    prompt = render_prompt(
        extract_prompt_cfg["template"], {
            **global_cfg,
            **field_prompt,
            "output_contract": output_contract,
            "chunks": chunks
            })

    model = init_chat_model(model="gpt-4.1-mini", temperature=0.0,
                            top_p=1.0, max_tokens=3000)
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": question},
        {"role": "assistant", "content": chunks}
        ]
    response = model.invoke(messages)

    return response


# =====================================================
# Main code
# =====================================================
if __name__ == "__main__":

    company = "BBVA"
    paths = load_all_field_paths()

    results = []
    for path in paths:
        # time
        start_time = time.perf_counter()
        field_cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        template = PromptTemplate.from_template(field_cfg["task"])
        question = template.format(company=company)
        output = retrieval_with_answer(question, field_cfg)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time  # seconds

        results.append({
            "path": str(path),
            "question": question,
            "response": output,
            "time_seconds": elapsed_time,
            "tokens_used": output.usage_metadata['input_tokens']
        })

clean_results = []

for item in results:
    raw_text = item["response"].content
    clean_text = raw_text.strip()

    if clean_text.startswith("```"):
        clean_text = clean_text.split("```")[1]
        if clean_text.startswith("json"):
            clean_text = clean_text[4:]

    data = json.loads(clean_text)

    clean_results.append({
        "path": item["path"],
        "question": item["question"],
        "response": data,
        "time_seconds": item["time_seconds"],
        "tokens_used": item["tokens_used"]
    })

filename = f"data_{company.lower()}.json"
with open(filename, "w", encoding="utf-8") as f:
    json.dump(clean_results, f, ensure_ascii=False, indent=4)
print("---- Saved in data.json ----")
