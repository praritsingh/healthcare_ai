"""
rag_pipeline.py
----------------
Retrieval-Augmented Generation over a small local medical knowledge base
(MedlinePlus-style snippets). This is the "4. RAG Pipeline" +
"5. Insight Synthesis" stages.

Stack: sentence-transformers (embeddings) -> ChromaDB (vector store) ->
Flan-T5 (grounded generation).
"""

import os
import json
import argparse

import chromadb
from chromadb.utils import embedding_functions
from transformers import pipeline as hf_pipeline

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
KB_PATH = os.path.join(DATA_DIR, "medlineplus_snippets.json")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_store")
COLLECTION_NAME = "medical_kb"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GENERATION_MODEL = "google/flan-t5-base"


def load_knowledge_base(path: str = KB_PATH):
    with open(path, "r") as f:
        return json.load(f)


def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=embed_fn
    )
    return collection


def build_index():
    kb = load_knowledge_base()
    collection = get_chroma_collection()

    collection.upsert(
        ids=[item["id"] for item in kb],
        documents=[item["text"] for item in kb],
        metadatas=[{"topic": item["topic"]} for item in kb],
    )
    print(f"Indexed {len(kb)} knowledge base entries into ChromaDB at {CHROMA_DIR}")


def retrieve(query: str, k: int = 3):
    collection = get_chroma_collection()
    results = collection.query(query_texts=[query], n_results=k)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    return list(zip(docs, metas))


class InsightSynthesizer:
    """Wraps Flan-T5 to generate a grounded, plain-language summary."""

    def __init__(self, model_name: str = GENERATION_MODEL):
        self.generator = hf_pipeline("text2text-generation", model=model_name)

    def synthesize(self, patient_summary: str, risk_factors: list, retrieved_context: list) -> str:
        context_text = "\n".join(f"- {doc}" for doc, _ in retrieved_context)
        factors_text = ", ".join(f"{f['feature']} ({f['impact']:+.2f})" for f in risk_factors)

        prompt = (
            "You are a clinical assistant. Using ONLY the context below, write a short, "
            "plain-language summary (3-4 sentences) explaining this patient's health risk "
            "and one or two practical next steps. Do not invent facts not in the context.\n\n"
            f"Patient notes: {patient_summary}\n"
            f"Top risk factors from the model: {factors_text}\n\n"
            f"Relevant medical context:\n{context_text}\n\n"
            "Summary:"
        )
        output = self.generator(prompt, max_new_tokens=180, do_sample=False)
        return output[0]["generated_text"]


def run_full_rag_demo(patient_note: str, risk_factors: list):
    retrieved = retrieve(patient_note, k=3)
    synthesizer = InsightSynthesizer()
    summary = synthesizer.synthesize(patient_note, risk_factors, retrieved)
    return {
        "retrieved_context": [{"text": d, "topic": m["topic"]} for d, m in retrieved],
        "summary": summary,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-index", action="store_true", help="Build the ChromaDB index from the local KB")
    parser.add_argument("--query", type=str, default=None, help="Run a sample retrieval query")
    args = parser.parse_args()

    if args.build_index:
        build_index()

    if args.query:
        results = retrieve(args.query)
        for doc, meta in results:
            print(f"[{meta['topic']}] {doc[:120]}...")
