# Healthcare Risk Prediction & Insights System (Advanced)

An end-to-end healthcare AI system that combines **structured risk prediction**,
**model explainability**, **biomedical NLP**, and **Retrieval-Augmented Generation (RAG)**
to turn raw patient data into explainable, actionable health insights.

## Architecture

```
                    ┌─────────────────────┐
   Patient Data ───►│  1. Risk Model       │──► Risk Score (0-1)
   (structured)      │  XGBoost Classifier  │
                    └─────────┬───────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  2. Explainability   │──► Top risk factors
                    │  SHAP values         │    (per patient)
                    └─────────┬───────────┘
                              │
                              ▼
   Clinical Notes ──►┌─────────────────────┐
   (unstructured)     │  3. Biomedical NER   │──► Extracted entities
                    │  scispaCy            │    (conditions, drugs)
                    └─────────┬───────────┘
                              │
                              ▼
   Medical Corpus ──►┌─────────────────────┐
   (MedlinePlus etc)  │  4. RAG Pipeline     │──► Retrieved context
                    │  Embeddings+ChromaDB │
                    └─────────┬───────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  5. Insight Synthesis│──► Plain-language,
                    │  Flan-T5             │    explainable summary
                    └─────────┬───────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  6. Gradio UI        │
                    └─────────────────────┘
```

## Why this design is "advanced"

- **Not just a classifier**: SHAP explainability turns a black-box risk score into
  auditable, per-patient feature attributions — required for any real clinical use case.
- **Structured + unstructured fusion**: biomedical NER extracts entities from free-text
  clinical notes and feeds them into the same pipeline as tabular risk features.
- **Grounded generation**: the final summary is not free-form LLM output — it's
  RAG-grounded against a real medical knowledge base (MedlinePlus-style content),
  reducing hallucination risk in a health context.
- **Fully free/open stack**: everything runs on CPU-friendly, open-source models
  (XGBoost, SHAP, scispaCy, sentence-transformers, ChromaDB, Flan-T5) — no paid APIs.

## Project structure

```
healthcare_ai/
├── data/
│   └── sample_patients.csv        # synthetic sample data (no real PHI)
│   └── medlineplus_snippets.json  # small local knowledge base for RAG demo
├── src/
│   ├── risk_model.py              # XGBoost training + SHAP explainability
│   ├── ner_extractor.py           # scispaCy biomedical NER
│   ├── rag_pipeline.py            # sentence-transformers + ChromaDB + Flan-T5
│   └── pipeline.py                # orchestrates all 4 stages end-to-end
├── app/
│   └── gradio_app.py              # UI tying everything together
├── notebooks/
│   └── train_and_explore.ipynb    # Colab-friendly training notebook
├── requirements.txt
└── README.md
```

## Setup (Google Colab recommended — free GPU/CPU, no local installs)

```bash
pip install -r requirements.txt
python -m spacy download en_core_sci_sm   # or en_ner_bc5cdr_md for disease/chem NER
```

> Note: scispaCy models are hosted outside PyPI. If `pip install scispacy` model
> URLs fail in a restricted environment, download the `.whl` directly from
> https://allenai.github.io/scispacy/ and `pip install <file>.whl`.

## Run

```bash
# 1. Train the risk model (writes model.pkl + shap values)
python src/risk_model.py

# 2. Build the RAG vector index (one-time)
python src/rag_pipeline.py --build-index

# 3. Launch the interactive app
python app/gradio_app.py
```

## Pinned versions (from real-world debugging)

```
transformers==4.44.0
langchain==0.2.16
chromadb==0.5.5
sentence-transformers==3.0.1
```

These versions were confirmed compatible together — newer combinations of
`transformers`/`sentence-transformers` frequently break ChromaDB's embedding
function interface.

## Resume bullet this backs up

> Identified key patient risk factors and flagged data anomalies, by engineering
> features and applying XGBoost with SHAP-based interpretability to explain
> individual model predictions.
>
> Delivered explainable, actionable health insights to end users, by building a
> Retrieval-Augmented Generation pipeline (biomedical NER, semantic search, ChromaDB)
> that synthesized findings from structured medical data.

## Disclaimer

This is an educational/portfolio project using **synthetic data only**. It is not
a validated clinical tool and must not be used for real medical decision-making.
