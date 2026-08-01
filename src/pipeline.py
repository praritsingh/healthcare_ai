"""
pipeline.py
-----------
Orchestrates the full 5-stage pipeline for a single patient:

  1. Risk prediction (XGBoost)
  2. Explainability (SHAP)
  3. Biomedical NER on clinical notes (scispaCy)
  4. RAG retrieval over medical knowledge base (ChromaDB)
  5. Grounded insight synthesis (Flan-T5)

Usage:
    python src/pipeline.py --patient_id P0001
"""

import argparse
import pandas as pd

from risk_model import load_data, load_artifacts, explain_patient, DATA_PATH
from ner_extractor import BiomedicalNER
from rag_pipeline import run_full_rag_demo


def run_pipeline_for_patient(patient_id: str) -> dict:
    df = load_data(DATA_PATH)
    model, explainer = load_artifacts()

    # Stage 1 + 2: risk score + SHAP explanation
    risk_result = explain_patient(model, explainer, df, patient_id)

    # Stage 3: biomedical NER on the clinical note
    row = df[df["patient_id"] == patient_id].iloc[0]
    ner = BiomedicalNER()
    entities = ner.extract(row["clinical_note"])

    # Stage 4 + 5: RAG retrieval + grounded synthesis
    rag_result = run_full_rag_demo(
        patient_note=row["clinical_note"],
        risk_factors=risk_result["top_factors"],
    )

    return {
        "patient_id": patient_id,
        "risk_score": risk_result["risk_score"],
        "top_factors": risk_result["top_factors"],
        "extracted_entities": entities,
        "retrieved_context": rag_result["retrieved_context"],
        "insight_summary": rag_result["summary"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient_id", type=str, required=True)
    args = parser.parse_args()

    result = run_pipeline_for_patient(args.patient_id)

    print(f"\n=== Patient {result['patient_id']} ===")
    print(f"Risk score: {result['risk_score']}")
    print("Top risk factors:")
    for f in result["top_factors"]:
        print(f"  - {f['feature']}: {f['impact']:+.3f}")
    print("\nExtracted clinical entities:")
    for e in result["extracted_entities"]:
        print(f"  - {e['entity']} [{e['label']}]")
    print("\nInsight summary:")
    print(result["insight_summary"])
