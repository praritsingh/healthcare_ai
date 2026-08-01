"""
gradio_app.py
-------------
Interactive UI for the Healthcare Risk Prediction & Insights System.

Run:
    python app/gradio_app.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import gradio as gr
import pandas as pd

from risk_model import load_data, load_artifacts, explain_patient, DATA_PATH
from ner_extractor import BiomedicalNER
from rag_pipeline import run_full_rag_demo

df = load_data(DATA_PATH)
model, explainer = load_artifacts()
ner = BiomedicalNER()


def analyze_patient(patient_id: str):
    if patient_id not in df["patient_id"].values:
        return "Patient not found.", "", "", ""

    row = df[df["patient_id"] == patient_id].iloc[0]

    risk_result = explain_patient(model, explainer, df, patient_id)
    entities = ner.extract(row["clinical_note"])
    rag_result = run_full_rag_demo(
        patient_note=row["clinical_note"],
        risk_factors=risk_result["top_factors"],
    )

    risk_md = f"### Risk Score: **{risk_result['risk_score']:.1%}**\n\n"
    risk_md += "**Top contributing factors:**\n\n"
    for f in risk_result["top_factors"]:
        direction = "⬆️ increases risk" if f["impact"] > 0 else "⬇️ decreases risk"
        risk_md += f"- `{f['feature']}` — {direction} ({f['impact']:+.3f})\n"

    entities_md = "**Extracted clinical entities:**\n\n" + "\n".join(
        f"- {e['entity']} ({e['label']})" for e in entities
    ) if entities else "No entities detected."

    context_md = "**Retrieved medical context:**\n\n" + "\n\n".join(
        f"*{c['topic']}*: {c['text']}" for c in rag_result["retrieved_context"]
    )

    summary_md = f"### Plain-language Insight Summary\n\n{rag_result['summary']}"

    return risk_md, entities_md, context_md, summary_md


with gr.Blocks(title="Healthcare Risk Prediction & Insights System") as demo:
    gr.Markdown("# 🩺 Healthcare Risk Prediction & Insights System")
    gr.Markdown(
        "Educational demo using **synthetic data only**. "
        "Not a validated clinical tool."
    )

    with gr.Row():
        patient_dropdown = gr.Dropdown(
            choices=df["patient_id"].tolist(),
            label="Select a patient",
            value=df["patient_id"].iloc[0],
        )
        analyze_btn = gr.Button("Analyze", variant="primary")

    with gr.Row():
        risk_output = gr.Markdown()
        entities_output = gr.Markdown()

    context_output = gr.Markdown()
    summary_output = gr.Markdown()

    analyze_btn.click(
        analyze_patient,
        inputs=[patient_dropdown],
        outputs=[risk_output, entities_output, context_output, summary_output],
    )

if __name__ == "__main__":
    demo.launch()
