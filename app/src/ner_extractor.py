"""
ner_extractor.py
-----------------
Extracts biomedical entities (conditions, drugs, symptoms) from free-text
clinical notes using scispaCy. This is the "3. Biomedical NER" stage.

Falls back gracefully to a lightweight keyword matcher if scispaCy / its
models are not installed, so the rest of the pipeline can still be
exercised in restricted environments (e.g. CI, quick demos).
"""

from typing import List, Dict

try:
    import spacy
    _SCISPACY_AVAILABLE = True
except ImportError:
    _SCISPACY_AVAILABLE = False


# A small curated fallback vocabulary — used only if scispaCy isn't installed.
FALLBACK_TERMS = [
    "hypertension", "diabetes", "type 2 diabetes", "atrial fibrillation",
    "chest tightness", "shortness of breath", "fatigue", "blurred vision",
    "headaches", "dizziness", "palpitations", "neuropathy", "cough",
    "metformin", "lisinopril", "atorvastatin", "warfarin",
]


class BiomedicalNER:
    def __init__(self, model_name: str = "en_core_sci_sm"):
        self.model_name = model_name
        self.nlp = None
        if _SCISPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(model_name)
            except OSError:
                print(
                    f"[ner_extractor] Model '{model_name}' not found locally. "
                    f"Run: python -m spacy download {model_name}  "
                    f"(or install the scispaCy wheel per README). Falling back "
                    f"to keyword matching for now."
                )
                self.nlp = None

    def extract(self, text: str) -> List[Dict]:
        if self.nlp is not None:
            return self._extract_with_scispacy(text)
        return self._extract_with_fallback(text)

    def _extract_with_scispacy(self, text: str) -> List[Dict]:
        doc = self.nlp(text)
        return [
            {"entity": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
            for ent in doc.ents
        ]

    def _extract_with_fallback(self, text: str) -> List[Dict]:
        text_lower = text.lower()
        found = []
        for term in FALLBACK_TERMS:
            idx = text_lower.find(term)
            if idx != -1:
                found.append({
                    "entity": text[idx: idx + len(term)],
                    "label": "CONDITION_OR_DRUG",
                    "start": idx,
                    "end": idx + len(term),
                })
        return found


if __name__ == "__main__":
    ner = BiomedicalNER()
    sample_note = (
        "Patient reports chest tightness on exertion and occasional shortness "
        "of breath. History of hypertension, prescribed lisinopril."
    )
    entities = ner.extract(sample_note)
    print("Extracted entities:")
    for e in entities:
        print(f"  {e['entity']!r:35s} [{e['label']}]")
