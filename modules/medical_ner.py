
import re
import spacy

nlp = spacy.load("en_core_web_sm")

MEDICAL_TERMS = {
    "diabetes": "Condition",
    "type 1 diabetes": "Condition",
    "type 2 diabetes": "Condition",
    "asthma": "Condition",
    "cancer": "Condition",
    "hypertension": "Condition",
    "high blood pressure": "Condition",
    "heart disease": "Condition",
    "anemia": "Condition",
    "arthritis": "Condition",
    "migraine": "Condition",
    "depression": "Condition",
    "anxiety": "Condition",
    "headache": "Symptom",
    "fever": "Symptom",
    "cough": "Symptom",
    "chest pain": "Symptom",
    "shortness of breath": "Symptom",
    "difficulty breathing": "Symptom",
    "nausea": "Symptom",
    "vomiting": "Symptom",
    "fatigue": "Symptom",
    "stomach pain": "Symptom",
    "abdominal pain": "Symptom",
    "insulin": "Medication",
    "paracetamol": "Medication",
    "acetaminophen": "Medication",
    "ibuprofen": "Medication",
    "aspirin": "Medication",
    "antibiotics": "Medication",
    "heart": "Body part",
    "lungs": "Body part",
    "liver": "Body part",
    "kidney": "Body part",
    "brain": "Body part",
    "stomach": "Body part",
}


def extract_medical_entities(text):
    """Detect known medical terms and general named entities."""
    if not text or not text.strip():
        return []

    found = {}

    # Use spaCy for general named entity recognition.
    doc = nlp(text)

    for entity in doc.ents:
        key = entity.text.casefold()
        found[key] = {
            "text": entity.text,
            "label": entity.label_,
            "source": "spaCy",
        }

    # Match whole terms, including multi-word medical phrases.
    for term, category in MEDICAL_TERMS.items():
        pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"

        match = re.search(pattern, text, flags=re.IGNORECASE)

        if match:
            key = term.casefold()
            found[key] = {
                "text": match.group(0),
                "label": category,
                "source": "medical vocabulary",
            }

    return list(found.values())
