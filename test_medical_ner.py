
from modules.medical_ner import extract_medical_entities

questions = [
    "What are the symptoms of diabetes and asthma?",
    "Can high blood pressure cause headaches?",
    "What is insulin used for?",
    "I have difficulty breathing.",
]

for question in questions:
    print(f"\nQuestion: {question}")

    entities = extract_medical_entities(question)

    if entities:
        for entity in entities:
            print(f"  {entity['text']} -> {entity['label']}")
    else:
        print("  No medical terms detected.")
