
from pathlib import Path
import xml.etree.ElementTree as ET

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent.parent
MEDICAL_DATA_DIR = BASE_DIR / "medical_data"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model = None
_medical_qa = None
_embeddings = None


def get_model():
    """Load the embedding model once."""
    global _model

    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)

    return _model


def get_tag_text(element, tag_name):
    """Find a tag regardless of XML namespaces."""
    for child in element.iter():
        if child.tag.split("}")[-1].lower() == tag_name.lower():
            return " ".join(child.itertext()).strip()

    return ""


def load_medical_data():
    """Load question-answer pairs from MedQuAD XML files."""
    global _medical_qa

    if _medical_qa is not None:
        return _medical_qa

    if not MEDICAL_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Medical dataset folder not found: {MEDICAL_DATA_DIR}"
        )

    records = []

    for xml_file in MEDICAL_DATA_DIR.rglob("*.xml"):
        try:
            root = ET.parse(xml_file).getroot()

            for element in root.iter():
                if element.tag.split("}")[-1].lower() != "qapair":
                    continue

                question = get_tag_text(element, "Question")
                answer = get_tag_text(element, "Answer")

                if question and answer:
                    records.append({
                        "question": question,
                        "answer": answer,
                        "source": xml_file.name,
                    })

        except ET.ParseError:
            print(f"Skipping invalid XML file: {xml_file}")

    if not records:
        raise ValueError(
            "No question-answer pairs were found. "
            "Check the XML structure and dataset folder."
        )

    _medical_qa = records
    return _medical_qa


def prepare_medical_index():
    """Create embeddings for all medical questions."""
    global _embeddings

    records = load_medical_data()

    if _embeddings is None:
        model = get_model()
        questions = [item["question"] for item in records]

        _embeddings = model.encode(
            questions,
            convert_to_numpy=True,
            show_progress_bar=True
        )

    return records, _embeddings


def search_medical_answers(question, top_k=3, threshold=0.40):
    """Find the most relevant medical FAQ entries."""
    if not question or not question.strip():
        return []

    records, embeddings = prepare_medical_index()
    model = get_model()

    query_embedding = model.encode(
        [question.strip()],
        convert_to_numpy=True
    )

    scores = cosine_similarity(
        query_embedding, embeddings
    )[0]

    best_indices = scores.argsort()[::-1][:top_k]
    results = []

    for index in best_indices:
        score = float(scores[index])

        if score < threshold:
            continue

        results.append({
            **records[index],
            "similarity": score,
        })

    return results
