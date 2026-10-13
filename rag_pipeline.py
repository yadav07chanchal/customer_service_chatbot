
from pathlib import Path
from functools import lru_cache
import re

import numpy as np
from sentence_transformers import SentenceTransformer


# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
FAQ_FILE = BASE_DIR / "knowledge_base" / "customer_faq.txt"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.35
NO_RELEVANT_INFORMATION = "NO_RELEVANT_INFORMATION"


# ==================================================
# 2. LOAD THE EMBEDDING MODEL
# ==================================================

@lru_cache(maxsize=1)
def get_model():
    """Load and cache the semantic embedding model."""
    return SentenceTransformer(MODEL_NAME)


# ==================================================
# 3. LOAD AND VALIDATE FAQ ENTRIES
# ==================================================

def load_faqs():
    """Load valid FAQ entries from the knowledge-base file."""

    if not FAQ_FILE.exists():
        raise FileNotFoundError(
            f"FAQ file not found: {FAQ_FILE}"
        )

    text = FAQ_FILE.read_text(encoding="utf-8").strip()

    if not text:
        return []

    sections = re.split(
        r"(?im)(?=^Category:\s*)",
        text
    )

    documents = []

    for section in sections:
        section = section.strip()

        if not section.startswith("Category:"):
            continue

        category = re.search(
            r"(?m)^Category:\s*(.+)$", section
        )
        question = re.search(
            r"(?m)^Question:\s*(.+)$", section
        )
        answer = re.search(
            r"(?m)^Answer:\s*(.+)$", section
        )

        # Skip malformed entries instead of breaking the app.
        if not (category and question and answer):
            continue

        if not all(
            match.group(1).strip()
            for match in (category, question, answer)
        ):
            continue

        documents.append(section)

    return documents


# ==================================================
# 4. ADD A NEW FAQ
# ==================================================

def add_faq(category, question, answer):
    """Add an FAQ and prevent duplicate questions."""

    category = str(category).strip()
    question = str(question).strip()
    answer = str(answer).strip()

    if not category or not question or not answer:
        raise ValueError(
            "Category, question, and answer are required."
        )

    if any(
        "\n" in value or "\r" in value
        for value in (category, question, answer)
    ):
        raise ValueError(
            "Each FAQ field must be entered on a single line."
        )

    normalized_question = " ".join(
        question.casefold().split()
    )

    existing_faqs = load_faqs() if FAQ_FILE.exists() else []

    for faq in existing_faqs:
        match = re.search(
            r"(?m)^Question:\s*(.+)$", faq
        )

        if match:
            existing_question = " ".join(
                match.group(1).casefold().split()
            )

            if existing_question == normalized_question:
                return "A FAQ with this question already exists."

    FAQ_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    entry = (
        f"Category: {category}\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n"
    )

    with FAQ_FILE.open("a", encoding="utf-8") as file:
        if FAQ_FILE.stat().st_size > 0:
            file.write("\n")

        file.write(entry)

    # The next retrieval reads the updated FAQ file.
    # Clear cached embeddings so they are rebuilt if necessary.
    encode_faqs.cache_clear()

    return "FAQ added successfully."


# ==================================================
# 5. CREATE FAQ EMBEDDINGS
# ==================================================


@lru_cache(maxsize=4)
def encode_faqs(documents):
    """Create and cache embeddings for FAQ documents."""

    if not documents:
        return np.empty((0, 0), dtype=np.float32)

    model = get_model()
    searchable_texts = []

    for document in documents:
        category_match = re.search(
            r"(?m)^Category:\s*(.+)$", document
        )
        question_match = re.search(
            r"(?m)^Question:\s*(.+)$", document
        )

        category = (
            category_match.group(1).strip()
            if category_match else ""
        )
        question = (
            question_match.group(1).strip()
            if question_match else ""
        )

        searchable_texts.append(
            f"Category: {category}. "
            f"Question: {question}. "
            f"FAQ details: {document}"
        )

    return model.encode(
        searchable_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )


    return model.encode(
        searchable_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )


# ==================================================
# 6. SEMANTIC FAQ RETRIEVAL
# ==================================================

def retrieve_context(
    question,
    top_k=3,
    threshold=SIMILARITY_THRESHOLD
):
    """
    Retrieve relevant FAQ entries using semantic similarity.

    Returns matching FAQ text, or NO_RELEVANT_INFORMATION
    when no entry meets the similarity threshold.
    """

    if not isinstance(question, str) or not question.strip():
        return NO_RELEVANT_INFORMATION

    try:
        documents = load_faqs()

        if not documents:
            return NO_RELEVANT_INFORMATION

        document_embeddings = encode_faqs(tuple(documents))

        model = get_model()

        query_embedding = model.encode(
            [question.strip()],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )[0]

        # Embeddings are normalized, so dot product gives cosine similarity.
        scores = np.dot(
            document_embeddings,
            query_embedding
        )

        ranked_indices = np.argsort(scores)[::-1]

        if len(ranked_indices) == 0:
            return NO_RELEVANT_INFORMATION

        best_score = float(scores[ranked_indices[0]])

        if best_score < threshold:
            return NO_RELEVANT_INFORMATION

        results = []

        for index in ranked_indices:
            score = float(scores[index])

            if score < threshold:
                continue

            # Avoid adding FAQs substantially weaker than the best match.
            if score < best_score - 0.12:
                continue

            results.append(documents[int(index)])

            if len(results) >= max(1, int(top_k)):
                break

        if not results:
            return NO_RELEVANT_INFORMATION

        return "\n\n".join(results)

    except (FileNotFoundError, ValueError):
        raise


# ==================================================
# 7. COMMAND-LINE TESTING
# ==================================================

def main():
    print("\n=== Semantic FAQ Knowledge Base ===")
    print("1. Search FAQs")
    print("2. Add an FAQ")
    print("3. Exit")

    while True:
        choice = input(
            "\nChoose an option (1, 2, or 3): "
        ).strip()

        if choice == "1":
            question = input(
                "Enter customer question: "
            ).strip()

            try:
                result = retrieve_context(question)

                if result == NO_RELEVANT_INFORMATION:
                    print("\nNo relevant FAQ found.")
                else:
                    print("\nRetrieved FAQ information:\n")
                    print(result)

            except Exception as error:
                print(f"\nSearch failed: {error}")

        elif choice == "2":
            category = input("Enter category: ")
            question = input("Enter FAQ question: ")
            answer = input("Enter FAQ answer: ")

            try:
                print("\n" + add_faq(
                    category, question, answer
                ))
            except (ValueError, OSError) as error:
                print(f"\nCould not add FAQ: {error}")

        elif choice == "3":
            print("Exiting Semantic FAQ Knowledge Base.")
            break

        else:
            print("Invalid choice. Enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
