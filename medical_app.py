
import streamlit as st

from modules.medical_qa import (
    load_medical_data,
    search_medical_answers,
)
from modules.medical_ner import extract_medical_entities


st.set_page_config(
    page_title="MedQuAD Medical Q&A",
    page_icon="🩺",
    layout="wide",
)

st.title("🩺 Medical Question & Answer Chatbot")
st.caption("MedQuAD-powered medical information assistant")

st.warning(
    "Educational information only. This chatbot does not diagnose "
    "conditions or replace a qualified healthcare professional. "
    "For emergencies, contact local emergency services."
)

# Load the dataset
try:
    with st.spinner("Loading medical dataset..."):
        records = load_medical_data()

    st.success(
        f"Medical knowledge base loaded: {len(records):,} Q&A pairs"
    )

except Exception as error:
    st.error(f"Could not load the medical dataset: {error}")
    st.stop()


# Initialize chat history
if "medical_history" not in st.session_state:
    st.session_state.medical_history = []


# Display previous conversations
for item in st.session_state.medical_history:
    with st.chat_message("user"):
        st.write(item["question"])

    with st.chat_message("assistant"):
        entities = item.get("entities", [])

        if entities:
            st.markdown("**Detected medical terms**")
            st.write(
                ", ".join(
                    f"{entity['text']} ({entity['label']})"
                    for entity in entities
                )
            )

        if item["results"]:
            st.markdown("**Relevant information from MedQuAD**")

            for result in item["results"]:
                st.markdown(
                    f"**Related question:** {result['question']}"
                )
                st.write(result["answer"])
                st.caption(
                    f"Source: {result['source']} | "
                    f"Similarity score: {result['similarity']:.2f}"
                )
        else:
            st.info(
                "No sufficiently relevant answer was found in the "
                "available dataset. Please consult a qualified "
                "healthcare professional."
            )


# Accept a new question
question = st.chat_input("Ask a medical question...")

if question:

    normalized_question = question.casefold()

    urgent_phrases = [
        "chest pain",
        "difficulty breathing",
        "cannot breathe",
        "can't breathe",
        "severe bleeding",
        "unconscious",
        "signs of stroke",
    ]

    if any(phrase in normalized_question for phrase in urgent_phrases):
        st.warning(
            "Your question may describe a medical emergency. "
            "Please seek urgent medical care or contact your "
            "local emergency service. Do not rely on this chatbot "
            "to assess an emergency."
        )

    # Continue with your existing question-processing code below.
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            # Step 1: Extract medical entities
            with st.spinner("Identifying medical terms..."):
                entities = extract_medical_entities(question)

            if entities:
                st.markdown("### 🔎 Detected Medical Terms")

                for entity in entities:
                    st.write(
                        f"• **{entity['text']}** — {entity['label']}"
                    )
            else:
                st.info(
                    "No medical terms were recognized. "
                    "You can still search the medical knowledge base."
                )

            # Step 2: Search the MedQuAD dataset
            with st.spinner("Searching the medical knowledge base..."):
                results = search_medical_answers(question)

            if results:
                st.markdown("### 📚 Relevant Medical Information")

                for result in results:
                    st.markdown(
                        f"**Related question:** {result['question']}"
                    )
                    st.write(result["answer"])
                    st.caption(
                        f"Source: {result['source']} | "
                        f"Similarity score: {result['similarity']:.2f}"
                    )
            else:
                st.info(
                    "No sufficiently relevant answer was found in "
                    "the dataset. Please consult a qualified "
                    "healthcare professional."
                )

            # Step 3: Save the conversation
            st.session_state.medical_history.append({
                "question": question,
                "entities": entities,
                "results": results,
            })

        except Exception as error:
            st.error(f"Something went wrong: {error}")


# Sidebar information
with st.sidebar:
    st.header("About")
    st.write(
        "This prototype searches MedQuAD for relevant medical "
        "question-and-answer pairs."
    )

    st.write("**Features**")
    st.write("- Medical FAQ retrieval")
    st.write("- Basic medical entity recognition")
    st.write("- Conversation history")

    if st.button("Clear conversation"):
        st.session_state.medical_history = []
        st.rerun()
