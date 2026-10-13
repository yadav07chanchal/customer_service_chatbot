
import os
from dotenv import load_dotenv
from groq import Groq

from rag_pipeline import retrieve_context
from modules.sentiment import analyze_sentiment, empathy_message
from modules.multilingual import (
    detect_language,
    get_language_name,
    get_language_instruction,
    translate_to_english,
)

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY is missing from the .env file. "
        "Please check your environment configuration."
    )

client = Groq(api_key=api_key)

MODEL_NAME = "openai/gpt-oss-20b"


def get_response(question):
    """
    Generate a customer service response using:
    1. Language detection
    2. Sentiment analysis
    3. FAQ retrieval through RAG
    4. Groq LLM response generation

    Returns:
        tuple: (answer, source)
    """

    if not isinstance(question, str) or not question.strip():
        return "Please enter a valid question.", None

    question = question.strip()

    # Step 1: Detect the customer's language
    language_code = detect_language(question)
    language_name = get_language_name(language_code)
    language_instruction = get_language_instruction(language_code)

    # Step 2: Analyze customer sentiment
    sentiment_result = analyze_sentiment(question)
    sentiment = sentiment_result["sentiment"]

    # Step 3: Translate the question for English FAQ retrieval
    search_question = translate_to_english(question, language_code)

    # Step 4: Retrieve relevant FAQ context
    try:
        context = retrieve_context(search_question)
    except Exception as error:
        print(f"Knowledge retrieval error: {error}")

        answer = (
            "I'm sorry, but I'm unable to access the customer "
            "support knowledge base right now. Please try again "
            "later or contact customer support."
        )

        if sentiment == "negative":
            answer = empathy_message(sentiment) + "\n\n" + answer

        return answer, None

    # Step 5: Check whether relevant FAQ information exists
    if (
        context is None
        or (
            isinstance(context, str)
            and (
                not context.strip()
                or context.strip() == "NO_RELEVANT_INFORMATION"
            )
        )
    ):
        answer = (
            "I'm sorry, I couldn't find this information in our "
            "customer support knowledge base. Please contact "
            "customer support for assistance."
        )

        if sentiment == "negative":
            answer = empathy_message(sentiment) + "\n\n" + answer

        return answer, None

    # Step 5: Choose the response tone
    if sentiment == "negative":
        tone_instruction = (
            "The customer may be frustrated or upset. Show brief, "
            "natural empathy. Be calm, respectful, and solution-focused. "
            "Avoid repetitive apologies and never blame the customer."
        )
    elif sentiment == "positive":
        tone_instruction = (
            "Use a warm, friendly, appreciative, and helpful tone."
        )
    else:
        tone_instruction = (
            "Use a clear, professional, polite, and concise tone."
        )

    # Step 6: Generate the answer using Groq
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional multilingual customer "
                        "service assistant. Answer using only the supplied "
                        "FAQ context. Do not invent company policies, "
                        "delivery dates, refund decisions, prices, or facts. "
                        "If the context does not answer the question, "
                        "explain that the information is unavailable and "
                        "recommend contacting customer support. "
                        + language_instruction
                        + " "
                        + tone_instruction
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Customer question:\n{question}\n\n"
                        f"FAQ context:\n{context}\n\n"
                        "Answer the customer directly. Follow the required "
                        "response language and use the available FAQ facts."
                    ),
                },
            ],
            temperature=0.2,
        )

        answer = response.choices[0].message.content

        if not answer:
            answer = "I'm sorry, I couldn't generate an answer. Please try again."

    except Exception as error:
        print(f"Groq API error: {error}")

        answer = (
            "I'm sorry, but I'm unable to generate a response right now. "
            "Please try again in a moment."
        )

        if sentiment == "negative":
            answer = empathy_message(sentiment) + "\n\n" + answer

    # Step 7: Return the answer and its FAQ source
    return answer, context


if __name__ == "__main__":
    print("Multilingual Customer Service Chatbot")
    print("Supported languages: English, Hindi, Spanish, French")
    print("Type 'quit' to exit.")

    while True:
        question = input("\nCustomer: ").strip()

        if question.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if not question:
            print("Please enter a question.")
            continue

        language_code = detect_language(question)
        language_name = get_language_name(language_code)
        sentiment_result = analyze_sentiment(question)

        answer, source = get_response(question)

        print("Detected language:", language_name, f"({language_code})")
        print("Detected sentiment:", sentiment_result["sentiment"])
        print("Sentiment polarity:", sentiment_result["polarity"])
        print("\nCustomer Service Bot:", answer)

        if source:
            print("\nFAQ Source:\n", source)
