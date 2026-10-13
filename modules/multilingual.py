
import os
from dotenv import load_dotenv
from groq import Groq
from langdetect import detect, DetectorFactory, LangDetectException

load_dotenv()
DetectorFactory.seed = 0

SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French",
}

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def detect_language(text):
    if not text or not text.strip():
        return "en"

    try:
        code = detect(text.strip())
        return code if code in SUPPORTED_LANGUAGES else "en"
    except LangDetectException:
        return "en"


def get_language_name(language_code):
    return SUPPORTED_LANGUAGES.get(language_code, "English")


def get_language_instruction(language_code):
    language = get_language_name(language_code)
    return (
        f"Respond in {language}. Preserve the customer's meaning "
        "and conversation context. Use a natural, polite, professional "
        "tone. Keep product names and order identifiers unchanged. "
        "Use only the supplied FAQ information."
    )


def translate_to_english(text, language_code):
    """Translate a customer question into English for FAQ retrieval."""
    if language_code == "en":
        return text

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Translate the customer's message into English "
                        "for searching a customer-service FAQ. Return only "
                        "the translation. Do not answer the question."
                    ),
                },
                {"role": "user", "content": text},
            ],
            temperature=0,
        )

        translation = response.choices[0].message.content
        return translation.strip() if translation else text

    except Exception as error:
        print(f"Question translation error: {error}")
        return text


def get_multilingual_question(text):
    """Detect the customer's language and prepare the FAQ query."""
    language_code = detect_language(text)
    english_question = translate_to_english(text, language_code)

    return {
        "language_code": language_code,
        "language_name": get_language_name(language_code),
        "english_question": english_question,
        "language_instruction": get_language_instruction(language_code),
    }
