
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from rag_pipeline import retrieve_context, NO_RELEVANT_INFORMATION
from modules.multilingual import get_multilingual_question
from modules.sentiment import analyze_sentiment


# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
FAQ_FILE = BASE_DIR / "knowledge_base" / "customer_faq.txt"

load_dotenv(BASE_DIR / ".env")

MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

st.set_page_config(
    page_title="AI Customer Support",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ==================================================
# 2. LOAD FAQ KNOWLEDGE BASE
# ==================================================

@st.cache_data
def load_faq_text():
    """Read the customer FAQ knowledge base."""

    if not FAQ_FILE.exists():
        return ""

    return FAQ_FILE.read_text(encoding="utf-8").strip()


# ==================================================
# 3. GROQ CLIENT
# ==================================================

def get_groq_client():
    """Create a Groq client using the local environment key."""

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


# ==================================================
# 4. GENERATE RESPONSE: RAG + SENTIMENT + MULTILINGUAL
# ==================================================

def generate_response(question, history, client, faq_text=None):
    """Generate a multilingual, sentiment-aware RAG response."""

    # Step 1: Detect the customer's language and translate
    # the question into English for FAQ retrieval.
    language_info = get_multilingual_question(question)

    language_code = language_info["language_code"]
    english_question = language_info["english_question"]
    language_instruction = language_info["language_instruction"]

    # Step 2: Retrieve relevant FAQ information using semantic RAG.
    context = retrieve_context(
        english_question,
        top_k=3,
        threshold=0.35,
    )

    # Step 3: Analyze customer sentiment.
    sentiment_result = analyze_sentiment(question)

    sentiment = sentiment_result["sentiment"]
    tone = sentiment_result["tone"]
    polarity = sentiment_result["polarity"]

    # Step 4: Build the system prompt.
    if context == NO_RELEVANT_INFORMATION:
        faq_context = (
            "NO_RELEVANT_INFORMATION: No sufficiently relevant "
            "FAQ was found. Do not invent company-specific policies."
        )
    else:
        faq_context = context

    system_prompt = f"""
You are a polite and professional AI customer service assistant.

CUSTOMER SENTIMENT:
- Detected sentiment: {sentiment}
- Sentiment polarity: {polarity}
- Required response tone: {tone}

SENTIMENT RULES:
- For negative sentiment, acknowledge the customer's concern,
  respond empathetically, and offer a practical next step.
- For positive sentiment, respond warmly and appreciatively.
- For neutral sentiment, respond professionally and directly.
- Never argue with a frustrated customer.

LANGUAGE INSTRUCTIONS:
{language_instruction}

- Respond in the same language as the customer's latest message.
- Use English FAQ information to answer questions in other languages.
- Preserve the meaning of follow-up questions.
- Keep product names, order identifiers, and tracking numbers unchanged.
- If a message mixes languages, respond naturally in the dominant language.

CUSTOMER SUPPORT RULES:
- Use the retrieved FAQ information as your primary source of truth.
- Never invent company policies, order details, or tracking information.
- Never promise refunds, deliveries, or cancellations without evidence.
- Do not claim to have completed an actual transaction.
- If the FAQ does not contain the answer, explain that clearly and
  suggest contacting customer support.
- Ask a brief follow-up question when necessary.
- Keep answers concise, clear, and helpful.

RETRIEVED FAQ CONTEXT:
{faq_context}
"""

    # Step 5: Build messages using recent conversation history.
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    for item in history[-10:]:
        if (
            isinstance(item, dict)
            and item.get("role") in ("user", "assistant")
            and isinstance(item.get("content"), str)
        ):
            messages.append({
                "role": item["role"],
                "content": item["content"],
            })

    # Add the current question exactly once.
    messages.append({
        "role": "user",
        "content": question,
    })

    # Step 6: Generate the answer using Groq.
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.2,
        max_tokens=500,
    )

    answer = completion.choices[0].message.content

    if not answer or not answer.strip():
        return (
            "Sorry, I couldn't generate a useful response. "
            "Please try asking your question another way."
        )

    return answer.strip()


# ==================================================
# 5. PAGE STYLING
# ==================================================

st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 1.5rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #172554, #1d4ed8);
        color: white;
        margin-bottom: 1.2rem;
    }

    .hero h1 {
        color: white;
        margin-bottom: 0.4rem;
    }

    .hero p {
        color: #e2e8f0;
        margin-bottom: 0;
    }

    .small-note {
        color: #64748b;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# 6. SIDEBAR
# ==================================================

with st.sidebar:
    st.title("💬 Customer Support")
    st.caption("AI-powered customer assistance")

    st.divider()

    st.subheader("Available support")

    st.markdown(
        """
        - 📦 Delivery questions
        - ↩️ Returns and refunds
        - 💳 Payment issues
        - ❌ Order cancellation
        - 🛠️ Damaged products
        - 🛡️ Warranty questions
        """
    )

    st.divider()

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()

    st.caption(
        "Powered by Groq, semantic RAG, sentiment analysis, "
        "and multilingual support."
    )


# ==================================================
# 7. MAIN INTERFACE
# ==================================================

st.markdown(
    """
    <div class="hero">
        <h1>AI Customer Service Chatbot</h1>
        <p>
            Ask about delivery, returns, refunds, payments,
            damaged products, or warranties.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

faq_text = load_faq_text()
client = get_groq_client()

if not FAQ_FILE.exists():
    st.warning(
        "FAQ file not found. Create "
        "knowledge_base/customer_faq.txt."
    )
elif not faq_text:
    st.warning("Your customer FAQ file is empty.")

if client is None:
    st.error(
        "GROQ_API_KEY is missing. Check your local .env file."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==================================================
# 8. WELCOME SCREEN AND SUGGESTED QUESTIONS
# ==================================================

if not st.session_state.messages:
    st.markdown("### 👋 Welcome!")

    st.write(
        "I'm your virtual support assistant. "
        "Ask a question and I'll try to help using the FAQ knowledge base."
    )

    st.markdown("**Try asking:**")

    examples = [
        "What is your return policy?",
        "How can I request a refund?",
        "My product arrived damaged. Please help.",
        "Thank you for your excellent service!",
    ]

    columns = st.columns(2)

    for index, example in enumerate(examples):
        with columns[index % 2]:
            if st.button(
                example,
                key=f"example_{index}",
                use_container_width=True,
            ):
                st.session_state.pending_question = example
                st.rerun()


# ==================================================
# 9. DISPLAY CHAT HISTORY
# ==================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "user":
            result = analyze_sentiment(message["content"])

            st.caption(
                f"Detected sentiment: "
                f"{result['sentiment'].title()} "
                f"(polarity: {result['polarity']})"
            )


# ==================================================
# 10. PROCESS NEW QUESTIONS
# ==================================================

pending_question = st.session_state.pop(
    "pending_question",
    None,
)

typed_question = st.chat_input("Type your question here...")

question = typed_question or pending_question

if question and question.strip():
    question = question.strip()

    sentiment_result = analyze_sentiment(question)

    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

        st.caption(
            f"Detected sentiment: "
            f"{sentiment_result['sentiment'].title()} "
            f"(polarity: {sentiment_result['polarity']})"
        )

    with st.chat_message("assistant"):
        if client is None:
            answer = (
                "I can't connect to the AI service because the "
                "Groq API key is missing. Please check your .env file."
            )
            st.error(answer)

        elif not faq_text:
            answer = (
                "I can't access the FAQ knowledge base right now. "
                "Please check the FAQ file or contact customer support."
            )
            st.warning(answer)

        else:
            try:
                with st.spinner("Preparing your response..."):
                    answer = generate_response(
                        question=question,
                        history=st.session_state.messages[:-1],
                        client=client,
                        faq_text=faq_text,
                    )

                st.markdown(answer)

            except Exception as error:
                print(f"Chatbot error: {error}")

                answer = (
                    "Sorry, I couldn't generate a response right now. "
                    "Please try again in a moment."
                )

                st.error(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
    })


# ==================================================
# 11. FOOTER
# ==================================================

st.divider()

st.markdown(
    '<p class="small-note">'
    'AI-generated answers may not always be accurate. '
    'Sentiment detection is approximate. Contact official support '
    'for account-specific requests.'
    '</p>',
    unsafe_allow_html=True,
)
