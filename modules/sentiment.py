
from textblob import TextBlob


def analyze_sentiment(text: str) -> dict:
    """
    Classify a message as positive, negative, or neutral.
    Returns sentiment, polarity, and a suggested response tone.
    """
    if not text or not text.strip():
        return {
            "sentiment": "neutral",
            "polarity": 0.0,
            "tone": "helpful",
        }

    polarity = float(TextBlob(text).sentiment.polarity)

    if polarity > 0.1:
        sentiment = "positive"
        tone = "friendly and appreciative"
    elif polarity < -0.1:
        sentiment = "negative"
        tone = "empathetic and reassuring"
    else:
        sentiment = "neutral"
        tone = "professional and helpful"

    return {
        "sentiment": sentiment,
        "polarity": round(polarity, 3),
        "tone": tone,
    }
