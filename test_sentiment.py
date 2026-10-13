from modules.sentiment import analyze_sentiment

messages = [
    "I love your service. Thank you!",
    "My order is late and I am very disappointed.",
    "What is your return policy?",
]

for message in messages:
    result = analyze_sentiment(message)
    print(message)
    print(result)
    print()