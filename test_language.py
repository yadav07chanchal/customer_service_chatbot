
from langdetect import detect

messages = [
    "Hello, how are you?",
    "मेरा ऑर्डर कहाँ है?",
    "¿Dónde está mi pedido?",
    "Où est ma commande ?"
]

for message in messages:
    print("Message:", message)
    print("Language:", detect(message))
    print("-" * 30)
