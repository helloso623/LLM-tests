import pyttsx3
import time
from datetime import datetime

def main():
    """
    Ce script est exécuté lorsque le bouton Shelly est pressé.
    Il écrit un message dans un fichier log et l'affiche à l'écran.
    """
    log_file = "coderunshelly_log.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"coderunshelly.py exécuté à {timestamp}\n"

    print(message)

    try:
        with open(log_file, "a") as f:
            f.write(message)
        print(f"Message enregistré dans le fichier log : {log_file}")
    except Exception as e:
        print(f"Erreur lors de l'écriture dans le fichier log : {e}")

    # Initialisation du moteur de synthèse vocale
    engine = pyttsx3.init()
    message = "Hello, I love potato"

    # Log pour confirmer l'exécution
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

    # Faire parler le message
    engine.say(message)
    engine.runAndWait()

    # Gardez la fenêtre ouverte pendant 5 secondes pour voir le message
    time.sleep(5)

if __name__ == "__main__":
    main()
