import easyocr
from PIL import ImageGrab
import pyttsx3
import numpy as np
import time
import speech_recognition as sr
import threading
import tkinter as tk

reader = easyocr.Reader(['en'])

engine = pyttsx3.init()
# Affiche toutes les voix disponibles
print("Voix disponibles :")
for v in engine.getProperty('voices'):
    print(f"- {v.id} | {v.name}")
# Sélectionne une voix Microsoft plus naturelle si possible
selected_voice = None
for v in engine.getProperty('voices'):
    if 'zira' in v.id.lower() or 'zira' in v.name.lower():
        selected_voice = v.id
        break
if not selected_voice:
    for v in engine.getProperty('voices'):
        if 'david' in v.id.lower() or 'david' in v.name.lower():
            selected_voice = v.id
            break
if not selected_voice and engine.getProperty('voices'):
    selected_voice = engine.getProperty('voices')[0].id
if selected_voice:
    engine.setProperty('voice', selected_voice)
    print(f"Voix sélectionnée : {selected_voice}")
engine.setProperty('rate', 150)  # plus lent que la valeur par défaut (200)
engine.setProperty('volume', 1.0)  # volume max

def say_text(text):
    engine.say(text)
    engine.runAndWait()

def show_overlay(box, screen_size):
    root = tk.Tk()
    root.attributes('-topmost', True)
    root.overrideredirect(True)
    root.attributes('-alpha', 0.4)
    root.geometry(f"{screen_size[0]}x{screen_size[1]}+0+0")
    # Couleur de transparence peu utilisée
    transparent_color = '#222222'
    canvas = tk.Canvas(root, width=screen_size[0], height=screen_size[1], highlightthickness=0, bg=transparent_color)
    canvas.pack()
    x1, y1, x2, y2 = box
    # Rectangle bordure rouge
    canvas.create_rectangle(x1, y1, x2, y2, outline='#FF0000', width=6)
    root.lift()
    try:
        root.attributes('-transparentcolor', transparent_color)
    except Exception:
        pass
    return root

def capture_fullscreen_text():
    screen = ImageGrab.grab().convert('RGB')
    width, height = screen.size
    # Zone centrale : 50% plus large et 20% plus haute que 1/3, bien centrée
    box_width = int((width // 3) * 1.5)
    box_height = int((height // 3) * 1.2)
    center_x = width // 2
    center_y = height // 2
    box = (
        center_x - box_width // 2,
        center_y - box_height // 2,
        center_x + box_width // 2,
        center_y + box_height // 2
    )
    img_array = np.array(screen)
    # On noircit tout sauf la zone centrale
    mask = np.zeros_like(img_array)
    mask[box[1]:box[3], box[0]:box[2], :] = img_array[box[1]:box[3], box[0]:box[2], :]
    if mask.size == 0:
        print("Erreur : l'image capturée est vide.")
        return []
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if len(mask.shape) != 3 or mask.shape[2] != 3:
        print("Erreur : l'image n'est pas au format RGB attendu.")
        return []
    results = reader.readtext(mask)
    return results

def test_voice():
    say_text("Hello world, hello world, let's make this cool idea come to life!")

def wait_for_trigger():
    recognizer = sr.Recognizer()
    if not sr.Microphone.list_microphone_names():
        print("Aucun microphone détecté. Branchez un micro et relancez.")
        return None
    print("Microphones disponibles :", sr.Microphone.list_microphone_names())
    mic = sr.Microphone()
    # Calcul du cadre à afficher
    screen = ImageGrab.grab().convert('RGB')
    width, height = screen.size
    box_width = int((width // 3) * 1.5)
    box_height = int((height // 3) * 1.2)
    center_x = width // 2
    center_y = height // 2
    box = (
        center_x - box_width // 2,
        center_y - box_height // 2,
        center_x + box_width // 2,
        center_y + box_height // 2
    )
    overlay_root = None
    def overlay_thread():
        nonlocal overlay_root
        overlay_root = show_overlay(box, (width, height))
        overlay_root.mainloop()
    t = threading.Thread(target=overlay_thread)
    t.start()
    try:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            print("Calibration bruit ambiant... (1s)")
            time.sleep(1)
            print("Listening for trigger word...")
            got_start = False
            ocr_text = None
            while True:
                try:
                    print("Parlez maintenant...")
                    audio = recognizer.listen(source)
                    command = recognizer.recognize_google(audio).lower()
                    print(f"Heard: {command}")
                    if "test" in command:
                        print("Mot-clé 'test' détecté : test vocal !")
                        test_voice()
                    elif not got_start and "start" in command:
                        print("Mot-clé 'start' détecté : capture d'image !")
                        if overlay_root:
                            overlay_root.destroy()
                        results = capture_fullscreen_text()
                        if results:
                            ocr_text = ' '.join([res[1] for res in results])
                            print("Full OCR results:")
                            for res in results:
                                print(res)
                            print("\nExtracted text:")
                            print(ocr_text)
                        else:
                            ocr_text = None
                            print("No text detected.")
                        got_start = True
                        print("Dites 'go' pour lire le texte à voix haute...")
                    elif got_start and ("he's alive" in command or "go" in command):
                        print("Trigger word detected.")
                        print("=== CODE ACTIVÉ ===")
                        if ocr_text:
                            say_text(ocr_text)
                        else:
                            say_text("No text detected")
                        if overlay_root:
                            overlay_root.destroy()
                        return
                except sr.UnknownValueError:
                    print("Rien compris, recommencez...")
                    continue
                except sr.RequestError as e:
                    print(f"Recognition error: {e}")
                    break
    finally:
        if overlay_root:
            try:
                overlay_root.destroy()
            except:
                pass

# Main execution
wait_for_trigger()
