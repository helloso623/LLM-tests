import easyocr
from PIL import ImageGrab
import pyttsx3
import numpy as np
import time
import speech_recognition as sr
import tkinter as tk
import threading
import torch

try:
    use_gpu = torch.cuda.is_available()
    print(f"GPU CUDA disponible : {use_gpu}")
except Exception:
    use_gpu = False
    print("torch.cuda.is_available() non disponible, fallback CPU.")
reader = easyocr.Reader(['en'], gpu=True)  # GPU activé, accélération maximale

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

def get_ocr_box():
    # Zone plus grande, toujours en haut à gauche (ex: 8% du haut, 8% de la gauche, 50% largeur, 18% hauteur)
    width, height = ImageGrab.grab().size
    box_width = int(width * 0.50)
    box_height = int(height * 0.18)
    x1 = int(width * 0.08)
    y1 = int(height * 0.08)
    x2 = x1 + box_width
    y2 = y1 + box_height
    return x1, y1, x2, y2, width, height

def capture_fullscreen_text():
    # Capture l'écran complet sans masque ni crop
    screen = ImageGrab.grab().convert('RGB')
    img_array = np.array(screen)
    if img_array.size == 0:
        print("Erreur : l'image capturée est vide.")
        return []
    if img_array.dtype != np.uint8:
        img_array = img_array.astype(np.uint8)
    if len(img_array.shape) != 3 or img_array.shape[2] != 3:
        print("Erreur : l'image n'est pas au format RGB attendu.")
        return []
    results = reader.readtext(img_array)
    return results

def test_voice():
    say_text("Hello world, hello world, let's make this cool idea come to life!")

def test_lighting():
    say_text("Lighting is perfect, let's capture some text!")

def test_camera():
    say_text("Camera is working fine, let's capture some text!")

def test_mic():
    say_text("Microphone is ready, let's capture some text!")

def show_focus_overlay(overlay_closed):
    x1, y1, x2, y2, width, height = get_ocr_box()
    def _show():
        root = tk.Tk()
        root.overrideredirect(True)
        root.attributes('-topmost', True)
        transparent_color = '#123456'
        root.config(bg=transparent_color)
        try:
            root.wm_attributes('-transparentcolor', transparent_color)
        except Exception:
            pass
        root.geometry(f"{width}x{height}+0+0")
        canvas = tk.Canvas(root, width=width, height=height, highlightthickness=0, bg=transparent_color)
        canvas.pack()
        # Rectangle noir opaque partout
        canvas.create_rectangle(0, 0, width, height, fill='#000000', outline='')
        # Découpe la zone de la boîte (on dessine un rectangle transparent par-dessus)
        canvas.create_rectangle(x1, y1, x2, y2, fill=transparent_color, outline='#FF2222', width=5)
        def close_overlay():
            root.destroy()
        # Attend l'event pour fermer
        def wait_and_close():
            overlay_closed.wait()
            root.after(0, close_overlay)
        threading.Thread(target=wait_and_close, daemon=True).start()
        root.mainloop()
    threading.Thread(target=_show, daemon=True).start()

def wait_for_trigger():
    recognizer = sr.Recognizer()
    if not sr.Microphone.list_microphone_names():
        print("Aucun microphone détecté. Branchez un micro et relancez.")
        return None
    print("Microphones disponibles :", sr.Microphone.list_microphone_names())
    mic = sr.Microphone()
    got_start = False
    ocr_text = None
    overlay_closed = threading.Event()
    show_focus_overlay(overlay_closed)
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Calibration bruit ambiant... (1s)")
        time.sleep(1)
        print("Listening for trigger word...")
        while True:
            try:
                print("Parlez maintenant...")
                audio = recognizer.listen(source)
                command = recognizer.recognize_google(audio).lower()
                print(f"Heard: {command}")
                if  "camera" in command or "microphone" in command or "lighting" in command or "test" in command:
                    print("Mot-clé 'test' détecté : test vocal !")
            
                    if "camera" in command:
                        test_camera()
                    elif "lighting" in command:
                        test_lighting()
                    elif "microphone" in command:
                        test_mic()
                    else:
                        test_voice()
                elif not got_start and "start" in command:
                    print("Mot-clé 'start' détecté : capture d'image !")
                    results = capture_fullscreen_text()  # capture tout l'écran pendant que l'overlay est là
                    overlay_closed.set()  # retire l'overlay juste après
                    time.sleep(0.2)
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
                elif got_start and ("he's alive" in command):
                    print("Trigger word detected.")
                    print("=== CODE ACTIVÉ ===")
                    if ocr_text:
                        say_text(ocr_text)
                    else:
                        say_text("No text detected")
                    return
            except sr.UnknownValueError:
                print("Rien compris, recommencez...")
                continue
            except sr.RequestError as e:
                print(f"Recognition error: {e}")
                break
            except OSError as e:
                print(f"Erreur micro : {e}")
                break

# Main execution
wait_for_trigger()