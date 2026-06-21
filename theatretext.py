import easyocr
import pyttsx3
import numpy as np
from PIL import ImageGrab, ImageDraw
import speech_recognition as sr
import torch
import time
import tkinter as tk
import threading

reader = easyocr.Reader(['en'], gpu=False)  # Désactive le GPU pour éviter les lenteurs et bugs

engine = pyttsx3.init()
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
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

def say_text(text):
    engine.say(text)
    engine.runAndWait()

def capture_fullscreen_text():
    screen = ImageGrab.grab().convert('RGB')
    width, height = screen.size
    # Zone centrale : 50% plus large et 20% plus haute que 1/3, bien centrée
    box_width = int((width // 3) * 1.5)
    box_height = int((height // 3) * 1.2)
    center_x = width // 2
    center_y = height // 2
    x1 = center_x - box_width // 2
    y1 = center_y - box_height // 2
    x2 = center_x + box_width // 2
    y2 = center_y + box_height // 2
    # Crop la zone utile (plus rapide et plus fiable que le masque)
    crop_img = screen.crop((x1, y1, x2, y2))
    # Redimensionne pour accélérer l'OCR (max 600px de large)
    max_width = 600
    if crop_img.width > max_width:
        ratio = max_width / crop_img.width
        new_size = (max_width, int(crop_img.height * ratio))
        crop_img = crop_img.resize(new_size, Image.LANCZOS)
    crop_array = np.array(crop_img)
    if crop_array.size == 0:
        print("Erreur : l'image capturée est vide.")
        return []
    if crop_array.dtype != np.uint8:
        crop_array = crop_array.astype(np.uint8)
    if len(crop_array.shape) != 3 or crop_array.shape[2] != 3:
        print("Erreur : l'image n'est pas au format RGB attendu.")
        return []
    results = reader.readtext(crop_array)
    return results

def test_voice():
    say_text("Hello world, hello world, let's make this cool idea come to life!")

def show_overlay(box, screen_size):
    # Overlay : bordure noire évidée, centre transparent, clics pass-through
    import ctypes
    root = tk.Tk()
    root.attributes('-topmost', True)
    root.overrideredirect(True)
    root.attributes('-alpha', 0.7)
    root.geometry(f"{screen_size[0]}x{screen_size[1]}+0+0")
    try:
        root.wm_attributes('-disabled', True)
    except Exception:
        pass
    canvas = tk.Canvas(root, width=screen_size[0], height=screen_size[1], highlightthickness=0, bg=None)
    canvas.pack()
    x1, y1, x2, y2 = box
    canvas.create_rectangle(x1, y1, x2, y2, outline='black', width=6)
    root.lift()
    try:
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
        style = ctypes.windll.user32.GetWindowLongW(hwnd, -20)
        style |= 0x80000 | 0x20  # WS_EX_LAYERED | WS_EX_TRANSPARENT
        ctypes.windll.user32.SetWindowLongW(hwnd, -20, style)
    except Exception:
        pass
    return root

def wait_for_trigger():
    recognizer = sr.Recognizer()
    if not sr.Microphone.list_microphone_names():
        print("Aucun microphone détecté. Branchez un micro et relancez.")
        return None
    print("Microphones disponibles :", sr.Microphone.list_microphone_names())
    mic = sr.Microphone()
    # Calcul du cadre à afficher (même logique que capture_fullscreen_text)
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

# Lancement
wait_for_trigger()
