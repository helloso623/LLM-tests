"""
build.py — package memorix as a standalone Windows app.

  python build.py

Output: dist/memorix/memorix.exe   (zip the folder to share)

Requires PyInstaller:
  pip install pyinstaller
"""

import os, sys, subprocess

HERE   = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")

args = [
    sys.executable, "-m", "PyInstaller",
    "--name", "memorix",
    "--noconfirm",
    "--onedir",
    "--windowed",
    # bundle ambient audio files
    f"--add-data={ASSETS}{os.pathsep}assets",
    # collect native libs (portaudio, libsndfile)
    "--collect-all", "sounddevice",
    "--collect-all", "soundfile",
    # explicit hidden imports for dynamic imports inside functions
    "--hidden-import", "core.frequencies",
    "--hidden-import", "core.synth",
    "--hidden-import", "core.ambient",
    "--hidden-import", "output.sound",
    "--hidden-import", "numpy",
    os.path.join(HERE, "main.py"),
]

print("building memorix…")
subprocess.run(args, check=True, cwd=HERE)
print("\ndone → dist/memorix/memorix.exe")
print("zip the dist/memorix/ folder to share.")
