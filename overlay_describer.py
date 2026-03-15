import time
import ctypes
from ctypes import wintypes
from capture_vrchat import find_vrchat_window, get_window_rect, capture_window
from PIL import Image
import tkinter as tk
import threading
import os

GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020

def dominant_color_name(image: Image.Image) -> str:
    small = image.resize((50, 50)).convert('RGB')
    result = small.getcolors(50*50)
    result.sort(key=lambda x: x[0], reverse=True)
    rgb = result[0][1]
    # simple mapping
    r, g, b = rgb
    if r > 200 and g > 200 and b > 200:
        return 'white'
    if r < 50 and g < 50 and b < 50:
        return 'black'
    if r > g and r > b:
        return 'red' if r - max(g,b) > 20 else 'brown'
    if g > r and g > b:
        return 'green'
    if b > r and b > g:
        return 'blue'
    return f'rgb({r},{g},{b})'

def brightness_name(image: Image.Image) -> str:
    gray = image.convert('L')
    mean = sum(gray.getdata())/ (gray.size[0]*gray.size[1])
    return 'bright' if mean > 127 else 'dark'

class OverlayApp:
    def __init__(self, hwnd, rect):
        self.hwnd = hwnd
        left, top, right, bottom = rect
        self.width = right - left
        self.height = bottom - top
        self.left = left
        self.top = top

        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.geometry(f"{self.width}x{self.height}+{self.left}+{self.top}")
        # pick a color key that is unlikely to appear
        self.key_color = 'magenta'
        self.root.configure(bg=self.key_color)
        # make transparent color
        self.root.wm_attributes('-transparentcolor', self.key_color)
        self.root.wm_attributes('-topmost', True)

        self.label = tk.Label(self.root, text='', font=('Segoe UI', 20, 'bold'), fg='white', bg=self.key_color)
        self.label.place(relx=0.5, rely=0.05, anchor='n')

        # make window click-through
        self.make_clickthrough()

        self.running = True

    def make_clickthrough(self):
        hwnd = wintypes.HWND(self.root.winfo_id())
        user32 = ctypes.windll.user32
        gwl = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, gwl | WS_EX_LAYERED | WS_EX_TRANSPARENT)

    def update_text(self, text):
        self.label.config(text=text)

    def run(self):
        self.root.mainloop()

def worker(app: OverlayApp):
    captures_dir = os.path.join(os.getcwd(), 'captures')
    os.makedirs(captures_dir, exist_ok=True)
    while app.running:
        # capture
        left, top, right, bottom = app.left, app.top, app.left+app.width, app.top+app.height
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        path = os.path.join(captures_dir, f'overlay_{timestamp}.png')
        capture_window((left, top, right, bottom), path)
        try:
            img = Image.open(path)
            color = dominant_color_name(img)
            bright = brightness_name(img)
            desc = f'{timestamp} — {bright}, mostly {color} — {img.width}x{img.height}'
            app.update_text(desc)
        except Exception as e:
            app.update_text(f'Error describing image: {e}')

        time.sleep(1.0)

def main():
    hwnd, title = find_vrchat_window()
    if not hwnd:
        print('VRChat window not found; ensure VRChat is running and visible.')
        return
    rect = get_window_rect(hwnd)
    app = OverlayApp(hwnd, rect)
    t = threading.Thread(target=worker, args=(app,), daemon=True)
    t.start()
    app.run()

if __name__ == '__main__':
    main()
