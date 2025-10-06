import sys
import os
try:
    from tkinter import Tk
except Exception as e:
    print("tkinter unavailable:", e)
    Tk = None

sys.path.append(os.path.join(os.path.dirname(__file__), 'gui'))

from gui.app import AudioPanningApp

# Center the window on the screen
def center_window(root, width=800, height=600):
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    position_top = int(screen_height / 2 - height / 2)
    position_right = int(screen_width / 2 - width / 2)
    root.geometry(f"{width}x{height}+{position_right}+{position_top}")
    
if __name__ == "__main__":
    
    root = Tk()
    root.title("VBAP Panning")
    center_window(root)
    root.configure(bg='orange')
    
    app = AudioPanningApp(master=root)
    root.mainloop()