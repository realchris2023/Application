import sys
import os
from pathlib import Path


def _ensure_project_virtualenv():
    """Restart the app using the project venv interpreter if we're outside it."""
    repo_root = Path(__file__).resolve().parents[1]
    venv_dir = repo_root / "venv"
    if not venv_dir.exists():
        return  # No managed venv to enforce

    if os.name == "nt":
        python_path = venv_dir / "Scripts" / "python.exe"
    else:
        python_path = venv_dir / "bin" / "python"

    if not python_path.exists():
        return  # Venv present but missing interpreter; let the import fail normally

    try:
        already_using_venv = os.path.samefile(sys.executable, python_path)
    except FileNotFoundError:
        already_using_venv = False
    except AttributeError:
        # Fallback for platforms without samefile (very rare)
        already_using_venv = Path(sys.executable).resolve() == python_path.resolve()

    if not already_using_venv:
        print(f"[VBAP] Restarting with project virtualenv interpreter at {python_path}")
        try:
            sys.stdout.flush()
        except Exception:
            pass
        os.execv(str(python_path), [str(python_path), *sys.argv])


_ensure_project_virtualenv()

try:
    from tkinter import Tk
except Exception as e:
    print("tkinter unavailable:", e)
    Tk = None

sys.path.append(os.path.join(os.path.dirname(__file__), 'gui'))

from gui.app import AudioPanningApp

# Center the window on the screen
def center_window(root, width=1600, height=1200):
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
