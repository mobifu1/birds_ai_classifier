import os
import shutil
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

# --- KONFIGURATION ---
DEST_FOLDER_NAME = "birds_grabber"
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}

class GrabberGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Birds Grabber (Bild-Sammler)")
        self.root.geometry("600x500")
        
        self.is_running = False
        self.thread = None

        # Titel
        tk.Label(root, text="Bilder einsammeln (Rekursiv)", font=("Arial", 16, "bold")).pack(pady=10)

        # Quell-Ordner Auswahl
        frame_src = tk.LabelFrame(root, text="Quell-Ordner (Woher?)", padx=10, pady=10)
        frame_src.pack(pady=5, padx=20, fill="x")

        self.entry_src = tk.Entry(frame_src, width=45)
        self.entry_src.pack(side=tk.LEFT, padx=5)
        tk.Button(frame_src, text="Durchsuchen...", command=self.select_folder).pack(side=tk.LEFT)

        # Ziel-Ordner Info
        frame_dest = tk.LabelFrame(root, text="Ziel-Ordner (Wohin?)", padx=10, pady=10)
        frame_dest.pack(pady=5, padx=20, fill="x")
        
        self.dest_path = os.path.join(os.getcwd(), DEST_FOLDER_NAME)
        tk.Label(frame_dest, text=f"Bilder werden kopiert nach:\n{self.dest_path}", fg="blue", justify=tk.LEFT).pack(anchor=tk.W)
        tk.Button(frame_dest, text="Ziel-Ordner öffnen", command=self.open_dest_folder).pack(anchor=tk.W, pady=5)

        # Buttons
        frame_controls = tk.Frame(root)
        frame_controls.pack(pady=15)
        
        self.btn_start = tk.Button(frame_controls, text="Kopiervorgang Starten", command=self.toggle_process, bg="#dddddd", width=25, height=2)
        self.btn_start.pack()

        # Log Bereich
        tk.Label(root, text="Verlauf:").pack(anchor=tk.W, padx=20)
        self.log_text = tk.Text(root, height=12, width=70, state='disabled')
        self.log_text.pack(pady=5, padx=20)

        # Fenster-Close Handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def select_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.entry_src.delete(0, tk.END)
            self.entry_src.insert(0, path)

    def open_dest_folder(self):
        if not os.path.exists(self.dest_path):
            os.makedirs(self.dest_path)
            self.log("Zielordner wurde erstellt.")
        
        try:
            if os.name == 'nt':  # Windows
                os.startfile(self.dest_path)
            else:  # macOS / Linux
                import subprocess
                opener = 'open' if sys.platform == 'darwin' else 'xdg-open'
                subprocess.call([opener, self.dest_path])
        except Exception as e:
            self.log(f"Fehler beim Öffnen: {e}")

    def toggle_process(self):
        if self.is_running:
            self.is_running = False
            self.btn_start.config(text="Stoppe...", state="disabled")
        else:
            src = self.entry_src.get()
            if not src or not os.path.exists(src):
                messagebox.showwarning("Fehler", "Bitte einen gültigen Quell-Ordner wählen.")
                return
            
            self.is_running = True
            self.btn_start.config(text="Vorgang Stoppen", bg="#ffcccc")
            self.thread = threading.Thread(target=self.run_copy_process, args=(src,), daemon=True)
            self.thread.start()

    def run_copy_process(self, source_folder):
        self.log(f"Starte Suche in: {source_folder}")
        self.log(f"Ziel: {self.dest_path}")
        
        if not os.path.exists(self.dest_path):
            os.makedirs(self.dest_path)

        count_copied = 0
        count_errors = 0
        
        # Alle Dateien rekursiv finden
        files_to_copy = []
        try:
            for root_dir, dirs, files in os.walk(source_folder):
                if not self.is_running: break
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext in IMAGE_EXTENSIONS:
                        full_path = os.path.join(root_dir, file)
                        files_to_copy.append(full_path)
        except Exception as e:
            self.log(f"Fehler beim Scannen: {e}")
            self.reset_ui()
            return

        self.log(f"{len(files_to_copy)} Bilder gefunden. Starte Kopieren...")

        for src_file in files_to_copy:
            if not self.is_running:
                self.log("Vorgang durch Benutzer abgebrochen.")
                break

            try:
                filename = os.path.basename(src_file)
                dest_file = os.path.join(self.dest_path, filename)

                # Kollisions-Check: Wenn Datei schon existiert, umbenennen (z.B. bild_1.jpg)
                if os.path.exists(dest_file):
                    base, extension = os.path.splitext(filename)
                    counter = 1
                    while os.path.exists(os.path.join(self.dest_path, f"{base}_{counter}{extension}")):
                        counter += 1
                    dest_file = os.path.join(self.dest_path, f"{base}_{counter}{extension}")

                shutil.copy2(src_file, dest_file)
                count_copied += 1
                
                # Log nicht bei jeder Datei fluten, sondern alle 10 oder bei Fehlern
                if count_copied % 10 == 0:
                    self.log(f"{count_copied} Bilder kopiert...")
                    
            except Exception as e:
                self.log(f"Fehler bei {src_file}: {e}")
                count_errors += 1

        self.log("-" * 30)
        self.log(f"FERTIG! Kopiert: {count_copied}, Fehler: {count_errors}")
        self.reset_ui()

    def reset_ui(self):
        self.is_running = False
        self.btn_start.config(text="Kopiervorgang Starten", bg="#dddddd", state="normal")

    def log(self, message):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')

    def on_close(self):
        self.is_running = False
        self.root.destroy()
        os._exit(0)

if __name__ == "__main__":
    root = tk.Tk()
    app = GrabberGUI(root)
    root.mainloop()