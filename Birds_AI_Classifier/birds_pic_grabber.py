import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
import shutil
from pathlib import Path
import threading

class BirdGrabberApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Birds Pic Grabber")
        self.root.geometry("600x450")

        # --- Variablen ---
        self.source_folder = tk.StringVar()
        self.target_folder_name = "pic_grabber"
        
        # --- GUI Elemente erstellen ---
        self.create_widgets()

    def create_widgets(self):
        # 1. Bereich: Ordner Auswahl
        frame_top = tk.Frame(self.root, padx=10, pady=10)
        frame_top.pack(fill="x")

        lbl_info = tk.Label(frame_top, text="1. Wähle den Ordner, der durchsucht werden soll:", font=("Arial", 10, "bold"))
        lbl_info.pack(anchor="w")

        # Container für Eingabefeld und Button
        frame_select = tk.Frame(frame_top)
        frame_select.pack(fill="x", pady=5)

        entry_path = tk.Entry(frame_select, textvariable=self.source_folder, width=50)
        entry_path.pack(side="left", fill="x", expand=True, padx=(0, 5))

        btn_browse = tk.Button(frame_select, text="Ordner wählen...", command=self.select_folder)
        btn_browse.pack(side="right")

        # 2. Bereich: Start Button
        frame_mid = tk.Frame(self.root, padx=10, pady=10)
        frame_mid.pack(fill="x")

        self.btn_start = tk.Button(frame_mid, text="Bilder grabben & verschieben", command=self.start_thread, bg="#4CAF50", fg="white", font=("Arial", 11, "bold"), height=2)
        self.btn_start.pack(fill="x")

        # 3. Bereich: Log-Fenster (Ausgabe)
        frame_bottom = tk.Frame(self.root, padx=10, pady=10)
        frame_bottom.pack(fill="both", expand=True)

        lbl_log = tk.Label(frame_bottom, text="Status-Protokoll:", anchor="w")
        lbl_log.pack(fill="x")

        self.log_text = scrolledtext.ScrolledText(frame_bottom, state='disabled', height=10)
        self.log_text.pack(fill="both", expand=True)

    def select_folder(self):
        """Öffnet den Dialog zur Ordnerwahl."""
        folder = filedialog.askdirectory()
        if folder:
            self.source_folder.set(folder)

    def log(self, message):
        """Hilfsfunktion, um Text in das Log-Fenster zu schreiben."""
        self.log_text.config(state='normal') # Schreibschutz aufheben
        
        # Begrenzung auf 1000 Zeilen, um Speicherüberlauf zu vermeiden
        line_count = int(self.log_text.index('end-1c').split('.')[0])
        if line_count > 1000:
            self.log_text.delete('1.0', '2.0')

        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END) # Automatisch nach unten scrollen
        self.log_text.config(state='disabled') # Schreibschutz wieder aktivieren

    def get_unique_filename(self, destination_folder, filename):
        """Verhindert das Überschreiben von Dateien."""
        base_name = Path(filename).stem
        extension = Path(filename).suffix
        counter = 1
        new_filename = filename
        
        while (destination_folder / new_filename).exists():
            new_filename = f"{base_name}_{counter}{extension}"
            counter += 1
        return new_filename

    def start_thread(self):
        """Startet den Prozess in einem separaten Thread, damit die GUI nicht einfriert."""
        source = self.source_folder.get()
        if not source:
            messagebox.showwarning("Achtung", "Bitte wähle zuerst einen Quellordner aus!")
            return
        
        # Button deaktivieren, damit man nicht doppelt klickt
        self.btn_start.config(state="disabled", text="Arbeite...")
        
        # Thread starten
        worker_thread = threading.Thread(target=self.run_process, args=(source,))
        worker_thread.start()

    def run_process(self, source_path_str):
        """Die eigentliche Logik (läuft im Hintergrund)."""
        source_path = Path(source_path_str)
        target_path = Path.cwd() / self.target_folder_name
        
        self.log(f"--- Start ---")
        self.log(f"Quelle: {source_path}")
        
        # Zielordner erstellen
        if not target_path.exists():
            target_path.mkdir()
            self.log(f"Zielordner erstellt: {target_path}")
        else:
            self.log(f"Zielordner existiert bereits: {target_path}")

        # Sicherheitscheck
        if source_path == target_path:
            self.log("FEHLER: Quell- und Zielordner sind identisch!")
            self.btn_start.config(state="normal", text="Bilder grabben & verschieben")
            return

        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.raw'}
        moved_count = 0
        errors = 0

        self.log("Suche läuft...")

        # Rekursive Suche
        for file_path in source_path.rglob('*'):
            if file_path.is_file():
                if file_path.suffix.lower() in image_extensions:
                    # Sicherstellen, dass wir keine Bilder aus dem Zielordner verschieben 
                    # (falls der Zielordner IM Quellordner liegt)
                    if target_path in file_path.parents:
                        continue

                    unique_name = self.get_unique_filename(target_path, file_path.name)
                    destination_file = target_path / unique_name
                    
                    try:
                        shutil.move(str(file_path), str(destination_file))
                        self.log(f"Verschoben: {file_path.name} -> {unique_name}")
                        moved_count += 1
                    except Exception as e:
                        self.log(f"FEHLER bei {file_path.name}: {e}")
                        errors += 1

        self.log(f"--- Fertig ---")
        self.log(f"Gesamt verschoben: {moved_count}")
        if errors > 0:
            self.log(f"Fehler aufgetreten: {errors}")
        
        messagebox.showinfo("Fertig", f"Prozess abgeschlossen!\n{moved_count} Bilder verschoben.")
        
        # Button wieder aktivieren
        self.btn_start.config(state="normal", text="Bilder grabben & verschieben")

if __name__ == "__main__":
    root = tk.Tk()
    app = BirdGrabberApp(root)
    root.mainloop()