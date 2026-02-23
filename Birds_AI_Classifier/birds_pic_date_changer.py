import tkinter as tk
from tkinter import filedialog, messagebox
import os
import time
from datetime import datetime
import platform

# Versuchen, das Windows-spezifische Modul für das Erstelldatum zu laden
try:
    import win32_setctime
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False

class DateChangerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bild-Datum Änderer")
        self.root.geometry("400x350")
        self.root.configure(padx=20, pady=20)

        self.selected_file_path = None

        # --- GUI Elemente (Widgets) erstellen ---
        
        # 1. Datei auswählen Button
        self.btn_select = tk.Button(root, text="Bild auswählen (.jpg / .png)", command=self.select_file)
        self.btn_select.pack(pady=10)

        # 2. Label zur Anzeige der ausgewählten Datei
        self.lbl_file = tk.Label(root, text="Keine Datei ausgewählt", wraplength=350, fg="gray")
        self.lbl_file.pack(pady=5)

        # 3. Eingabefeld für das Datum
        tk.Label(root, text="Neues Datum (TT.MM.JJJJ):").pack(pady=(15, 0))
        self.entry_date = tk.Entry(root, width=20)
        self.entry_date.insert(0, datetime.now().strftime("%d.%m.%Y")) # Aktuelles Datum als Standard
        self.entry_date.pack()

        # 4. Eingabefeld für die Uhrzeit
        tk.Label(root, text="Neue Uhrzeit (HH:MM:SS):").pack(pady=(10, 0))
        self.entry_time = tk.Entry(root, width=20)
        self.entry_time.insert(0, datetime.now().strftime("%H:%M:%S")) # Aktuelle Zeit als Standard
        self.entry_time.pack()

        # 5. Button zum Ausführen der Änderung
        self.btn_apply = tk.Button(root, text="Datum & Uhrzeit aktualisieren", command=self.change_dates, bg="#4CAF50", fg="white")
        self.btn_apply.pack(pady=25)

    def select_file(self):
        """Öffnet einen Dialog, um ein Bild auszuwählen."""
        file_path = filedialog.askopenfilename(
            title="Wähle ein Bild",
            filetypes=[("Bilder", "*.jpg *.jpeg *.png"), ("Alle Dateien", "*.*")]
        )
        if file_path:
            self.selected_file_path = file_path
            # Zeigt nur den Dateinamen im Label an, um Platz zu sparen
            self.lbl_file.config(text=os.path.basename(file_path), fg="black")

    def change_dates(self):
        """Liest die Eingaben aus und wendet die neuen Zeitstempel auf die Datei an."""
        if not self.selected_file_path:
            messagebox.showwarning("Achtung", "Bitte wähle zuerst ein Bild aus!")
            return

        date_str = self.entry_date.get().strip()
        time_str = self.entry_time.get().strip()

        try:
            # Kombiniere Datum und Zeit und wandle es in ein Python-Datetime-Objekt um
            datetime_str = f"{date_str} {time_str}"
            new_datetime = datetime.strptime(datetime_str, "%d.%m.%Y %H:%M:%S")
            
            # Wandle das Datetime-Objekt in einen UNIX-Zeitstempel um (Sekunden seit 1970)
            timestamp = new_datetime.timestamp()

            # 1. Änderungsdatum (und Zugriffsdatum) für alle Betriebssysteme ändern
            os.utime(self.selected_file_path, (timestamp, timestamp))

            # 2. Erstelldatum ändern (Spezifisch für Windows)
            if platform.system() == "Windows":
                if WIN32_AVAILABLE:
                    win32_setctime.setctime(self.selected_file_path, timestamp)
                else:
                    messagebox.showinfo("Hinweis", "Das Änderungsdatum wurde aktualisiert.\n\nUm auch das Erstelldatum auf Windows zu ändern, installiere bitte das Modul:\npip install win32-setctime")
                    return

            messagebox.showinfo("Erfolg", "Das Datum wurde erfolgreich geändert!")

        except ValueError:
            messagebox.showerror("Fehler", "Falsches Format! Bitte benutze TT.MM.JJJJ für das Datum und HH:MM:SS für die Uhrzeit.")
        except Exception as e:
            messagebox.showerror("Fehler", f"Ein unerwarteter Fehler ist aufgetreten:\n{str(e)}")

# --- Hauptprogramm starten ---
if __name__ == "__main__":
    # Erstellt das Hauptfenster
    root = tk.Tk()
    # Initialisiert unsere App-Klasse
    app = DateChangerApp(root)
    # Startet die Endlosschleife der GUI, damit das Fenster offen bleibt
    root.mainloop()