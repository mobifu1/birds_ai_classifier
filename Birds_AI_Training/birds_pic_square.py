import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image


class BirdsPicSquareChecker:
    """
    GUI-App zum Aussortieren von nicht-quadratischen Bildern.
    Bilder, die über einen einstellbaren Prozentwert vom perfekten
    Quadrat abweichen, werden in einen Unterordner 'rectangle_pics' verschoben.
    """

    IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff', '.tif')
    TARGET_FOLDER_NAME = "rectangle_pics"

    def __init__(self, root):
        self.root = root
        self.root.title("Bilder-Quadrat-Prüfer")
        self.root.geometry("700x550")
        self.root.resizable(True, True)

        # --- Titel ---
        title = tk.Label(
            root,
            text="Nicht-quadratische Bilder aussortieren",
            font=("Arial", 14, "bold")
        )
        title.pack(pady=10)

        desc = tk.Label(
            root,
            text=(
                "Durchsucht einen Ordner (inkl. Unterordner) nach Bildern und verschiebt\n"
                "alle Bilder, die zu weit vom perfekten Quadrat abweichen,\n"
                f"in den Unterordner '{self.TARGET_FOLDER_NAME}'."
            ),
            justify="center"
        )
        desc.pack(pady=5)

        # --- Toleranz-Einstellung ---
        tolerance_frame = tk.Frame(root)
        tolerance_frame.pack(pady=10)

        tk.Label(
            tolerance_frame,
            text="Erlaubte Abweichung vom Quadrat (%):",
            font=("Arial", 11)
        ).pack(side=tk.LEFT, padx=5)

        self.tolerance_var = tk.IntVar(value=15)
        self.tolerance_spinbox = tk.Spinbox(
            tolerance_frame,
            from_=0,
            to=100,
            width=5,
            textvariable=self.tolerance_var,
            font=("Arial", 11)
        )
        self.tolerance_spinbox.pack(side=tk.LEFT, padx=5)

        tk.Label(
            tolerance_frame,
            text="%",
            font=("Arial", 11)
        ).pack(side=tk.LEFT)

        # --- Erklärung ---
        explanation = tk.Label(
            root,
            text=(
                "Beispiel: Bei 10% werden Bilder verschoben, deren kürzere Seite\n"
                "mehr als 10% kürzer ist als die längere Seite.\n"
                "0% = nur perfekt quadratische Bilder behalten | 100% = alle behalten"
            ),
            fg="grey",
            font=("Arial", 9)
        )
        explanation.pack(pady=5)

        # --- Buttons ---
        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)

        self.btn_select = tk.Button(
            button_frame,
            text="Ordner auswählen & Starten",
            command=self.process_images,
            bg="#008CBA",
            fg="white",
            font=("Arial", 11),
            padx=20,
            pady=10
        )
        self.btn_select.pack(side=tk.LEFT, padx=5)

        self.btn_undo = tk.Button(
            button_frame,
            text="Rückgängig machen",
            command=self.undo_move,
            bg="#FF9800",
            fg="white",
            font=("Arial", 11),
            padx=20,
            pady=10,
            state='disabled'
        )
        self.btn_undo.pack(side=tk.LEFT, padx=5)

        # --- Log-Bereich ---
        self.log_text = scrolledtext.ScrolledText(root, height=15, width=80, state='disabled')
        self.log_text.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)

        # --- Statusleiste ---
        self.status_label = tk.Label(root, text="Warte auf Start...", fg="grey")
        self.status_label.pack(pady=5)

        # Für Rückgängig-Funktion: Liste der verschobenen Dateien
        self.moved_files = []

    def log(self, message):
        """Schreibt Nachrichten in das Log-Fenster."""
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')
        self.root.update()

    def clear_log(self):
        """Leert das Log-Fenster."""
        self.log_text.config(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state='disabled')

    @staticmethod
    def calculate_deviation(width, height):
        """
        Berechnet die Abweichung eines Bildes vom perfekten Quadrat in Prozent.

        Formel: ((längere Seite - kürzere Seite) / längere Seite) * 100

        Beispiele:
        - 100x100 -> 0.0%   (perfektes Quadrat)
        - 110x100 -> 9.09%  (fast quadratisch)
        - 200x100 -> 50.0%  (stark rechteckig)
        """
        if width == 0 or height == 0:
            return 100.0

        longer = max(width, height)
        shorter = min(width, height)
        deviation = ((longer - shorter) / longer) * 100.0
        return deviation

    def process_images(self):
        """Hauptlogik: Ordner auswählen, Bilder prüfen und verschieben."""
        root_path = filedialog.askdirectory(title="Ordner mit Bildern auswählen")
        if not root_path:
            return

        tolerance = self.tolerance_var.get()
        self.clear_log()
        self.moved_files = []
        self.btn_undo.config(state='disabled')

        self.log(f"Quell-Ordner: {root_path}")
        self.log(f"Erlaubte Abweichung: {tolerance}%")
        self.log("=" * 60)

        count_total = 0
        count_moved = 0
        count_kept = 0
        count_errors = 0

        for current_root, dirs, files in os.walk(root_path):
            # Den rectangle_pics-Ordner selbst überspringen
            if self.TARGET_FOLDER_NAME in current_root.split(os.sep):
                continue

            for filename in sorted(files):
                if not filename.lower().endswith(self.IMAGE_EXTENSIONS):
                    continue

                file_path = os.path.join(current_root, filename)
                count_total += 1

                try:
                    with Image.open(file_path) as img:
                        width, height = img.size

                    deviation = self.calculate_deviation(width, height)

                    if deviation > tolerance:
                        # Zielordner erstellen (rectangle_pics im selben Verzeichnis)
                        target_dir = os.path.join(current_root, self.TARGET_FOLDER_NAME)
                        os.makedirs(target_dir, exist_ok=True)

                        # Datei verschieben
                        target_path = os.path.join(target_dir, filename)

                        # Falls gleichnamige Datei existiert, Suffix anhängen
                        if os.path.exists(target_path):
                            name, ext = os.path.splitext(filename)
                            counter = 1
                            while os.path.exists(target_path):
                                target_path = os.path.join(target_dir, f"{name}_{counter}{ext}")
                                counter += 1

                        shutil.move(file_path, target_path)
                        self.moved_files.append((target_path, file_path))
                        count_moved += 1

                        self.log(
                            f"[VERSCHOBEN] {filename}  "
                            f"({width}x{height}, Abweichung: {deviation:.1f}%)"
                        )
                    else:
                        count_kept += 1

                except Exception as e:
                    count_errors += 1
                    self.log(f"[FEHLER] {filename}: {e}")

        # Zusammenfassung
        self.log("")
        self.log("=" * 60)
        self.log("ZUSAMMENFASSUNG:")
        self.log(f"  Geprüfte Bilder insgesamt:  {count_total}")
        self.log(f"  Behalten (quadratisch):     {count_kept}")
        self.log(f"  Verschoben (rechteckig):    {count_moved}")
        if count_errors > 0:
            self.log(f"  Fehler:                     {count_errors}")
        self.log("=" * 60)

        self.status_label.config(text="Fertig!", fg="green")

        if count_moved > 0:
            self.btn_undo.config(state='normal')

        messagebox.showinfo(
            "Abschluss",
            f"Fertig!\n\n"
            f"Geprüft: {count_total} Bilder\n"
            f"Behalten: {count_kept}\n"
            f"Verschoben: {count_moved}\n"
            f"Fehler: {count_errors}"
        )

    def undo_move(self):
        """Macht die letzte Verschiebung rückgängig."""
        if not self.moved_files:
            messagebox.showinfo("Info", "Nichts zum Rückgängigmachen.")
            return

        confirm = messagebox.askyesno(
            "Rückgängig machen",
            f"{len(self.moved_files)} Datei(en) zurück verschieben?"
        )
        if not confirm:
            return

        self.clear_log()
        self.log("Rückgängig machen...")
        self.log("=" * 60)

        restored = 0
        errors = 0

        for target_path, original_path in self.moved_files:
            try:
                if os.path.exists(target_path):
                    # Sicherstellen, dass der Original-Ordner existiert
                    os.makedirs(os.path.dirname(original_path), exist_ok=True)
                    shutil.move(target_path, original_path)
                    restored += 1
                    self.log(f"[ZURÜCK] {os.path.basename(original_path)}")
                else:
                    self.log(f"[ÜBERSPRUNGEN] Datei nicht mehr vorhanden: {target_path}")
            except Exception as e:
                errors += 1
                self.log(f"[FEHLER] {os.path.basename(target_path)}: {e}")

        # Leere rectangle_pics-Ordner aufräumen
        cleaned_dirs = set()
        for target_path, _ in self.moved_files:
            target_dir = os.path.dirname(target_path)
            if target_dir not in cleaned_dirs and os.path.exists(target_dir):
                try:
                    if not os.listdir(target_dir):
                        os.rmdir(target_dir)
                        self.log(f"[AUFGERÄUMT] Leerer Ordner entfernt: {target_dir}")
                except Exception:
                    pass
                cleaned_dirs.add(target_dir)

        self.moved_files = []
        self.btn_undo.config(state='disabled')

        self.log("")
        self.log("=" * 60)
        self.log(f"Wiederhergestellt: {restored} | Fehler: {errors}")
        self.log("=" * 60)

        self.status_label.config(text="Rückgängig gemacht!", fg="orange")
        messagebox.showinfo("Fertig", f"{restored} Datei(en) wiederhergestellt.")


if __name__ == "__main__":
    root = tk.Tk()
    app = BirdsPicSquareChecker(root)
    root.mainloop()
