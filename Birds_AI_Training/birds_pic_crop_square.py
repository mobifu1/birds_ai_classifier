import os
import random
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image


class BirdsPicCropSquare:
    """
    GUI-App zum quadratischen Zuschneiden von Bildern.
    - Querformat: links und rechts gleichmäßig beschnitten.
    - Hochformat: oben und unten gleichmäßig beschnitten.
    Zusätzlich wird der Dateiname bereinigt:
      - Alles vor dem ersten Unterstrich (inkl. Unterstrich) wird entfernt.
      - Eine 6-stellige Zufallszahl mit Unterstrich wird angehängt.
    """

    IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff', '.tif')

    def __init__(self, root):
        self.root = root
        self.root.title("Bilder quadratisch zuschneiden")
        self.root.geometry("750x550")
        self.root.resizable(True, True)

        # --- Titel ---
        title = tk.Label(
            root,
            text="Bilder quadratisch zuschneiden",
            font=("Arial", 14, "bold")
        )
        title.pack(pady=10)

        desc = tk.Label(
            root,
            text=(
                "Durchsucht einen Ordner (inkl. Unterordner) nach Bildern.\n"
                "Querformat: links/rechts beschnitten. Hochformat: oben/unten beschnitten.\n"
                "Optional wird der Dateiname bereinigt und eine Zufallszahl angehängt."
            ),
            justify="center"
        )
        desc.pack(pady=5)

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

        # --- Log-Bereich ---
        self.log_text = scrolledtext.ScrolledText(root, height=18, width=90, state='disabled')
        self.log_text.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)

        # --- Statusleiste ---
        self.status_label = tk.Label(root, text="Warte auf Start...", fg="grey")
        self.status_label.pack(pady=5)

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
    def clean_filename(filename):
        """
        Bereinigt den Dateinamen:
        - Entfernt alles von links bis zum ersten Unterstrich (inkl. Unterstrich),
          falls ein Unterstrich vorhanden ist.
        - Hängt eine 6-stellige Zufallszahl mit Unterstrich an.

        Beispiel:
          'PREFIX_vogel.jpg' -> 'vogel_384729.jpg'
          'bild_ohne_prefix.png' -> 'ohne_prefix_194827.png'
          'kein_unterstrich.jpg' -> 'unterstrich_582041.jpg' (Achtung: auch hier wird entfernt!)
          'ohneUnterstrich.jpg' -> 'ohneUnterstrich_629184.jpg'
        """
        name, ext = os.path.splitext(filename)

        # Alles vor dem ersten Unterstrich entfernen (inkl. Unterstrich)
        if '_' in name:
            name = name.split('_', 1)[1]

        # 6-stellige Zufallszahl anhängen
        random_suffix = random.randint(100000, 999999)
        new_name = f"{name}_{random_suffix}{ext}"

        return new_name

    @staticmethod
    def crop_to_square(image):
        """
        Schneidet ein Bild quadratisch zu:
        - Querformat (breiter als hoch): links und rechts gleichmäßig abschneiden.
        - Hochformat (höher als breit): oben und unten gleichmäßig abschneiden.
        """
        width, height = image.size

        if width == height:
            # Bild ist bereits quadratisch -> nichts schneiden
            return image

        if width > height:
            # Querformat: Links und rechts gleichmäßig abschneiden
            crop_total = width - height
            crop_each_side = crop_total // 2

            left = crop_each_side
            top = 0
            right = width - (crop_total - crop_each_side)
            bottom = height
        else:
            # Hochformat: Oben und unten gleichmäßig abschneiden
            crop_total = height - width
            crop_each_side = crop_total // 2

            left = 0
            top = crop_each_side
            right = width
            bottom = height - (crop_total - crop_each_side)

        return image.crop((left, top, right, bottom))

    def process_images(self):
        """Hauptlogik: Ordner auswählen, Bilder zuschneiden und umbenennen."""
        root_path = filedialog.askdirectory(title="Ordner mit Bildern auswählen")
        if not root_path:
            return

        self.clear_log()

        self.log(f"Quell-Ordner: {root_path}")
        self.log("=" * 70)

        count_total = 0
        count_cropped = 0
        count_skipped = 0
        count_errors = 0

        for current_root, dirs, files in os.walk(root_path):
            for filename in sorted(files):
                if not filename.lower().endswith(self.IMAGE_EXTENSIONS):
                    continue

                file_path = os.path.join(current_root, filename)
                count_total += 1

                try:
                    with Image.open(file_path) as img:
                        width, height = img.size

                        if width == height:
                            count_skipped += 1
                            self.log(
                                f"[ÜBERSPRUNGEN] {filename}  "
                                f"({width}x{height} - bereits quadratisch)"
                            )
                            continue

                        # Bild quadratisch zuschneiden
                        cropped_img = self.crop_to_square(img)
                        new_w, new_h = cropped_img.size

                        # Neuen Dateinamen erzeugen
                        new_filename = self.clean_filename(filename)

                        # Sicherstellen, dass der neue Dateiname nicht existiert
                        new_path = os.path.join(current_root, new_filename)
                        while os.path.exists(new_path):
                            new_filename = self.clean_filename(filename)
                            new_path = os.path.join(current_root, new_filename)

                        # Zugeschnittenes Bild speichern
                        cropped_img.save(new_path, quality=95)

                    # Originaldatei löschen
                    os.remove(file_path)
                    count_cropped += 1

                    self.log(
                        f"[ZUGESCHNITTEN] {filename} -> {new_filename}  "
                        f"({width}x{height} -> {new_w}x{new_h})"
                    )

                except Exception as e:
                    count_errors += 1
                    self.log(f"[FEHLER] {filename}: {e}")

        # Zusammenfassung
        self.log("")
        self.log("=" * 70)
        self.log("ZUSAMMENFASSUNG:")
        self.log(f"  Bilder insgesamt gefunden:   {count_total}")
        self.log(f"  Zugeschnitten & umbenannt:   {count_cropped}")
        self.log(f"  Übersprungen (quadratisch):   {count_skipped}")
        if count_errors > 0:
            self.log(f"  Fehler:                      {count_errors}")
        self.log("=" * 70)

        self.status_label.config(text="Fertig!", fg="green")

        messagebox.showinfo(
            "Abschluss",
            f"Fertig!\n\n"
            f"Bilder gefunden: {count_total}\n"
            f"Zugeschnitten: {count_cropped}\n"
            f"Übersprungen: {count_skipped}\n"
            f"Fehler: {count_errors}"
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = BirdsPicCropSquare(root)
    root.mainloop()
