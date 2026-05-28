import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import os


class BirdsAiSupercut:
    """
    GUI-App zum intelligenten quadratischen Ausschneiden von Vögeln aus Bildern.

    Verwendet YOLOv8 (vortrainiert auf COCO) zur Erkennung von Vögeln.
    Der erkannte Vogel wird quadratisch ausgeschnitten, wobei so wenig
    Hintergrund wie möglich mitgenommen wird und keine Verzerrungen entstehen.

    COCO-Klasse 14 = 'bird'
    """

    BIRD_CLASS_ID = 14  # COCO-Klasse für 'bird'
    YOLO_MODEL = "yolov8m.pt"  # Nano-Modell (schnell, klein, ausreichend genau)
    CONFIDENCE_THRESHOLD = 0.3  # Mindest-Konfidenz für die Erkennung

    def __init__(self, root):
        self.root = root
        self.root.title("Birds AI Yolo Supercut – Vogel-Ausschnitt")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        self.root.configure(bg="#f0f0f0")

        # Interner Zustand
        self.original_image = None       # PIL Image des geladenen Bildes
        self.cropped_image = None         # PIL Image des ausgeschnittenen Vogels
        self.original_photo = None        # ImageTk für Anzeige (Original)
        self.cropped_photo = None         # ImageTk für Anzeige (Ergebnis)
        self.loaded_filepath = None       # Pfad der geladenen Datei

        # YOLOv8-Modell laden
        self.model = None

        # --- GUI aufbauen ---
        self._build_gui()

    def _build_gui(self):
        """Erstellt die GUI-Elemente."""

        # --- Titel ---
        title = tk.Label(
            self.root,
            text="Birds AI Yolo Supercut",
            font=("Arial", 16, "bold"),
            bg="#f0f0f0"
        )
        title.pack(pady=(10, 2))

        desc = tk.Label(
            self.root,
            text=(
                "Öffne ein Bild (JPEG, 800×448), um den Vogel automatisch\n"
                "zu erkennen und quadratisch auszuschneiden."
            ),
            font=("Arial", 10),
            fg="grey",
            bg="#f0f0f0",
            justify="center"
        )
        desc.pack(pady=(0, 10))

        # --- Button-Leiste ---
        button_frame = tk.Frame(self.root, bg="#f0f0f0")
        button_frame.pack(pady=5)

        self.btn_open = tk.Button(
            button_frame,
            text="📂 Bild öffnen",
            command=self.open_image,
            bg="#008CBA",
            fg="white",
            font=("Arial", 11),
            padx=20,
            pady=8,
            cursor="hand2"
        )
        self.btn_open.pack(side=tk.LEFT, padx=10)

        self.btn_save = tk.Button(
            button_frame,
            text="💾 Ergebnis speichern",
            command=self.save_result,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 11),
            padx=20,
            pady=8,
            cursor="hand2",
            state="disabled"
        )
        self.btn_save.pack(side=tk.LEFT, padx=10)

        # --- Settings-Leiste ---
        settings_frame = tk.Frame(self.root, bg="#f0f0f0")
        settings_frame.pack(pady=5)

        tk.Label(
            settings_frame,
            text="Confidence Threshold:",
            bg="#f0f0f0",
            font=("Arial", 10)
        ).pack(side=tk.LEFT, padx=5)

        self.confidence_var = tk.DoubleVar(value=self.CONFIDENCE_THRESHOLD)
        self.confidence_scale = tk.Scale(
            settings_frame,
            variable=self.confidence_var,
            from_=0.05,
            to=1.0,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            bg="#f0f0f0",
            length=200,
            highlightthickness=0
        )
        self.confidence_scale.pack(side=tk.LEFT, padx=5)
        self.confidence_scale.bind("<ButtonRelease-1>", self._on_confidence_change)

        # --- Bildanzeige-Bereich ---
        images_frame = tk.Frame(self.root, bg="#f0f0f0")
        images_frame.pack(pady=10, fill=tk.BOTH, expand=True, padx=15)

        # Linke Seite: Originalbild
        left_frame = tk.LabelFrame(
            images_frame,
            text="Originalbild",
            font=("Arial", 10, "bold"),
            bg="#f0f0f0",
            padx=5,
            pady=5
        )
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.canvas_original = tk.Canvas(
            left_frame,
            bg="#d9d9d9",
            highlightthickness=0
        )
        self.canvas_original.pack(fill=tk.BOTH, expand=True)

        # Rechte Seite: Ausgeschnittener Vogel
        right_frame = tk.LabelFrame(
            images_frame,
            text="Ausgeschnittener Vogel",
            font=("Arial", 10, "bold"),
            bg="#f0f0f0",
            padx=5,
            pady=5
        )
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        self.canvas_cropped = tk.Canvas(
            right_frame,
            bg="#d9d9d9",
            highlightthickness=0
        )
        self.canvas_cropped.pack(fill=tk.BOTH, expand=True)

        # --- Statusleiste ---
        self.status_label = tk.Label(
            self.root,
            text="Bereit – Bitte ein Bild öffnen.",
            fg="grey",
            bg="#f0f0f0",
            font=("Arial", 10)
        )
        self.status_label.pack(pady=(5, 10))

        # --- Info-Label für Erkennung ---
        self.info_label = tk.Label(
            self.root,
            text="",
            fg="#333",
            bg="#f0f0f0",
            font=("Arial", 9),
            justify="center"
        )
        self.info_label.pack(pady=(0, 5))

    def _load_model(self):
        """Lädt das YOLOv8-Modell (beim ersten Aufruf)."""
        if self.model is None:
            self.status_label.config(
                text="Lade YOLOv8-Modell (erstmaliger Download kann etwas dauern)...",
                fg="orange"
            )
            self.root.update()
            self.model = YOLO(self.YOLO_MODEL)
            self.status_label.config(text="Modell geladen.", fg="green")
            self.root.update()

    def open_image(self):
        """Öffnet einen Dateidialog zum Auswählen eines JPEG-Bildes."""
        filepath = filedialog.askopenfilename(
            title="Bild auswählen",
            filetypes=[
                ("JPEG-Bilder", "*.jpg *.jpeg"),
                ("Alle Bilddateien", "*.jpg *.jpeg *.png *.bmp *.BMP *.webp"),
                ("Alle Dateien", "*.*")
            ]
        )
        if not filepath:
            return

        try:
            self.original_image = Image.open(filepath).convert("RGB")
            self.loaded_filepath = filepath
        except Exception as e:
            messagebox.showerror("Fehler", f"Bild konnte nicht geöffnet werden:\n{e}")
            return

        # Originalbild auf Canvas anzeigen
        self._display_image_on_canvas(self.original_image, self.canvas_original, "original")

        # Ergebnis-Canvas und Status zurücksetzen
        self.canvas_cropped.delete("all")
        self.cropped_image = None
        self.cropped_photo = None
        self.btn_save.config(state="disabled")
        self.info_label.config(text="")

        w, h = self.original_image.size
        self.status_label.config(
            text=f"Bild geladen: {os.path.basename(filepath)} ({w}×{h}px) – Vogelerkennung läuft...",
            fg="blue"
        )
        self.root.update()

        # Vogelerkennung starten
        self._detect_and_crop()

    def _on_confidence_change(self, event=None):
        """Wird aufgerufen, wenn der Slider losgelassen wird."""
        if self.original_image is not None:
            self.canvas_cropped.delete("all")
            self.cropped_image = None
            self.cropped_photo = None
            self.btn_save.config(state="disabled")
            self.info_label.config(text="")
            self.status_label.config(text="Aktualisiere Erkennung...", fg="blue")
            self.root.update()
            self._detect_and_crop()

    def _display_image_on_canvas(self, pil_image, canvas, tag):
        """
        Zeigt ein PIL-Bild skaliert auf einem Canvas an.
        Das Bild wird so skaliert, dass es in den Canvas passt (aspect ratio erhalten).
        """
        canvas.update_idletasks()
        canvas_w = canvas.winfo_width()
        canvas_h = canvas.winfo_height()

        if canvas_w <= 1 or canvas_h <= 1:
            canvas_w = 400
            canvas_h = 350

        img_w, img_h = pil_image.size
        scale = min(canvas_w / img_w, canvas_h / img_h)
        new_w = max(1, int(img_w * scale))
        new_h = max(1, int(img_h * scale))

        resized = pil_image.resize((new_w, new_h), Image.LANCZOS)
        photo = ImageTk.PhotoImage(resized)

        # Referenz halten, damit GC das Bild nicht entfernt
        if tag == "original":
            self.original_photo = photo
        else:
            self.cropped_photo = photo

        canvas.delete("all")
        x = canvas_w // 2
        y = canvas_h // 2
        canvas.create_image(x, y, image=photo, anchor=tk.CENTER)

    def _detect_and_crop(self):
        """
        Erkennt Vögel im geladenen Bild mit YOLOv8 und schneidet
        den Vogel quadratisch aus.
        """
        if self.original_image is None:
            return

        # Modell laden (falls noch nicht geschehen)
        self._load_model()

        self.status_label.config(text="Vogelerkennung läuft...", fg="blue")
        self.root.update()

        # YOLOv8 Inferenz
        results = self.model(self.original_image, verbose=False)

        # Alle erkannten Vögel filtern
        bird_detections = []
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                print(f"[DEBUG] Erkannt: Klasse {cls_id}, Konfidenz {conf:.2f} (Schwellenwert: {self.confidence_var.get():.2f})")
                if cls_id == self.BIRD_CLASS_ID and conf >= self.confidence_var.get():
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    bird_detections.append({
                        "bbox": (x1, y1, x2, y2),
                        "confidence": conf
                    })

        if not bird_detections:
            # Kein Vogel erkannt
            self.status_label.config(
                text="⚠ Kein Vogel auf dem Bild erkannt!",
                fg="red"
            )
            self.info_label.config(
                text="Es wurde kein Vogel auf diesem Bild gefunden.\n"
                     "Bitte versuche ein anderes Bild.",
                fg="red"
            )
            self.canvas_cropped.delete("all")
            self.canvas_cropped.update_idletasks()
            cw = self.canvas_cropped.winfo_width()
            ch = self.canvas_cropped.winfo_height()
            self.canvas_cropped.create_text(
                cw // 2, ch // 2,
                text="Kein Vogel erkannt!",
                font=("Arial", 14, "bold"),
                fill="red"
            )
            return

        # Den Vogel mit der höchsten Konfidenz auswählen
        best = max(bird_detections, key=lambda d: d["confidence"])
        x1, y1, x2, y2 = best["bbox"]
        conf = best["confidence"]

        # Quadratischen Ausschnitt berechnen
        self.cropped_image = self._make_square_crop(x1, y1, x2, y2)

        crop_w, crop_h = self.cropped_image.size

        # Ergebnis anzeigen
        self._display_image_on_canvas(self.cropped_image, self.canvas_cropped, "cropped")
        self.btn_save.config(state="normal")

        self.status_label.config(
            text=f"✔ Vogel erkannt! (Konfidenz: {conf:.1%}) – Ausschnitt: {crop_w}×{crop_h}px",
            fg="green"
        )

        detected_count = len(bird_detections)
        if detected_count > 1:
            self.info_label.config(
                text=f"{detected_count} Vögel erkannt – der Vogel mit der höchsten "
                     f"Konfidenz ({conf:.1%}) wurde ausgewählt.",
                fg="#555"
            )
        else:
            self.info_label.config(
                text=f"1 Vogel erkannt mit {conf:.1%} Konfidenz.",
                fg="#555"
            )

    def _make_square_crop(self, x1, y1, x2, y2):
        """
        Erstellt einen quadratischen Ausschnitt um die Bounding-Box (x1, y1, x2, y2).

        Strategie:
        - Die Seitenlänge des Quadrats ist die längere Seite der Bounding-Box.
        - Das Quadrat wird zentriert um die Bounding-Box positioniert.
        - Falls das Quadrat über den Bildrand hinausgeht, wird es verschoben,
          damit es vollständig im Bild liegt.
        - Es wird KEIN Padding/Füllfarbe verwendet – nur echte Bildpixel.
        """
        img_w, img_h = self.original_image.size

        # Bounding-Box Abmessungen
        bbox_w = x2 - x1
        bbox_h = y2 - y1

        # Quadrat-Seitenlänge = längere Seite der Bounding-Box
        side = max(bbox_w, bbox_h)

        # Etwas Rand hinzufügen (5% der Seitenlänge), damit der Vogel nicht
        # direkt am Rand klebt
        margin = side * 0.05
        side = side + 2 * margin

        # Maximale Seitenlänge begrenzen auf die kürzere Bildseite,
        # damit das Quadrat ins Bild passt
        side = min(side, img_w, img_h)

        # Mittelpunkt der Bounding-Box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        # Quadrat zentriert um den Mittelpunkt
        crop_x1 = cx - side / 2
        crop_y1 = cy - side / 2
        crop_x2 = cx + side / 2
        crop_y2 = cy + side / 2

        # Sicherstellen, dass das Quadrat im Bild liegt (Verschiebung, kein Clipping)
        if crop_x1 < 0:
            crop_x2 -= crop_x1  # nach rechts verschieben
            crop_x1 = 0
        if crop_y1 < 0:
            crop_y2 -= crop_y1  # nach unten verschieben
            crop_y1 = 0
        if crop_x2 > img_w:
            crop_x1 -= (crop_x2 - img_w)  # nach links verschieben
            crop_x2 = img_w
        if crop_y2 > img_h:
            crop_y1 -= (crop_y2 - img_h)  # nach oben verschieben
            crop_y2 = img_h

        # Sicherheitscheck: Koordinaten im gültigen Bereich
        crop_x1 = max(0, crop_x1)
        crop_y1 = max(0, crop_y1)
        crop_x2 = min(img_w, crop_x2)
        crop_y2 = min(img_h, crop_y2)

        # Integer-Koordinaten
        crop_x1 = int(round(crop_x1))
        crop_y1 = int(round(crop_y1))
        crop_x2 = int(round(crop_x2))
        crop_y2 = int(round(crop_y2))

        # Exakt quadratisch machen (Rundungsfehler ausgleichen)
        final_w = crop_x2 - crop_x1
        final_h = crop_y2 - crop_y1
        if final_w != final_h:
            final_side = min(final_w, final_h)
            crop_x2 = crop_x1 + final_side
            crop_y2 = crop_y1 + final_side

        return self.original_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

    def save_result(self):
        """Speichert das ausgeschnittene Vogelbild in einem Ordner nach Wahl."""
        if self.cropped_image is None:
            messagebox.showwarning("Hinweis", "Kein Ergebnis zum Speichern vorhanden.")
            return

        # Standarddateiname aus Originaldateiname ableiten
        default_name = "vogel_ausschnitt.jpg"
        if self.loaded_filepath:
            name, ext = os.path.splitext(os.path.basename(self.loaded_filepath))
            default_name = f"{name}_supercut.jpg"

        filepath = filedialog.asksaveasfilename(
            title="Ergebnis speichern",
            defaultextension=".jpg",
            initialfile=default_name,
            filetypes=[
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("Alle Dateien", "*.*")
            ]
        )
        if not filepath:
            return

        try:
            # Qualität 95 für JPEG
            if filepath.lower().endswith(('.jpg', '.jpeg')):
                self.cropped_image.save(filepath, quality=95)
            else:
                self.cropped_image.save(filepath)

            self.status_label.config(
                text=f"✔ Gespeichert: {filepath}",
                fg="green"
            )
            messagebox.showinfo("Erfolg", f"Bild gespeichert unter:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Fehler", f"Speichern fehlgeschlagen:\n{e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = BirdsAiSupercut(root)
    root.mainloop()
