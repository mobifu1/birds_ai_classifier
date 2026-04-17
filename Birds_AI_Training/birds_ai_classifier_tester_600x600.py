import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageFilter
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as tf_image # type: ignore
from tensorflow.keras.applications.inception_v3 import preprocess_input # type: ignore

class BirdAITesterApp:
    # =====================================================================
    # Kamera-Auflösung einstellen (Breite x Höhe in Pixel)
    # Das eingehende Bild wird zuerst auf diese Größe skaliert,
    # dann per Blur-Padding quadratisch auf 600x600 gebracht.
    # =====================================================================
    BLUR_RADIUS = 40  # Stärke des Gaussian Blur für den Hintergrund
    # =====================================================================

    def __init__(self, root):
        self.root = root
        self.root.title("Birds AI Classifier - Tester")
        self.root.geometry("900x800")
        
        # Model and Labels
        self.model_path = "my_birds_modell_600x600.keras"
        self.labels_path = "model_labels.json"
        self.model = None
        self.labels_map = {}
        
        # UI Elements
        self.top_frame = tk.Frame(root)
        self.top_frame.pack(pady=10)
        
        self.btn_load_image = tk.Button(self.top_frame, text="Bild laden und klassifizieren", command=self.load_and_classify, font=("Arial", 14), bg="#0d47a1", fg="white", padx=10, pady=5)
        self.btn_load_image.pack()
        
        # Padding-Modus Dropdown
        self.padding_frame = tk.Frame(root)
        self.padding_frame.pack(pady=5)
        
        tk.Label(self.padding_frame, text="Bildbearbeitung / Padding:", font=("Arial", 11)).pack(side=tk.LEFT, padx=(0, 5))
        
        self.padding_modes = [
            "Blur-Padding",
            "Resize 600x600",
            "Replicate / Edge-Padding",
            "Center-Crop (quadratisch)"
        ]
        self.padding_var = tk.StringVar(value=self.padding_modes[0])
        self.padding_dropdown = ttk.Combobox(
            self.padding_frame,
            textvariable=self.padding_var,
            values=self.padding_modes,
            state="readonly",
            width=25,
            font=("Arial", 11)
        )
        self.padding_dropdown.pack(side=tk.LEFT)
        
        self.lbl_status = tk.Label(root, text="Modell wird geladen...", font=("Arial", 12), fg="blue")
        self.lbl_status.pack(pady=5)
        
        self.lbl_model_path = tk.Label(root, text="Modell-Pfad: wird ermittelt...", font=("Arial", 10), fg="gray")
        self.lbl_model_path.pack(pady=2)
        
        self.img_label = tk.Label(root)
        self.img_label.pack(pady=10)
        
        self.lbl_results = tk.Label(root, text="", font=("Arial", 16, "bold"), justify="left")
        self.lbl_results.pack(pady=20)
        
        # Initialisierung im Hintergrund starten
        self.root.after(100, self.init_ai)

    def init_ai(self):
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Modell nicht gefunden: {self.model_path}")
            if not os.path.exists(self.labels_path):
                raise FileNotFoundError(f"Labels nicht gefunden: {self.labels_path}")
                
            self.model = load_model(self.model_path)
            with open(self.labels_path, 'r', encoding='utf-8') as f:
                raw_labels = json.load(f)
                self.labels_map = {int(k): v for k, v in raw_labels.items()}
                
            self.lbl_status.config(text="Modell erfolgreich geladen. Bereit.", fg="green")
            self.lbl_model_path.config(text=f"Modell-Pfad: {os.path.abspath(self.model_path)}")
        except Exception as e:
            self.lbl_status.config(text=f"Fehler beim Laden: {str(e)}", fg="red")
            messagebox.showerror("Fehler", f"Fehler bei der Initialisierung:\n{str(e)}")

    def blur_pad_to_square(self, img, target_size=600):
        """Blur-Padding: Bild quadratisch machen mit unscharfem Hintergrund.
        
        Wie bei TV-Dokumentationen, wenn ein Hochkant-Video gezeigt wird:
        Der Hintergrund ist eine stark weichgezeichnete, aufgeblasene Version
        des Originalbildes. Das skalierte Original wird mittig darübergelegt.
        So entstehen keine harten Kanten, nur natürliche Farben.
        """
        w, h = img.size

        # Schritt 1: Hintergrund erstellen – Originalbild auf Zielgröße
        #            aufblasen (verzerrt, füllt das gesamte Quadrat)
        background = img.resize((target_size, target_size), Image.LANCZOS)

        # Schritt 2: Starken Gaussian Blur auf den Hintergrund legen
        background = background.filter(ImageFilter.GaussianBlur(radius=self.BLUR_RADIUS))

        # Schritt 3: Originalbild proportional in das Quadrat einpassen
        scale = min(target_size / w, target_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        foreground = img.resize((new_w, new_h), Image.LANCZOS)

        # Schritt 4: Vordergrundbild mittig auf den unscharfen Hintergrund legen
        offset_x = (target_size - new_w) // 2
        offset_y = (target_size - new_h) // 2
        background.paste(foreground, (offset_x, offset_y))

        return background

    def resize_to_square(self, img, target_size=600):
        """Hartes Resize: Bild direkt auf 600x600 skalieren (verzerrt)."""
        return img.resize((target_size, target_size), Image.LANCZOS)

    def edge_pad_to_square(self, img, target_size=600):
        """Replicate / Edge-Padding: Die äußerste Pixelreihe wird bis zum
        Rand wiederholt, um das Bild quadratisch auf target_size zu bringen.
        """
        w, h = img.size

        # Bild proportional in das Quadrat einpassen
        scale = min(target_size / w, target_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = img.resize((new_w, new_h), Image.LANCZOS)

        # NumPy-Array für Pixel-Replikation
        arr = np.array(resized)

        # Vertikales Padding (oben/unten) – oberste/unterste Zeile replizieren
        pad_top = (target_size - new_h) // 2
        pad_bottom = target_size - new_h - pad_top

        if pad_top > 0:
            top_row = arr[0:1, :, :]  # Form: (1, W, 3)
            top_fill = np.repeat(top_row, pad_top, axis=0)
            arr = np.concatenate([top_fill, arr], axis=0)
        if pad_bottom > 0:
            bottom_row = arr[-1:, :, :]  # Form: (1, W, 3)
            bottom_fill = np.repeat(bottom_row, pad_bottom, axis=0)
            arr = np.concatenate([arr, bottom_fill], axis=0)

        # Horizontales Padding (links/rechts) – äußerste Spalte replizieren
        pad_left = (target_size - new_w) // 2
        pad_right = target_size - new_w - pad_left

        if pad_left > 0:
            left_col = arr[:, 0:1, :]  # Form: (H, 1, 3)
            left_fill = np.repeat(left_col, pad_left, axis=1)
            arr = np.concatenate([left_fill, arr], axis=1)
        if pad_right > 0:
            right_col = arr[:, -1:, :]  # Form: (H, 1, 3)
            right_fill = np.repeat(right_col, pad_right, axis=1)
            arr = np.concatenate([arr, right_fill], axis=1)

        return Image.fromarray(arr)

    def crop_to_square(self, img, target_size=600):
        """Center-Crop: Bild links und rechts gleichmäßig abschneiden,
        sodass es quadratisch wird, dann auf target_size skalieren.
        """
        w, h = img.size
        if w > h:
            # Breiter als hoch → links und rechts abschneiden
            offset = (w - h) // 2
            img = img.crop((offset, 0, offset + h, h))
        elif h > w:
            # Höher als breit → oben und unten abschneiden
            offset = (h - w) // 2
            img = img.crop((0, offset, w, offset + w))
        # Jetzt ist das Bild quadratisch → auf Zielgröße skalieren
        return img.resize((target_size, target_size), Image.LANCZOS)

    def preprocess_image(self, file_path):
        """Bild laden und durch die Vorverarbeitung führen:
        2. Je nach gewähltem Padding-Modus auf 600x600 bringen
        """
        img = Image.open(file_path).convert('RGB')
        
        # Schritt 2: Padding-Modus anwenden
        mode = self.padding_var.get()
        if mode == "Resize 600x600":
            img = self.resize_to_square(img, target_size=600)
        elif mode == "Replicate / Edge-Padding":
            img = self.edge_pad_to_square(img, target_size=600)
        elif mode == "Center-Crop (quadratisch)":
            img = self.crop_to_square(img, target_size=600)
        else:
            # Standard: Blur-Padding
            img = self.blur_pad_to_square(img, target_size=600)
        return img

    def load_and_classify(self):
        if self.model is None:
            messagebox.showwarning("Warnung", "Das Modell ist noch nicht geladen oder fehlerhaft.")
            return
            
        file_path = filedialog.askopenfilename(
            title="Bild auswählen",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
            
        self.lbl_status.config(text=f"Verarbeite: {os.path.basename(file_path)}", fg="blue")
        self.root.update()
        
        # Bild vorverarbeiten (Kamera-Resize → Blur-Padding → 600x600)
        try:
            processed_img = self.preprocess_image(file_path)
        except Exception as e:
            messagebox.showerror("Fehler", f"Konnte Bild nicht verarbeiten:\n{str(e)}")
            return

        # Vorverarbeitetes Bild in der GUI anzeigen
        try:
            self.tk_img = ImageTk.PhotoImage(processed_img)
            self.img_label.config(image=self.tk_img)
        except Exception as e:
            messagebox.showerror("Fehler", f"Konnte Bild nicht anzeigen:\n{str(e)}")
            return
            
        # Klassifizierung mit dem vorverarbeiteten Bild
        try:
            x = np.array(processed_img, dtype='float32')
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            
            preds = self.model.predict(x, verbose=0)
            preds_array = preds[0]
            
            # Top 3 Indices finden
            top3_indices = np.argsort(preds_array)[-3:][::-1]
            
            result_text = "Top 3 Ergebnisse:\n\n"
            for i, idx in enumerate(top3_indices):
                confidence = float(preds_array[idx]) * 100
                label_name = self.labels_map.get(idx, "Unbekannt").replace('_', ' ').title()
                result_text += f"{i+1}. {label_name}: {confidence:.2f}%\n"
                
            self.lbl_results.config(text=result_text, fg="#004400")
            self.lbl_status.config(text="Klassifizierung abgeschlossen.", fg="green")
            
        except Exception as e:
            self.lbl_status.config(text="Fehler bei der Klassifizierung", fg="red")
            messagebox.showerror("Fehler", f"Fehler bei der Klassifizierung:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = BirdAITesterApp(root)
    root.mainloop()
