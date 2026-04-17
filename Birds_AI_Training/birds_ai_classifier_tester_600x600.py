import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox
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
    CAMERA_X = 800   # Breite der Kamera in Pixel
    CAMERA_Y = 448   # Höhe der Kamera in Pixel
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

    def preprocess_image(self, file_path):
        """Bild laden und durch die Vorverarbeitung führen:
        1. Auf Kamera-Auflösung skalieren (CAMERA_X × CAMERA_Y)
        2. Per Blur-Padding auf 600x600 bringen
        """
        img = Image.open(file_path).convert('RGB')
        # Schritt 1: Auf Kamera-Auflösung skalieren
        img = img.resize((self.CAMERA_X, self.CAMERA_Y), Image.LANCZOS)
        # Schritt 2: Blur-Padding auf 600x600
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
