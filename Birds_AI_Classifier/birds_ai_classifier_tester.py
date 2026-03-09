import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as tf_image
from tensorflow.keras.applications.inception_v3 import preprocess_input

class BirdAITesterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Birds AI Classifier - Tester")
        self.root.geometry("900x700")
        
        # Model and Labels
        self.model_path = "my_birds_modell_800x448.keras"
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
        except Exception as e:
            self.lbl_status.config(text=f"Fehler beim Laden: {str(e)}", fg="red")
            messagebox.showerror("Fehler", f"Fehler bei der Initialisierung:\n{str(e)}")

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
        
        # Bild in der GUI anzeigen
        try:
            display_img = Image.open(file_path)
            display_img.thumbnail((800, 450)) # Skalieren für die GUI
            self.tk_img = ImageTk.PhotoImage(display_img)
            self.img_label.config(image=self.tk_img)
        except Exception as e:
            messagebox.showerror("Fehler", f"Konnte Bild nicht anzeigen:\n{str(e)}")
            return
            
        # Klassifizierung
        try:
            img = tf_image.load_img(file_path, target_size=(448, 800))
            x = tf_image.img_to_array(img)
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
