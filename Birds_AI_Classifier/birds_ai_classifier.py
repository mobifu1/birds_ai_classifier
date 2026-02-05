import os
import time
import threading
import multiprocessing  # Für Prozess-Trennung
import sqlite3
import webbrowser
import datetime
from pathlib import Path
import io
import base64
import json

# --- WICHTIG: Matplotlib Einstellung für Threads/Prozesse ---
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# GUI & System
import tkinter as tk
from tkinter import filedialog, messagebox

# Web & Data
from flask import Flask, render_template_string, request, url_for
import pandas as pd

# AI (TensorFlow)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image as tf_image
from tensorflow.keras.models import load_model 
import numpy as np

# --- KONFIGURATION ---
DB_FILE = "birds_stats.db"
FLASK_PORT = 5000
CHECK_INTERVAL_SECONDS = 5 

# --- MASK PARAMETER ---
MASK_TOP = 14  
MASK_BOTTOM = 10

# Globale Variable für den Pfad
CURRENT_MONITOR_PATH = None

# --- HELFER: ORDNERGRÖSSE BERECHNEN ---
def get_dir_size_mb(folder, recursive=False):
    """Berechnet die Größe eines Ordners in Megabyte."""
    if not folder or not os.path.exists(folder):
        return 0.0
    total_size = 0
    try:
        path_obj = Path(folder)
        iterator = path_obj.rglob('*') if recursive else os.scandir(folder)
        for entry in iterator:
            if recursive:
                if entry.is_file():
                    total_size += entry.stat().st_size
            else:
                if entry.is_file():
                    total_size += entry.stat().st_size
        return total_size / (1024 * 1024) # MB
    except Exception as e:
        print(f"Fehler bei Größenberechnung: {e}")
        return 0.0

# --- DATENBANK SETUP (Optimiert) ---
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        # WAL-Modus für gleichzeitigen Zugriff
        c.execute('PRAGMA journal_mode=WAL;')
        
        c.execute('''CREATE TABLE IF NOT EXISTS detections 
                     (id INTEGER PRIMARY KEY, filename TEXT UNIQUE, species TEXT, timestamp TEXT, confidence REAL)''')
        
        # Index für Performance
        c.execute('CREATE INDEX IF NOT EXISTS idx_species ON detections(species);')
    except Exception as e:
        print(f"DB Init Fehler: {e}") 
    conn.commit()
    conn.close()

# --- KI KLASSIFIZIERUNG ---
class BirdAI:
    def __init__(self):
        self.custom_model_path = "my_birds_modell.keras"
        self.labels_path = "model_labels.json"
        self.use_custom = False
        self.labels_map = {}
        self.model = None

        if os.path.exists(self.custom_model_path) and os.path.exists(self.labels_path):
            try:
                self.model = load_model(self.custom_model_path)
                with open(self.labels_path, 'r') as f:
                    raw_labels = json.load(f)
                    self.labels_map = {int(k): v for k, v in raw_labels.items()}
                self.use_custom = True
                print("Eigenes Modell geladen.")
            except Exception as e:
                print(f"Fehler beim Laden des eigenen Modells: {e}")
                self.load_standard_model()
        else:
            self.load_standard_model()

    def load_standard_model(self):
        self.use_custom = False
        self.model = MobileNetV2(weights='imagenet')
        self.translations = {
            'robin': 'Rotkehlchen', 'goldfinch': 'Stieglitz'
        }

    def analyze_image(self, img_path):
        try:
            img = tf_image.load_img(img_path, target_size=(224, 224))
            x = tf_image.img_to_array(img)
            
            # Maskierung
            x[:MASK_TOP, :, :] = 0
            h = x.shape[0]
            x[h-MASK_BOTTOM:, :, :] = 0
            
            # Debug
            debug_dir = "debug_live_masking"
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)
            
            debug_file = os.path.join(debug_dir, "live_check.png")
            if not os.path.exists(debug_file):
                tf_image.array_to_img(x).save(debug_file)

            # KI
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            preds = self.model.predict(x, verbose=0)

            if self.use_custom:
                best_index = np.argmax(preds[0])
                confidence = float(preds[0][best_index])
                label_name = self.labels_map.get(best_index, "Unbekannt")
                return label_name.replace('_', ' ').title(), confidence
            else:
                results = decode_predictions(preds, top=1)[0]
                english_label = results[0][1]
                confidence = results[0][2]
                translated_label = self.translations.get(english_label, english_label)
                return translated_label.replace('_', ' ').title(), confidence
        except Exception as e:
            print(f"Fehler in analyze_image bei {img_path}: {e}")
            return "Fehler", 0.0

# --- HINTERGRUND ÜBERWACHUNG ---
class FolderMonitor:
    def __init__(self, update_log_callback, get_threshold_callback, update_size_callback):
        self.running = False
        self.folder_path = ""
        self.recursive = False 
        self.ai = None
        self.log_callback = update_log_callback
        self.get_threshold = get_threshold_callback
        self.update_size_callback = update_size_callback
        self.thread = None

    def start(self, folder_path, recursive=False): 
        if not folder_path: return
        self.folder_path = folder_path
        self.recursive = recursive
        global CURRENT_MONITOR_PATH
        CURRENT_MONITOR_PATH = folder_path
        self.running = True
        
        if self.ai is None:
            self.ai = BirdAI()
            
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        global CURRENT_MONITOR_PATH
        CURRENT_MONITOR_PATH = None

    def loop(self):
        mode_str = "rekursiv" if self.recursive else "nur Hauptebene"
        self.log_callback(f"Überwachung gestartet ({mode_str}): {self.folder_path}")
        
        while self.running:
            try:
                self.scan_folder()
                current_size = get_dir_size_mb(self.folder_path, self.recursive)
                self.update_size_callback(current_size)
            except Exception as e:
                print(f"Fehler im Loop: {e}")
                
            for _ in range(CHECK_INTERVAL_SECONDS):
                if not self.running: break
                time.sleep(1)

    def scan_folder(self):
        # Timeout verhindert "Database locked"
        conn = sqlite3.connect(DB_FILE, timeout=10)
        c = conn.cursor()
        
        try:
            c.execute("SELECT filename FROM detections")
            processed_files = {row[0] for row in c.fetchall()}
        except:
            processed_files = set()
        
        extensions = ['*.jpg', '*.jpeg', '*.JPG', '*.png']
        files_found = []
        for ext in extensions:
            if self.folder_path:
                if self.recursive:
                    files_found.extend(Path(self.folder_path).rglob(ext))
                else:
                    files_found.extend(Path(self.folder_path).glob(ext))

        new_files = [f for f in files_found if f.name not in processed_files]

        if new_files:
            current_threshold = self.get_threshold() 
            self.log_callback(f"{len(new_files)} neue Bilder. Filter ab {current_threshold}%...")
            
            for file_path in new_files:
                if not self.running: break
                
                species, conf = self.ai.analyze_image(str(file_path))
                conf_percent = int(conf * 100)
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                try:
                    if conf_percent >= current_threshold:
                        c.execute("INSERT INTO detections (filename, species, timestamp, confidence) VALUES (?, ?, ?, ?)",
                                  (file_path.name, species, timestamp, conf))
                        conn.commit()
                        self.log_callback(f"[{file_path.name}] ✅ {species} ({conf_percent}%) -> Gespeichert")
                    else:
                        c.execute("INSERT INTO detections (filename, species, timestamp, confidence) VALUES (?, ?, ?, ?)",
                                  (file_path.name, "IGNORED_LOW_CONFIDENCE", timestamp, conf))
                        conn.commit()
                        self.log_callback(f"[{file_path.name}] ❌ {species} ({conf_percent}%) -> Ignoriert (zu unsicher)")
                except sqlite3.IntegrityError:
                    print(f"Duplikat übersprungen: {file_path.name}")
        
        conn.close()

# --- WEB SERVER (FLASK) ---
app = Flask(__name__)

@app.route('/')
def dashboard():
    chart_type = request.args.get('type', 'donut')
    
    # Timeout für Webserver-Stabilität
    conn = sqlite3.connect(DB_FILE, timeout=10)
    try:
        df = pd.read_sql_query("SELECT species, COUNT(*) as count FROM detections GROUP BY species", conn)
        if not df.empty:
            df['species'] = df['species'].replace('IGNORED_LOW_CONFIDENCE', 'Nicht erkannt')
    except:
        df = pd.DataFrame()
    finally:
        conn.close()
        
    total_count = df['count'].sum() if not df.empty else 0
    unknown_percent_str = "0.0 %"
    if total_count > 0 and not df.empty:
        unknown_row = df[df['species'] == 'Nicht erkannt']
        if not unknown_row.empty:
            u_count = unknown_row.iloc[0]['count']
            pct = (u_count / total_count) * 100
            unknown_percent_str = f"{pct:.1f} %"
    
    # Speicherplatzanzeige wurde hier entfernt (Logik gelöscht)

    icon_map = {}
    static_folder = os.path.join(app.root_path, 'static', 'bird_icons')
    if os.path.exists(static_folder):
        for f in os.listdir(static_folder):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                name_key = os.path.splitext(f)[0]
                icon_map[name_key] = f"bird_icons/{f}"

    chart_url = ""
    if not df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        if chart_type == 'bar':
            colors = ['lightgray' if sp == 'Nicht erkannt' else 'skyblue' for sp in df['species']]
            ax.bar(df['species'], df['count'], color=colors)
            ax.set_xlabel('Art')
            ax.set_ylabel('Anzahl')
            ax.set_title('Erkannte Arten (inkl. nicht erkannte)')
            plt.xticks(rotation=45)
        else:
            wedges, texts, autotexts = ax.pie(df['count'], labels=df['species'], autopct='%1.1f%%', startangle=90, pctdistance=0.85)
            centre_circle = plt.Circle((0,0),0.70,fc='white')
            fig.gca().add_artist(centre_circle)
            ax.axis('equal') 
            ax.set_title('Verteilung der Arten', pad=20)
            plt.setp(autotexts, size=10, weight="bold", color="white")
        plt.tight_layout()
        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        chart_url = base64.b64encode(img.getvalue()).decode()
        plt.close(fig)

    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Vogel-Statistik</title>
        <meta http-equiv="refresh" content="30">
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; text-align: center; padding: 20px; background-color: #f4f4f9; color: #333; }
            .container { max-width: 900px; margin: auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; margin-bottom: 10px; }
            .status-box { background: #eef2f3; padding: 15px; border-radius: 8px; margin-bottom: 25px; font-weight: bold; border-left: 5px solid #007bff; display: inline-block; }
            .chart-toggle { margin: 20px 0; }
            .toggle-btn { text-decoration: none; padding: 10px 20px; border: 1px solid #007bff; color: #007bff; margin: 0 5px; border-radius: 20px; transition: all 0.3s; }
            .toggle-btn.active { background-color: #007bff; color: white; }
            .toggle-btn:hover { background-color: #0056b3; color: white; }
            table { width: 100%; max-width: 600px; margin: 30px auto; border-collapse: collapse; }
            th { background-color: #007bff; color: white; padding: 12px; text-align: left; }
            td { padding: 12px; border-bottom: 1px solid #ddd; vertical-align: middle; text-align: left; }
            tr:hover { background-color: #f1f1f1; }
            .flex-center { display: flex; align-items: center; justify-content: flex-start; gap: 12px; }
            .bird-icon { width: 24px; height: 24px; object-fit: contain; border-radius: 4px; }
            tfoot tr.total-row { background-color: #e3f2fd; font-weight: bold; border-top: 2px solid #007bff; }
            tfoot tr.sub-row { background-color: #f8f9fa; color: #666; font-size: 0.9em; border-top: none; }
            .refresh-btn { display: inline-block; margin-top: 20px; padding: 10px 20px; background: #6c757d; color: white; border-radius: 5px; text-decoration: none; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Vogel-Beobachtungs-Statistik</h1>
            
            {% if chart_url %}
                <div class="chart-toggle">
                    <a href="/?type=donut" class="toggle-btn {% if request.args.get('type') != 'bar' %}active{% endif %}">🍩 Donut</a>
                    <a href="/?type=bar" class="toggle-btn {% if request.args.get('type') == 'bar' %}active{% endif %}">📊 Balken</a>
                </div>
                <img src="data:image/png;base64,{{ chart_url }}" alt="Diagramm" style="max-width:100%; height:auto; border-radius:8px;">
                
                <br><br>
                
                <h2>Detaillierte Liste</h2>
                <table>
                    <thead><tr><th>Vogelart</th><th style="text-align: right;">Anzahl</th></tr></thead>
                    <tbody>
                    {% for index, row in df.iterrows() %}
                    <tr>
                        <td>
                            <div class="flex-center">
                                {% if row['species'] in icon_map %}
                                    <img src="{{ url_for('static', filename=icon_map[row['species']]) }}" class="bird-icon" alt="icon">
                                {% else %}
                                    <div style="width:24px; height:24px; background:#eee; border-radius:50%; text-align:center; line-height:24px; font-size:12px; color:#666;">?</div>
                                {% endif %}
                                <span>{{ row['species'] }}</span>
                            </div>
                        </td>
                        <td style="text-align: right; font-weight: bold;">{{ row['count'] }}</td>
                    </tr>
                    {% endfor %}
                    </tbody>
                    <tfoot>
                        <tr class="total-row"><td>GESAMT</td><td style="text-align: right; font-size: 1.1em;">{{ total_count }}</td></tr>
                        <tr class="sub-row"><td style="padding-left: 20px;">↳ Anteil "Nicht erkannt"</td><td style="text-align: right;">{{ unknown_percent }}</td></tr>
                    </tfoot>
                </table>
            {% else %}
                <p>Noch keine Vögel erkannt (oder Datenbank leer).</p>
            {% endif %}
            <p><a href="/" class="refresh-btn">Seite aktualisieren</a></p>
        </div>
    </body>
    </html>
    """
    return render_template_string(html, chart_url=chart_url, df=df, icon_map=icon_map, total_count=total_count, unknown_percent=unknown_percent_str)

def run_flask():
    app.run(port=FLASK_PORT, debug=False, use_reloader=False)

# --- HAUPTANWENDUNG (GUI) ---
class AppGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Birds-AI-Classifier (Deutsch)")
        self.root.geometry("600x680") 
        
        self.monitor = FolderMonitor(self.update_log, 
                                     lambda: self.scale_threshold.get(),
                                     self.update_size_display)
        
        tk.Label(root, text="Vogel-Überwachung", font=("Arial", 16, "bold")).pack(pady=10)
        
        frame_folder = tk.Frame(root)
        frame_folder.pack(pady=5)
        tk.Label(frame_folder, text="Bilder-Ordner:").pack(side=tk.LEFT, padx=5)
        self.entry_path = tk.Entry(frame_folder, width=40)
        self.entry_path.pack(side=tk.LEFT, padx=5)
        tk.Button(frame_folder, text="Durchsuchen...", command=self.select_folder).pack(side=tk.LEFT)
        
        # Checkbox für rekursive Suche
        self.recursive_var = tk.BooleanVar(value=True)
        tk.Checkbutton(root, text="Unterordner ebenfalls durchsuchen (Rekursiv)", 
                       variable=self.recursive_var).pack(pady=2, anchor=tk.W, padx=20)

        frame_settings = tk.LabelFrame(root, text="KI Einstellungen", padx=10, pady=10)
        frame_settings.pack(pady=10, padx=20, fill="x")
        
        tk.Label(frame_settings, text="Mindest-Wahrscheinlichkeit (%):").pack(anchor=tk.W)
        self.scale_threshold = tk.Scale(frame_settings, from_=0, to=100, orient=tk.HORIZONTAL, length=400, tickinterval=20)
        self.scale_threshold.set(65) 
        self.scale_threshold.pack()
        tk.Label(frame_settings, text="(Bilder unter diesem Wert werden als 'Nicht erkannt' markiert)", font=("Arial", 8)).pack(anchor=tk.W)

        frame_controls = tk.Frame(root)
        frame_controls.pack(pady=10)
        
        self.btn_start = tk.Button(frame_controls, text="Überwachung Starten", command=self.start_monitoring, bg="#dddddd", width=20)
        self.btn_start.pack(side=tk.LEFT, padx=5)
        
        tk.Button(frame_controls, text="Statistik öffnen", command=self.open_web).pack(side=tk.LEFT, padx=5)
        tk.Button(frame_controls, text="Datenbank leeren", command=self.reset_database, bg="#ff9999").pack(side=tk.LEFT, padx=5)
        
        self.lbl_size = tk.Label(root, text="Ordnergröße: 0.00 MB", font=("Arial", 10, "bold"), fg="blue")
        self.lbl_size.pack(pady=5)

        tk.Label(root, text="Status-Log:").pack(anchor=tk.W, padx=20)
        self.log_text = tk.Text(root, height=10, width=70, state='disabled')
        self.log_text.pack(pady=5)
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def select_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.entry_path.delete(0, tk.END)
            self.entry_path.insert(0, path)
            size = get_dir_size_mb(path, self.recursive_var.get())
            self.update_size_display(size)

    def start_monitoring(self):
        path = self.entry_path.get()
        if not path:
            messagebox.showwarning("Fehler", "Bitte wähle zuerst einen Ordner aus.")
            return
            
        if self.btn_start['text'] == "Überwachung Starten":
            self.monitor.start(path, self.recursive_var.get())
            self.btn_start.config(text="Stoppen", bg="#ffcccc")
            self.update_log(f"Service gestartet (Schwelle: {self.scale_threshold.get()}%)...")
        else:
            self.monitor.stop()
            self.btn_start.config(text="Überwachung Starten", bg="#dddddd")
            self.update_log("Service gestoppt.")

    def update_size_display(self, size_mb):
        def _update():
            color = "green"
            if size_mb > 100: color = "orange"
            if size_mb > 500: color = "red"
            self.lbl_size.config(text=f"Ordnergröße: {size_mb:.2f} MB", fg=color)
        self.root.after(0, _update)

    def open_web(self):
        webbrowser.open(f"http://localhost:{FLASK_PORT}")

    def reset_database(self):
        confirm = messagebox.askyesno("Warnung", "Alle Daten löschen?")
        if confirm:
            try:
                conn = sqlite3.connect(DB_FILE, timeout=10)
                c = conn.cursor()
                c.execute("DELETE FROM detections")
                conn.commit()
                conn.close()
                self.update_log("Datenbank geleert.")
                messagebox.showinfo("Erfolg", "Datenbank geleert.")
            except Exception as e:
                messagebox.showerror("Fehler", f"{e}")

    def update_log(self, message):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')
        
    def on_close(self):
        self.monitor.stop()
        self.root.destroy()
        os._exit(0)

if __name__ == "__main__":
    init_db()
    
    # Flask Prozess Start
    flask_process = multiprocessing.Process(target=run_flask, daemon=True)
    flask_process.start()
    
    root = tk.Tk()
    app_gui = AppGUI(root)
    root.mainloop()