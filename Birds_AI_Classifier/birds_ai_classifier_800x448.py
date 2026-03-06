import os
import time
import threading
import multiprocessing
import sqlite3
import webbrowser
import datetime
from pathlib import Path
import io
import base64
import json
import re
import random 
import gc 
import shutil 
import psutil
import subprocess
import platform

# --- NEU: Pillow für EXIF-Daten ---
from PIL import Image, ExifTags

# --- NEU: Production Server Import ---
from waitress import serve

# --- WICHTIG: Matplotlib Einstellung ---
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# GUI & System
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

# Web & Data
from flask import Flask, render_template_string, request, url_for
import pandas as pd

# AI (TensorFlow)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf

# InceptionV3 Importe
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image as tf_image
from tensorflow.keras.models import load_model 
import numpy as np

# --- KONFIGURATION ---
DB_FILE = "birds_stats.db"
GREYLIST_FILE = "greylist.json" 
BLACKLIST_FILE = "blacklist.json" 
BACKLOG_FILE = "backlog.json"
SETTINGS_FILE = "settings.json" 
FLASK_PORT = 5000
CHECK_INTERVAL_SECONDS = 5 
STATIC_FOLDER = "static" 
LAST_IMG_NAME = "last_detection.jpg" 

# --- MASK PARAMETER ---
MASK_TOP = 0  
MASK_BOTTOM = 0

# --- HELFER: DATUM AUS BILD LESEN ---
def get_original_date(file_path):
    """
    Versucht das Aufnahmedatum aus den EXIF-Daten zu lesen.
    Falls nicht vorhanden, wird das Änderungsdatum des Dateisystems verwendet.
    """
    # 1. Versuch: EXIF
    try:
        image = Image.open(file_path)
        exif_data = image._getexif()
        
        if exif_data:
            for tag, value in exif_data.items():
                decoded = ExifTags.TAGS.get(tag, tag)
                if decoded == 'DateTimeOriginal':
                    if len(str(value)) >= 19 and str(value)[4] == ':':
                        return str(value)[:4] + '-' + str(value)[5:7] + '-' + str(value)[8:]
                    return str(value)
    except Exception:
        pass 

    # 2. Versuch: Dateisystem
    try:
        timestamp = os.path.getmtime(file_path)
        return datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
    except:
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# --- HELFER: SETTINGS ---
def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {}

def save_setting(key, value):
    data = load_settings()
    data[key] = value
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        print(f"Fehler beim Speichern der Settings: {e}")

# --- HELFER: ORDNERGRÖSSE ---
def get_dir_size_mb(folder, recursive=False):
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
        return total_size / (1024 * 1024) 
    except Exception as e:
        print(f"Fehler bei Größenberechnung: {e}")
        return 0.0

# --- HELFER: LISTEN LADEN/SPEICHERN ---
def load_greylist():
    if os.path.exists("blacklist.json") and not os.path.exists(GREYLIST_FILE):
        try:
            os.rename("blacklist.json", GREYLIST_FILE)
        except: pass

    if os.path.exists(GREYLIST_FILE):
        try:
            with open(GREYLIST_FILE, 'r', encoding='utf-8') as f:
                return set(json.load(f))
        except:
            return set()
    return set()

def save_greylist(greylist_set):
    try:
        with open(GREYLIST_FILE, 'w', encoding='utf-8') as f:
            json.dump(list(greylist_set), f, ensure_ascii=False, indent=2)
    except: pass

def load_blacklist():
    defaults = {"Hintergrund", "Unbekannt"}
    if os.path.exists(BLACKLIST_FILE):
        try:
            with open(BLACKLIST_FILE, 'r', encoding='utf-8') as f:
                return set(json.load(f))
        except:
            return defaults
    return defaults

def save_blacklist(blacklist_set):
    try:
        with open(BLACKLIST_FILE, 'w', encoding='utf-8') as f:
            json.dump(list(blacklist_set), f, ensure_ascii=False, indent=2)
    except: pass

def load_backlog():
    if os.path.exists(BACKLOG_FILE):
        try:
            with open(BACKLOG_FILE, 'r', encoding='utf-8') as f:
                return set(json.load(f))
        except:
            return set()
    return set()

def save_backlog(backlog_set):
    try:
        with open(BACKLOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(list(backlog_set), f, ensure_ascii=False, indent=2)
    except: pass

def load_categories():
    cat_file = "species_categories.json"
    if os.path.exists(cat_file):
        try:
            with open(cat_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

# --- DATENBANK SETUP ---
def init_db():
    # --- AUTO-ROTATION: JÄHRLICHES ARCHIVIEREN ---
    if os.path.exists(DB_FILE):
        try:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            # Prüfen, ob es Einträge aus einem *vergangenen* Jahr gibt
            cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM detections")
            row = cursor.fetchone()
            conn.close()

            if row and row[0]:
                first_date = row[0] # Format: YYYY-MM-DD HH:MM:SS
                last_date = row[1]
                
                db_year = int(first_date[:4])
                current_year = datetime.datetime.now().year
                
                # Wenn das Jahr des ersten Eintrags kleiner ist als das aktuelle Jahr,
                # dann ist es eine "alte" Datenbank -> Archivieren.
                if db_year < current_year:
                    archive_name = f"birds_stats_{db_year}.db"
                    if not os.path.exists(archive_name):
                        print(f"Jahreswechsel erkannt! Archiviere Datenbank nach {archive_name}...")
                        os.rename(DB_FILE, archive_name)
                    else:
                        print(f"Warnung: Archiv {archive_name} existiert bereits. Überspringe Rotation.")
        except Exception as e:
            print(f"Fehler bei DB-Rotationsprüfung: {e}")

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        c.execute('PRAGMA journal_mode=WAL;')
        c.execute('''CREATE TABLE IF NOT EXISTS detections 
                     (id INTEGER PRIMARY KEY, filename TEXT UNIQUE, species TEXT, timestamp TEXT, confidence REAL)''')
        c.execute('CREATE INDEX IF NOT EXISTS idx_species ON detections(species);')
        c.execute('CREATE INDEX IF NOT EXISTS idx_filename ON detections(filename);') 
        # Optional: Index auf Timestamp für schnellere zeitbasierte Abfragen
        c.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON detections(timestamp);')
    except Exception as e:
        print(f"DB Init Fehler: {e}") 
    conn.commit()
    conn.close()

    if not os.path.exists(STATIC_FOLDER):
        os.makedirs(STATIC_FOLDER)

# --- KI KLASSIFIZIERUNG ---
class BirdAI:
    def __init__(self):
        self.custom_model_path = "my_birds_modell_800x448.keras"
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
        print("Lade Standard-Modell (InceptionV3)...")
        self.model = InceptionV3(weights='imagenet')
        self.translations = {
            'robin': 'Rotkehlchen', 'goldfinch': 'Stieglitz', 
            'house_sparrow': 'Haussperling', 'magpie': 'Elster',
            'black_grouse': 'Birkhuhn', 'jay': 'Eichelhäher'
        }

    def analyze_image(self, img_path):
        try:
            img = tf_image.load_img(img_path, target_size=(448, 800))
            x = tf_image.img_to_array(img)
            x[:MASK_TOP, :, :] = 0
            h = x.shape[0]
            x[h-MASK_BOTTOM:, :, :] = 0
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
            return "Fehler", 0.0

# --- HINTERGRUND ÜBERWACHUNG ---
class FolderMonitor:
    def __init__(self, update_log_callback, get_threshold_callback, update_size_callback, 
                 get_rename_callback, get_delete_callback, 
                 get_greylist_active_callback, get_greylist_callback,
                 get_backlog_active_callback, get_backlog_callback,
                 get_blacklist_callback, update_duration_callback=None,
                 update_remaining_callback=None, get_algo_active_callback=None): 
        self.running = False
        self.folder_path = ""
        self.recursive = False 
        self.ai = None
        self.log_callback = update_log_callback
        self.get_threshold = get_threshold_callback
        self.update_size_callback = update_size_callback
        self.get_rename_enabled = get_rename_callback
        self.get_delete_enabled = get_delete_callback
        self.get_greylist_active = get_greylist_active_callback
        self.get_greylist = get_greylist_callback
        self.get_backlog_active = get_backlog_active_callback
        self.get_backlog_callback = get_backlog_callback
        self.get_blacklist = get_blacklist_callback
        self.get_algo_active = get_algo_active_callback if get_algo_active_callback else lambda: False
        self.update_duration_callback = update_duration_callback
        self.update_remaining_callback = update_remaining_callback
        self.thread = None
        self.normal_timers = {}
        self.lazy_occupier = None
        self.categories = load_categories()

    def start(self, folder_path, recursive=False): 
        if not folder_path: return
        self.folder_path = folder_path
        self.recursive = recursive
        self.running = True
        
        # NEU: Lese- und Schreibrechte prüfen und protokollieren
        has_read = os.access(self.folder_path, os.R_OK)
        has_write = False
        try:
            test_file = os.path.join(self.folder_path, '.permission_test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            has_write = True
        except Exception:
            has_write = False

        if not has_read:
            self.log_callback(f"❌ FEHLER: Keine Leseberechtigung auf '{self.folder_path}'.")
            self.log_callback("   -> Es können keine Bilder gefunden oder gelesen werden. (Nichts passiert)")
        elif not has_write:
            self.log_callback(f"⚠️ WARNUNG: Keine Schreibrechte oder Löschrechte auf '{self.folder_path}'.")
            self.log_callback("   -> Umbenennen, Verschieben oder Löschen von Bildern wird fehlschlagen.")

        if self.ai is None:
            self.ai = BirdAI()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

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
        conn = sqlite3.connect(DB_FILE, timeout=10)
        c = conn.cursor()
        extensions = ['*.jpg', '*.jpeg', '*.JPG', '*.png']
        files_found_iterators = []
        path_obj = Path(self.folder_path)
        
        for ext in extensions:
            if self.folder_path:
                if self.recursive:
                    files_found_iterators.append(path_obj.rglob(ext))
                else:
                    files_found_iterators.append(path_obj.glob(ext))
        
        new_files_set = set()
        for iterator in files_found_iterators:
            for file_path in iterator:
                if not self.running: break
                try:
                    c.execute("SELECT 1 FROM detections WHERE filename = ? LIMIT 1", (file_path.name,))
                    if c.fetchone() is None:
                        # Use resolved path or absolute path string to ensure uniqueness
                        new_files_set.add(str(file_path.resolve()))
                except Exception: pass
                
        new_files = [Path(p) for p in new_files_set]

        if self.update_remaining_callback:
            self.update_remaining_callback(len(new_files))

        if len(new_files) > 0:
            files_to_process = new_files
            current_threshold = self.get_threshold() 
            rename_active = self.get_rename_enabled() 
            delete_unsure_active = self.get_delete_enabled() 
            greylist_active = self.get_greylist_active()
            current_greylist = self.get_greylist()
            backlog_active = self.get_backlog_active()
            current_backlog = self.get_backlog_callback()
            current_blacklist = self.get_blacklist() 

            self.log_callback(f"{len(files_to_process)} neue Bilder gefunden. Verarbeite...")
            
            for index, file_path in enumerate(files_to_process):
                if self.update_remaining_callback:
                    self.update_remaining_callback(len(files_to_process) - index)
                if not self.running: break
                if not os.path.exists(file_path): continue 

                start_time = time.time()
                try:
                    species, conf = self.ai.analyze_image(str(file_path))
                except Exception: continue
                duration_ms = int((time.time() - start_time) * 1000)
                if self.update_duration_callback:
                    self.update_duration_callback(duration_ms)
                if species == "Fehler": continue
                
                gc.collect() 
                
                try:
                    target_img = os.path.join(STATIC_FOLDER, LAST_IMG_NAME)
                    shutil.copy2(file_path, target_img)
                except: pass

                conf_percent = int(conf * 100)
                timestamp = get_original_date(str(file_path))
                final_filename = file_path.name
                
                algo_active = self.get_algo_active()
                algo_ignore = False
                
                if algo_active and conf_percent >= current_threshold and species not in ["Hintergrund", "Unbekannt"]:
                    cat = self.categories.get(species, "normal").lower()
                    if cat == "lazy":
                        if self.lazy_occupier == species:
                            algo_ignore = True
                            self.log_callback(f"[{final_filename}] ⏳ {species} -> Ignoriert (Algorithmus: Lazy Occupier)")
                        else:
                            self.lazy_occupier = species
                            
                    elif cat == "normal":
                        self.lazy_occupier = None # Futterplatz wieder frei
                        now = time.time()
                        last_seen = self.normal_timers.get(species, 0)
                        if (now - last_seen) < 120:
                            algo_ignore = True
                            self.log_callback(f"[{final_filename}] ⏳ {species} -> Ignoriert (Algorithmus: Normal < 2 Min)")
                        else:
                            self.normal_timers[species] = now
                            
                    elif cat == "hectic":
                        self.lazy_occupier = None # Futterplatz wieder frei
                        # Wird sofort gezählt

                if algo_ignore:
                    # Bild ins Backlog verschieben als _algo_ignore und in DB nicht speichern
                    try:
                        app_dir = Path(os.path.abspath(os.path.dirname(__file__)))
                        backlog_dir = app_dir / "backlog"
                        backlog_dir.mkdir(exist_ok=True)
                        file_ext = file_path.suffix
                        rand_id = random.randint(100000, 999999)
                        target_backlog_path = backlog_dir / f"{rand_id}_algo_ignore{file_ext}"
                        shutil.move(str(file_path), str(target_backlog_path))
                        self.log_callback(f"[{final_filename}] ⏳ Verschiebe ignoriertes Bild ins Backlog -> {target_backlog_path.name}")
                    except Exception as e:
                        self.log_callback(f"[{final_filename}] ❌ Fehler beim Verschieben (Algo-Backlog): {e}")
                    continue
                
                if rename_active:
                    try:
                        if not os.path.exists(file_path): continue
                        file_ext = file_path.suffix
                        if conf_percent >= current_threshold:
                            clean_species = species.replace(" ", "_")
                        else:
                            clean_species = f"Unbekannt_{species.replace(' ', '_')}_{conf_percent}pct"
                        while True:
                            rand_id = random.randint(100000, 999999)
                            new_name = f"{rand_id}_{clean_species}{file_ext}"
                            new_full_path = file_path.parent / new_name
                            if not new_full_path.exists():
                                break
                        old_name = file_path.name
                        os.rename(file_path, new_full_path)
                        final_filename = new_name
                        file_path = new_full_path 
                        self.log_callback(f"[{old_name}] ✏️ Umbenannt zu: {final_filename}")
                    except Exception as e:
                        self.log_callback(f"[{file_path.name}] ❌ Fehler beim Umbenennen: {e}")

                if backlog_active:
                    is_low_confidence = (conf_percent < current_threshold)
                    # Wenn das Bild unter dem Threshold liegt, behandeln wir es für den Backlog-Filter als "Unbekannt"
                    match_species = "Unbekannt" if is_low_confidence else species
                    
                    if species in current_backlog or match_species in current_backlog:
                        try:
                            # Der backlog Ordner liegt ab jetzt direkt im Verzeichnis des Python Skripts
                            app_dir = Path(os.path.abspath(os.path.dirname(__file__)))
                            backlog_dir = app_dir / "backlog"
                            backlog_dir.mkdir(exist_ok=True)
                            
                            target_backlog_path = backlog_dir / file_path.name
                            shutil.move(str(file_path), str(target_backlog_path))
                            self.log_callback(f"[{final_filename}] ⏳ {match_species} -> Ins Backlog verschoben")
                        except Exception as e:
                            self.log_callback(f"[{final_filename}] ❌ Fehler beim Verschieben (Backlog): {e}")
                        continue
                
                if greylist_active and species in current_greylist:
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            self.log_callback(f"[{final_filename}] 🚫 {species} -> Bild gelöscht (Greylist)")
                    except Exception as e:
                        self.log_callback(f"[{final_filename}] ❌ Fehler beim Löschen (Greylist): {e}")
                    try:
                        c.execute("INSERT INTO detections (filename, species, timestamp, confidence) VALUES (?, ?, ?, ?)",
                                  (final_filename, species, timestamp, conf))
                        conn.commit()
                    except sqlite3.IntegrityError: pass
                    continue 
                
                is_blacklisted = (species in current_blacklist) 
                should_delete_trash = delete_unsure_active and (conf_percent < current_threshold or is_blacklisted)

                if should_delete_trash:
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            reason = "Blacklist" if is_blacklisted else "Unsicher"
                            self.log_callback(f"[{final_filename}] 🗑️ {species} ({conf_percent}%) -> Gelöscht ({reason})")
                    except Exception as e:
                        self.log_callback(f"[{final_filename}] ❌ Fehler beim Löschen (Trash): {e}")
                    continue 

                try:
                    if conf_percent >= current_threshold:
                        c.execute("INSERT INTO detections (filename, species, timestamp, confidence) VALUES (?, ?, ?, ?)",
                                  (final_filename, species, timestamp, conf))
                        conn.commit()
                        self.log_callback(f"[{final_filename}] ✅ {species} ({conf_percent}%) -> Gespeichert")
                    else:
                        c.execute("INSERT INTO detections (filename, species, timestamp, confidence) VALUES (?, ?, ?, ?)",
                                  (final_filename, "IGNORED_LOW_CONFIDENCE", timestamp, conf))
                        conn.commit()
                        self.log_callback(f"[{final_filename}] ❌ {species} ({conf_percent}%) -> Ignoriert (DB)")
                except sqlite3.IntegrityError: pass 
            
            if self.update_remaining_callback:
                self.update_remaining_callback(0)
        conn.close()

# --- NEU: VERSION ---
APP_VERSION = "Version 1.0-R"

# --- WEB SERVER (FLASK) ---
app = Flask(__name__)

# --- STYLE CSS CONSTANT ---
CSS_STYLE = """
<style>
    body { 
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
        text-align: center; 
        padding: 0; 
        margin: 0;
        background-color: #121212; 
        color: #e0e0e0; 
    }
    .container { 
        max-width: 1100px; 
        margin: 20px auto; 
        background: #1e1e1e; 
        padding: 30px; 
        border-radius: 12px; 
        box-shadow: 0 4px 15px rgba(0,0,0,0.5); 
    }
    .footer {
        margin-top: 30px;
        font-size: 0.8em;
        color: #555;
        border-top: 1px solid #333;
        padding-top: 10px;
    }
    .header-container {
        position: sticky;
        top: 0;
        background-color: #1e1e1e;
        z-index: 1000;
        padding: 10px 0;
        border-bottom: 1px solid #333;
        margin-bottom: 20px;
    }
    h1 { color: #ffffff; margin: 0; font-size: 1.8em; }
    h2 { color: #4fc3f7; margin-top: 30px;}
    
    a.button-link {
        display: inline-block;
        margin: 10px;
        padding: 10px 20px;
        background: #0d47a1;
        color: white;
        text-decoration: none;
        border-radius: 5px;
        font-weight: bold;
    }
    a.button-link:hover { background: #1565c0; }

    /* Chart / Images */
    .last-sighting { 
        background: #263238; 
        border: 2px solid #37474f; 
        border-radius: 10px; 
        padding: 15px; 
        margin: 20px 0;
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .last-sighting img { 
        max-height: 300px; 
        max-width: 100%; 
        border-radius: 8px; 
        margin-top: 10px;
    }
    
    /* Tables */
    table { width: 100%; max-width: 1000px; margin: 30px auto; border-collapse: collapse; }
    th { background-color: #0d47a1; color: white; padding: 12px; text-align: left; }
    td { padding: 12px; border-bottom: 1px solid #333; vertical-align: middle; text-align: left; }
    tr:not(.highlight-row):hover { background-color: #2c2c2c; } 
    
    /* Weekly Table Specifics (ULTRA KOMPAKT & STICKY COLUMN) */
    .table-responsive {
        overflow-x: auto;
        margin-top: 20px;
        position: relative; /* Für sticky positioning */
    }
    .weekly-table th, .weekly-table td {
        text-align: center;
        padding: 1px; 
        border: 1px solid #333;
        font-size: 0.75em;
        height: 22px; /* Etwas höher für Icon + Text */
    }
    .weekly-table th { 
        background-color: #37474f; 
        min-width: 25px;
        padding: 2px 0;
        vertical-align: bottom;
    }

    /* Sticky First Column (Vogelart mit Icon) */
    .weekly-table th:first-child,
    .weekly-table td:first-child {
        position: sticky;
        left: 0;
        z-index: 2; 
        background-color: #263238; 
        border-right: 2px solid #555;
        white-space: nowrap; 
        text-align: left;
        padding-left: 5px;
        min-width: 140px; /* Breiter für Icon */
    }
    .weekly-table th:first-child {
         background-color: #37474f;
         z-index: 3; 
    }

    /* Icon in Weekly Table */
    .species-wrapper {
        display: flex;
        align-items: center;
        height: 100%;
    }
    .bird-icon-small {
        width: 18px;
        height: 18px;
        object-fit: contain; /* Icons bleiben ganz sichtbar */
        margin-right: 6px;
        border-radius: 3px;
        background-color: #ffffff; /* Weiß hinterlegt wie gewünscht */
    }
    .bird-icon-placeholder {
        width: 18px;
        height: 18px;
        margin-right: 6px;
        background-color: #444;
        border-radius: 3px;
        display: inline-block;
        text-align: center;
        line-height: 18px;
        font-size: 9px;
        color: #888;
    }
    
    /* Legend CSS */
    .legend-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 20px;
        gap: 15px;
        background: #263238;
        padding: 10px;
        border-radius: 8px;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
    }
    .legend-item {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.9em;
    }
    .legend-box {
        width: 20px;
        height: 20px;
        border: 1px solid #555;
    }
    
    .flex-center { display: flex; align-items: center; justify-content: flex-start; gap: 12px; }
    .bird-icon { width: 24px; height: 24px; object-fit: contain; border-radius: 4px; background-color: #ffffff; }
</style>
"""

@app.route('/')
def dashboard():
    current_settings = load_settings()
    current_threshold = current_settings.get("threshold", 70)
    
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    
    conn = sqlite3.connect(DB_FILE, timeout=10)
    last_entry = None
    try:
        query = f"""
            SELECT 
                species, 
                COUNT(*) as count,
                SUM(CASE WHEN timestamp LIKE '{today_str}%' THEN 1 ELSE 0 END) as today_count
            FROM detections 
            GROUP BY species
        """
        df = pd.read_sql_query(query, conn)
        cursor = conn.cursor()
        cursor.execute("SELECT species, filename, timestamp, confidence FROM detections ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        if row:
            last_entry = { 'species': row[0], 'filename': row[1], 'timestamp': row[2], 'confidence': int(row[3] * 100) }
            if last_entry['species'] == "IGNORED_LOW_CONFIDENCE": last_entry['species'] = "Unbekannt"

        if not df.empty:
            df['species'] = df['species'].replace('IGNORED_LOW_CONFIDENCE', 'Unbekannt')
            df['count'] = pd.to_numeric(df['count'])
            df['today_count'] = pd.to_numeric(df['today_count']).fillna(0).astype(int)
            df = df.sort_values(by='count', ascending=False)
    except: df = pd.DataFrame()
    finally: conn.close()
        
    ping_active = False
    camera_online = False
    try:
        if os.path.exists("camera_status.json"):
            with open("camera_status.json", "r") as f:
                cstat = json.load(f)
                ping_active = cstat.get("ping_active", False)
                camera_online = cstat.get("camera_online", False)
    except:
        pass
        
    total_count = df['count'].sum() if not df.empty else 0
    today_total = df['today_count'].sum() if not df.empty else 0
    unknown_percent_str = "0.0 %"
    if total_count > 0 and not df.empty:
        unknown_row = df[df['species'] == 'Unbekannt']
        if not unknown_row.empty:
            pct = (unknown_row.iloc[0]['count'] / total_count) * 100
            unknown_percent_str = f"{pct:.1f} %"
    
    # Icons Logic
    icon_map = {}
    static_folder = os.path.join(app.root_path, 'static', 'bird_icons')
    if os.path.exists(static_folder):
        for f in os.listdir(static_folder):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                icon_map[os.path.splitext(f)[0]] = f"bird_icons/{f}"

    # Chart Generation (IMMER BALKEN)
    chart_url = ""
    if not df.empty:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1e1e1e')
        # Balkendiagramm Logik
        cmap = plt.get_cmap('tab20')
        colors = [('#555555' if sp == 'Unbekannt' else cmap(i % 20)) for i, sp in enumerate(df['species'])]
        ax.bar(df['species'], df['count'], color=colors)
        ax.tick_params(axis='x', colors='white', rotation=45)
        ax.tick_params(axis='y', colors='white')
        ax.set_title('Verteilung der Arten', color='white')
        ax.set_facecolor('#1e1e1e')
        
        plt.tight_layout()
        img = io.BytesIO()
        fig.savefig(img, format='png', facecolor='#1e1e1e')
        img.seek(0)
        chart_url = base64.b64encode(img.getvalue()).decode()
        plt.close(fig)

    timestamp_now = int(time.time())

    # HTML Template (Bereinigt)
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Vogel-Statistik</title>
        <meta http-equiv="refresh" content="30">
        <meta name="viewport" content="width=device-width, initial-scale=1.0"> 
        {{ css_style|safe }}
    </head>
    <body>
        <div class="container">
            <div class="header-container">
                <h1>📊 Vogel-Beobachtungs-Statistik (AI)</h1>
                <div style="margin-top:10px;">
                    <a href="/daily" class="button-link" style="background:#00838f;">📈 Tages-Ansicht (Verlauf)</a>
                    <a href="/weekly" class="button-link">📅 Wochen-Ansicht (Heatmap)</a>
                </div>
            </div>
            
            {% if last_entry %}
            <div class="last-sighting">
                <div class="last-info" style="font-size:1.2em;color:#81d4fa;">📸 Letzte Sichtung: <strong>{{ last_entry.species }}</strong></div>
                <div>Zeit: {{ last_entry.timestamp }} | Konfidenz: {{ last_entry.confidence }}%</div>
                <div style="display: flex; align-items: center; justify-content: center; gap: 15px;">
                    <img src="{{ url_for('static', filename='last_detection.jpg') }}?t={{ ts }}" alt="Warte auf Bild...">
                    {% if ping_active %}
                        <div style="display: flex; flex-direction: column; align-items: center;">
                            <div style="width: 20px; height: 20px; border-radius: 50%; background-color: {% if camera_online %}#00e676{% else %}#ff1744{% endif %}; box-shadow: 0 0 10px {% if camera_online %}#00e676{% else %}#ff1744{% endif %};"></div>
                            <small style="color: #aaa; margin-top: 5px;">Kamera</small>
                        </div>
                    {% endif %}
                </div>
            </div>
            {% endif %}

            {% if chart_url %}
                <div style="margin:20px 0;">
                    </div>
                <img src="data:image/png;base64,{{ chart_url }}" alt="Diagramm" style="max-width:100%; height:auto; border-radius:8px;">
                
                <h2>Detaillierte Liste</h2>
                <table>
                    <thead><tr><th>Vogelart</th><th style="text-align: right;">Heute</th><th style="text-align: right;">Gesamt</th></tr></thead>
                    <tbody>
                    {% for index, row in df.iterrows() %}
                    <tr>
                        <td>
                            <div class="flex-center">
                                {% if row['species'] in icon_map %}
                                    <img src="{{ url_for('static', filename=icon_map[row['species']]) }}" class="bird-icon">
                                {% else %}
                                    <div style="width:24px; height:24px; background:#555; border-radius:50%; text-align:center; line-height:24px; font-size:12px;">?</div>
                                {% endif %}
                                <span>{{ row['species'] }}</span>
                            </div>
                        </td>
                        <td style="text-align: right; font-weight: bold;">{{ row['today_count'] }}</td>
                        <td style="text-align: right; font-weight: bold;">{{ row['count'] }}</td>
                    </tr>
                    {% endfor %}
                    </tbody>
                    <tfoot>
                        <tr style="background-color:#0d47a1; font-weight:bold;"><td>GESAMT</td><td style="text-align: right;">{{ today_total }}</td><td style="text-align: right;">{{ total_count }}</td></tr>
                    </tfoot>
                </table>
            {% else %}
                <p>Noch keine Daten vorhanden.</p>
            {% endif %}
            <p><a href="/" class="button-link" style="background:#546e7a;">Seite aktualisieren</a></p>
            
            <div class="footer">
                {{ version }}
            </div>
        </div>
    </body>
    </html>
    """
    return render_template_string(html, 
                                  chart_url=chart_url, 
                                  df=df, 
                                  icon_map=icon_map, 
                                  total_count=total_count,
                                  today_total=today_total,
                                  last_entry=last_entry,
                                  css_style=CSS_STYLE,
                                  ts=timestamp_now,
                                  version=APP_VERSION,
                                  ping_active=ping_active,
                                  camera_online=camera_online)

@app.route('/weekly')
def weekly_stats():
    # --- SQL Aggregation (Speicheroptimierung) ---
    query = """
    SELECT 
        CASE WHEN species = 'IGNORED_LOW_CONFIDENCE' THEN 'Unbekannt' ELSE species END as species,
        strftime('%Y-%W', timestamp) as week_sort,
        strftime('%W', timestamp) || '<br><small style=''color:#aaa''>''' || substr(strftime('%Y', timestamp), 3, 2) || '</small>' as week_display,
        COUNT(*) as counts
    FROM detections
    WHERE timestamp IS NOT NULL AND timestamp != ''
    GROUP BY species, week_sort, week_display
    ORDER BY week_sort
    """
    
    conn = sqlite3.connect(DB_FILE, timeout=10)
    try:
        grouped = pd.read_sql_query(query, conn)
    except:
        grouped = pd.DataFrame()
    finally:
        conn.close()

    html_table = "<p>Keine Daten für die Wochenansicht.</p>"

    # --- Icons Logic (auch für Weekly) ---
    icon_map = {}
    static_folder = os.path.join(app.root_path, 'static', 'bird_icons')
    if os.path.exists(static_folder):
        for f in os.listdir(static_folder):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                icon_map[os.path.splitext(f)[0]] = f"bird_icons/{f}"

    if not grouped.empty:
        # Pivot Tabelle mit absoluten Zahlen erstellen
        pivot_counts = grouped.pivot(index='species', columns='week_display', values='counts').fillna(0)
        
        # Prozentrechnung
        week_totals = pivot_counts.sum(axis=0)
        pivot_pct = pivot_counts.div(week_totals, axis=1).mul(100).fillna(0)
        
        # Sortierung
        week_mapping = grouped[['week_sort', 'week_display']].drop_duplicates().sort_values('week_sort')
        sorted_columns = week_mapping['week_display'].tolist()
        pivot_pct = pivot_pct.reindex(columns=sorted_columns)
        
        total_counts = pivot_counts.sum(axis=1)
        pivot_pct['total_sort_idx'] = total_counts
        pivot_pct = pivot_pct.sort_values('total_sort_idx', ascending=False)
        pivot_pct = pivot_pct.drop('total_sort_idx', axis=1)

        # HTML Tabelle bauen
        html_table = '<div class="table-responsive"><table class="weekly-table">'
        
        # Header
        html_table += '<thead><tr><th style="text-align:left;">Vogelart</th>'
        for col in pivot_pct.columns:
            total_in_week = int(week_totals[col])
            html_table += f'<th>{col}<br><small style="color:#81d4fa;">(∑ {total_in_week})</small></th>'
        html_table += '</tr></thead><tbody>'
        
        # Zeilen
        for species, row in pivot_pct.iterrows():
            # --- Icon Vorbereitung ---
            img_tag = ""
            if species in icon_map:
                # url_for generieren
                img_src = url_for('static', filename=icon_map[species])
                img_tag = f'<img src="{img_src}" class="bird-icon-small">'
            else:
                img_tag = '<div class="bird-icon-placeholder">?</div>'

            # Zelle bauen (Sticky) mit Wrapper
            html_table += f'<tr><td style="text-align:left; font-weight:bold;"><div class="species-wrapper">{img_tag}<span>{species}</span></div></td>'
            
            for col_name, val in row.items():
                absolute_count = int(pivot_counts.at[species, col_name])
                total_in_week = int(week_totals[col_name])
                
                style = 'background-color: transparent;'
                if val > 0:
                    alpha = 0.15 + (val / 50.0) * 0.85 
                    alpha = min(alpha, 1.0) 
                    style = f'background-color: rgba(0, 255, 64, {alpha});'
                
                if total_in_week > 0:
                    tooltip = f"{val:.1f}% ({absolute_count} von {total_in_week} Vögeln)"
                else:
                    tooltip = "0%"
                    
                html_table += f'<td title="{tooltip}" style="{style}"></td>'
            html_table += '</tr>'
        html_table += '</tbody></table></div>'
        
        # Legende hinzufügen
        html_table += """
        <div class="legend-container">
            <div class="legend-item"><div class="legend-box" style="background-color: transparent;"></div><span>0 Sichtungen</span></div>
            <div class="legend-item"><div class="legend-box" style="background-color: rgba(0, 255, 64, 0.2);"></div><span>Wenige</span></div>
            <div class="legend-item"><div class="legend-box" style="background-color: rgba(0, 255, 64, 0.6);"></div><span>Mittel</span></div>
            <div class="legend-item"><div class="legend-box" style="background-color: rgba(0, 255, 64, 1.0);"></div><span>Viele</span></div>
        </div>
        """

    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Wochen-Statistik (Heatmap)</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0"> 
        {{ css_style|safe }}
    </head>
    <body>
        <div class="container" style="max-width: 95%;">
            <div class="header-container">
                <h1>📅 Wochen-Statistik (Heatmap)</h1>
                <div style="margin-top:10px;">
                    <a href="/" class="button-link" style="background:#546e7a;">&laquo; Zurück zur Übersicht</a>
                </div>
            </div>
            
            <p><strong>Relative Häufigkeit (Heatmap):</strong> Je heller das Grün, desto höher der Anteil dieser Art in der jeweiligen Woche.</p>
            
            {{ table_content|safe }}
            
            <br>
            <div class="footer">
                {{ version }}
            </div>
        </div>
    </body>
    </html>
    """
    return render_template_string(html, css_style=CSS_STYLE, table_content=html_table, version=APP_VERSION)

@app.route('/daily')
def daily_stats():
    # Aktuelles Datum und gewähltes Datum bestimmen
    today_date = datetime.datetime.now().date()
    today_str = today_date.strftime("%Y-%m-%d")
    
    selected_date_str = request.args.get('date', today_str)
    
    # Validierung des Formats
    try:
        selected_date = datetime.datetime.strptime(selected_date_str, "%Y-%m-%d").date()
    except ValueError:
        selected_date = today_date
        selected_date_str = today_str
        
    prev_date = (selected_date - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    next_date = (selected_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    is_today = (selected_date_str == today_str)
    
    # 1. Daten für gewähltes Datum aus der Datenbank lesen
    query = f"""
        SELECT 
            CASE WHEN species = 'IGNORED_LOW_CONFIDENCE' THEN 'Unbekannt' ELSE species END as species,
            timestamp
        FROM detections 
        WHERE timestamp LIKE '{selected_date_str}%'
    """
    
    conn = sqlite3.connect(DB_FILE, timeout=10)
    try:
        df = pd.read_sql_query(query, conn)
    except:
        df = pd.DataFrame()
    finally:
        conn.close()

    total_birds_day = len(df) if not df.empty else 0

    chart_url = ""
    if not df.empty:
        # 2. Zeitstempel in Pandas-Datetime konvertieren
        df['datetime'] = pd.to_datetime(df['timestamp'])
        
        # 3. Aggregation (Zusammenfassen der Sichtungen pro Stunde)
        # Wir extrahieren die Stunde (0-23) aus dem Datetime-Objekt
        df['hour'] = df['datetime'].dt.hour
        
        # Erstelle eine Matrix: Stunden als Zeilen (Index), Vogelarten als Spalten
        pivot_df = pd.crosstab(df['hour'], df['species'])
        
        # Fülle leere Stundenladungen auf (damit die X-Achse immer von 0 bis 23 geht)
        pivot_df = pivot_df.reindex(range(24), fill_value=0)

        # 4. Kurvendiagramm zeichnen
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1e1e1e')
        cmap = plt.get_cmap('tab20')
        
        for i, species in enumerate(pivot_df.columns):
            color = '#555555' if species == 'Unbekannt' else cmap(i % 20)
            # marker='o' zeigt Punkte auf der Linie an Variablen
            ax.plot(pivot_df.index, pivot_df[species], marker='o', label=species, color=color, linewidth=2.5)
            
        # Achsen und Layout konfigurieren
        ax.set_title(f'Tagesübersicht - {selected_date_str}', color='white', fontsize=14)
        ax.set_xlabel('Uhrzeit', color='white', fontsize=12)
        ax.set_ylabel('Anzahl der Sichtungen', color='white', fontsize=12)
        
        # X-Achse erzwingen von 00:00 bis 23:00 Uhr
        ax.set_xticks(range(24))
        ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45)
        
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.grid(True, color='#333333', linestyle='--', alpha=0.5)
        
        # Legende neben den Graphen platzieren
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), facecolor='#1e1e1e', labelcolor='white')
        ax.set_facecolor('#1e1e1e')
        
        # Als base64 Bild exportieren
        plt.tight_layout()
        img = io.BytesIO()
        fig.savefig(img, format='png', facecolor='#1e1e1e')
        img.seek(0)
        chart_url = base64.b64encode(img.getvalue()).decode()
        plt.close(fig)

    # 5. HTML Template für die Daily-Seite rendern
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Tages-Statistik (Daily)</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0"> 
        {% if is_today %}
        <meta http-equiv="refresh" content="30"> <!-- Autorefresh alle 30s nur für den heutigen Tag -->
        {% endif %}
        {{ css_style|safe }}
        <style>
            .date-controls {
                display: flex;
                justify-content: center;
                align-items: center;
                gap: 15px;
                margin: 20px 0;
                flex-wrap: wrap;
                background: #263238;
                padding: 15px;
                border-radius: 8px;
                border: 1px solid #37474f;
            }
            .date-input {
                padding: 8px 12px;
                background: #1e1e1e;
                color: #fff;
                border: 1px solid #555;
                border-radius: 4px;
                font-size: 1em;
                font-family: inherit;
            }
            .date-input::-webkit-calendar-picker-indicator {
                filter: invert(1);
                cursor: pointer;
            }
            a.jumper-btn {
                display: inline-block;
                padding: 8px 15px;
                background: #37474f;
                color: white;
                text-decoration: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 0.9em;
                transition: background 0.2s;
            }
            a.jumper-btn:hover { background: #455a64; }
        </style>
    </head>
    <body>
        <div class="container" style="max-width: 95%;">
            <div class="header-container">
                <h1>📈 Tagesübersicht (Verlauf)</h1>
                <div style="margin-top:10px;">
                    <a href="/" class="button-link" style="background:#546e7a;">&laquo; Zurück zur Übersicht</a>
                </div>
            </div>
            
            <div class="date-controls">
                <a href="/daily?date={{ prev_date }}" class="jumper-btn">&laquo; {% if not is_today %}Vorheriger Tag{% else %}Gestern{% endif %}</a>
                
                <input type="date" class="date-input" value="{{ selected_date_str }}" max="{{ today_str }}" id="datePicker">
                
                {% if not is_today %}
                <a href="/daily?date={{ next_date }}" class="jumper-btn">Nächster Tag &raquo;</a>
                <a href="/daily?date={{ today_str }}" class="jumper-btn" style="background: #00838f;">Heute</a>
                {% endif %}
            </div>
            
            <script>
                document.getElementById('datePicker').addEventListener('change', function() {
                    window.location.href = '/daily?date=' + this.value;
                });
            </script>
            
            <p>Sichtungen der Vogelarten im Verlauf des Tages ({% if is_today %}Heute, {% endif %}{{ selected_date_str }}).</p>
            
            <div style="font-size: 1.5em; font-weight: bold; margin: 15px 0; color: #81d4fa;">
                &sum; Gesamtsumme: {{ total_birds_day }}
            </div>
            
            {% if chart_url %}
                <div style="text-align:center; margin-top: 20px;">
                    <img src="data:image/png;base64,{{ chart_url }}" alt="Daily Chart" style="max-width:100%; height:auto; border-radius:8px; border: 1px solid #333;">
                </div>
            {% else %}
                <p style="color: #ff9800; font-weight:bold;">An diesem Tag ({{ selected_date_str }}) wurden keine Vögel gesichtet.</p>
            {% endif %}
            
            <br>
            <div class="footer">
                {{ version }}
            </div>
        </div>
    </body>
    </html>
    """
    return render_template_string(html, 
                                  css_style=CSS_STYLE, 
                                  chart_url=chart_url, 
                                  version=APP_VERSION,
                                  selected_date_str=selected_date_str,
                                  today_str=today_str,
                                  prev_date=prev_date,
                                  next_date=next_date,
                                  is_today=is_today,
                                  total_birds_day=total_birds_day)

def run_flask():
    print(f"Starte Waitress Server auf Port {FLASK_PORT}...")
    serve(app, host='0.0.0.0', port=FLASK_PORT)

# --- HAUPTANWENDUNG (GUI) ---
class AppGUI:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Birds-AI-Classifier (800x448) - {APP_VERSION}")
        self.root.geometry("780x850") 
        self.greylist = load_greylist()
        self.backlog = load_backlog()
        self.blacklist = load_blacklist()
        self.settings = load_settings()
        saved_threshold = self.settings.get("threshold", 70)

        self.autostart_timer = None
        self.in_autostart_mode = False

        self.current_size_mb = 0.0
        self.current_remaining = 0

        self.monitor = FolderMonitor(self.update_log, 
                                     lambda: self.scale_threshold.get(),
                                     self.update_size_display,
                                     lambda: self.rename_var.get(),
                                     lambda: self.delete_var.get(),
                                     lambda: self.greylist_var.get(), 
                                     lambda: self.greylist,
                                     lambda: self.backlog_var.get(),
                                     lambda: self.backlog,
                                     lambda: self.blacklist,
                                     update_duration_callback=self.update_duration_display,
                                     update_remaining_callback=self.update_remaining_display,
                                     get_algo_active_callback=lambda: self.algo_var.get()
                                     )
        
        tk.Label(root, text="Vogel-Überwachung", font=("Arial", 16, "bold")).pack(pady=10)
        
        frame_folder = tk.Frame(root)
        frame_folder.pack(pady=5)
        tk.Label(frame_folder, text="Bilder-Ordner:").pack(side=tk.LEFT, padx=5)
        self.entry_path = tk.Entry(frame_folder, width=40)
        self.entry_path.pack(side=tk.LEFT, padx=5)
        
        last_folder = self.settings.get("last_folder", "")
        if last_folder:
            self.entry_path.insert(0, last_folder)
            self.root.after(500, lambda: self.update_size_display(get_dir_size_mb(last_folder, True)))

        tk.Button(frame_folder, text="Durchsuchen...", command=self.select_folder).pack(side=tk.LEFT)
        self.recursive_var = tk.BooleanVar(value=True)
        tk.Checkbutton(root, text="Unterordner ebenfalls durchsuchen (Rekursiv)", 
                       variable=self.recursive_var, font=("Segoe UI", 10)).pack(pady=2, anchor=tk.W, padx=20)

        frame_settings = tk.LabelFrame(root, text="KI Einstellungen", padx=10, pady=10)
        frame_settings.pack(pady=10, padx=20, fill="x")
        
        tk.Label(frame_settings, text="Mindest-Wahrscheinlichkeit (%):").pack(anchor=tk.W)
        self.scale_threshold = tk.Scale(frame_settings, from_=0, to=100, orient=tk.HORIZONTAL, 
                                        length=400, tickinterval=20, 
                                        command=lambda v: save_setting("threshold", int(v)))
        self.scale_threshold.set(saved_threshold) 
        self.scale_threshold.pack()

        self.rename_var = tk.BooleanVar(value=True)
        tk.Checkbutton(frame_settings, text="Dateien umbenennen (Random + Class)", 
                       variable=self.rename_var, fg="blue", font=("Segoe UI", 10)).pack(anchor=tk.W, pady=5)

        frame_backlog = tk.Frame(frame_settings)
        frame_backlog.pack(anchor=tk.W, pady=5, fill="x")
        self.backlog_var = tk.BooleanVar(value=True)
        cb_bl = tk.Checkbutton(frame_backlog, text="Backlog: Verschieben - Datenbankeintrag", 
                       variable=self.backlog_var, fg="#d2691e", font=("Segoe UI", 10))
        cb_bl.pack(side=tk.LEFT)
        btn_bl_select = tk.Button(frame_backlog, text="[ Backlog Konfig ]", 
                                  command=lambda: self.open_list_config_window("Backlog", self.backlog, save_backlog), 
                                  font=("Arial", 8))
        btn_bl_select.pack(side=tk.LEFT, padx=10)

        frame_greylist = tk.Frame(frame_settings)
        frame_greylist.pack(anchor=tk.W, pady=5, fill="x")
        self.greylist_var = tk.BooleanVar(value=True)
        cb_gl = tk.Checkbutton(frame_greylist, text="Greylist: Löschen + Datenbankeintrag", 
                       variable=self.greylist_var, fg="darkred", font=("Segoe UI", 10))
        cb_gl.pack(side=tk.LEFT)
        btn_gl_select = tk.Button(frame_greylist, text="[ Greylist Konfig ]", 
                                  command=lambda: self.open_list_config_window("Greylist", self.greylist, save_greylist), 
                                  font=("Arial", 8))
        btn_gl_select.pack(side=tk.LEFT, padx=10)

        frame_trash = tk.Frame(frame_settings)
        frame_trash.pack(anchor=tk.W, pady=5, fill="x")
        self.delete_var = tk.BooleanVar(value=True)
        tk.Checkbutton(frame_trash, text="Blacklist: Löschen - Datenbankeintrag", 
                       variable=self.delete_var, fg="red", font=("Segoe UI", 10)).pack(side=tk.LEFT)
        btn_trash_config = tk.Button(frame_trash, text="[ Blacklist Konfig ]", 
                                     command=lambda: self.open_list_config_window("Blacklist (Trash)", self.blacklist, save_blacklist), 
                                     font=("Arial", 8))
        btn_trash_config.pack(side=tk.LEFT, padx=10)

        frame_ping = tk.Frame(frame_settings)
        frame_ping.pack(anchor=tk.W, pady=5, fill="x")
        self.ping_var = tk.BooleanVar(value=True)
        tk.Checkbutton(frame_ping, text="Camera: ping", variable=self.ping_var, fg="darkgreen", font=("Segoe UI", 10)).pack(side=tk.LEFT)

        frame_algo = tk.Frame(frame_settings)
        frame_algo.pack(anchor=tk.W, pady=5, fill="x")
        self.algo_var = tk.BooleanVar(value=self.settings.get("count_algo_active", True))
        tk.Checkbutton(frame_algo, text="Countalgorithm: (Hectic/Normal/Lazy)", 
                       variable=self.algo_var, fg="purple", font=("Segoe UI", 10), 
                       command=lambda: save_setting("count_algo_active", self.algo_var.get())).pack(side=tk.LEFT)

        frame_controls = tk.Frame(root)
        frame_controls.pack(pady=10)
        self.btn_start = tk.Button(frame_controls, text="Überwachung Starten", command=self.handle_start_button_click, bg="#dddddd", width=30)
        self.btn_start.pack(side=tk.LEFT, padx=5)
        
        tk.Button(frame_controls, text="Statistik öffnen", command=self.open_web).pack(side=tk.LEFT, padx=5)
        tk.Button(frame_controls, text="DB Sync", command=self.sync_database_orphans, bg="#ffdd99").pack(side=tk.LEFT, padx=5)
        tk.Button(frame_controls, text="DB Reset", command=self.reset_database, bg="#ff9999").pack(side=tk.LEFT, padx=5)
        
        self.lbl_size = tk.Label(root, text="Ordnergröße: 0.00 MB", font=("Arial", 10, "bold"), fg="blue")
        self.lbl_size.pack(pady=5)
        tk.Label(root, text="Status-Log:").pack(anchor=tk.W, padx=20)
        self.log_text = tk.Text(root, height=10, width=90, state='disabled')
        self.log_text.pack(pady=5)
        tk.Button(root, text="Log leeren", command=self.clear_log, font=("Arial", 8)).pack(pady=2)
        
        self.lbl_duration = tk.Label(root, text="Bild-Verarbeitung: - ms", font=("Arial", 9), fg="gray")
        self.lbl_duration.pack(pady=2)
        
        self.lbl_ram = tk.Label(root, text="RAM: - MB", font=("Arial", 9), fg="gray")
        self.lbl_ram.pack(pady=2)
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        self.update_ram_usage()
        self.init_autostart_sequence()
        self.schedule_ping()

    def schedule_ping(self):
        if self.ping_var.get():
            ip = self.settings.get("camera_ip", "")
            if ip:
                threading.Thread(target=self._do_ping_task, args=(ip,), daemon=True).start()
            else:
                self.update_log("Ping aktiv, aber keine 'camera_ip' in settings.json.")
                self._write_ping_status(True, False)
        else:
            self._write_ping_status(False, False)
            
        self.root.after(300000, self.schedule_ping) # 5 minutes

    def _do_ping_task(self, ip):
        try:
            param = '-n' if platform.system().lower() == 'windows' else '-c'
            command = ['ping', param, '1', ip]
            result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            is_online = (result.returncode == 0)
            status_text = "ONLINE" if is_online else "OFFLINE"
            self.update_log(f"Ping an {ip}: {status_text}")
            self._write_ping_status(True, is_online)
        except Exception as e:
            self.update_log(f"Fehler beim Ping an {ip}: {e}")
            self._write_ping_status(True, False)

    def _write_ping_status(self, active, online):
        try:
            with open("camera_status.json", "w") as f:
                json.dump({"ping_active": active, "camera_online": online}, f)
        except:
            pass

    def init_autostart_sequence(self):
        path = self.entry_path.get()
        if path and os.path.exists(path):
            self.in_autostart_mode = True
            self.countdown_loop(10)
    
    def countdown_loop(self, remaining):
        if not self.in_autostart_mode: return 
        if remaining <= 0:
            self.in_autostart_mode = False
            self.start_monitoring()
        else:
            self.btn_start.config(text=f"Autostart in {remaining}s (Klick = Abbruch)", bg="#ccffcc")
            self.autostart_timer = self.root.after(1000, lambda: self.countdown_loop(remaining - 1))

    def handle_start_button_click(self):
        if self.in_autostart_mode:
            if self.autostart_timer: self.root.after_cancel(self.autostart_timer)
            self.in_autostart_mode = False
            self.btn_start.config(text="Überwachung Starten", bg="#dddddd")
            self.update_log("Autostart manuell abgebrochen.")
        else:
            self.start_monitoring()

    def select_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.entry_path.delete(0, tk.END)
            self.entry_path.insert(0, path)
            save_setting("last_folder", path)
            size = get_dir_size_mb(path, self.recursive_var.get())
            self.update_size_display(size)

    def open_list_config_window(self, title, data_set, save_func):
        win = tk.Toplevel(self.root)
        win.title(f"{title} bearbeiten")
        win.geometry("400x500")
        tk.Label(win, text=f"Arten für {title} auswählen:", font=("Arial", 10, "bold")).pack(pady=10)
        
        known_species = set()
        try:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute("SELECT DISTINCT species FROM detections")
            for row in c.fetchall(): known_species.add(row[0])
            conn.close()
        except: pass
        
        normalized_set = set()
        for s in data_set: normalized_set.add(s)
        db_species = set()
        for s in known_species:
            if s == "IGNORED_LOW_CONFIDENCE": db_species.add("Unbekannt")
            else: db_species.add(s)
        all_species = sorted(list(db_species.union(normalized_set)))
        
        frame_list = tk.Frame(win)
        frame_list.pack(fill="both", expand=True, padx=10, pady=10)
        scrollbar = tk.Scrollbar(frame_list)
        scrollbar.pack(side=tk.RIGHT, fill="y")
        lb = tk.Listbox(frame_list, selectmode=tk.MULTIPLE, yscrollcommand=scrollbar.set)
        lb.pack(side=tk.LEFT, fill="both", expand=True)
        scrollbar.config(command=lb.yview)
        
        for sp in all_species:
            lb.insert(tk.END, sp)
            if sp in normalized_set: lb.selection_set(tk.END) 

        frame_add = tk.Frame(win)
        frame_add.pack(pady=5)
        entry_add = tk.Entry(frame_add)
        entry_add.pack(side=tk.LEFT, padx=5)
        
        def add_manual():
            val = entry_add.get().strip()
            if val:
                lb.insert(tk.END, val)
                lb.selection_set(tk.END)
                entry_add.delete(0, tk.END)

        tk.Button(frame_add, text="Manuell hinzufügen", command=add_manual).pack(side=tk.LEFT)
        def save_and_close():
            selected_indices = lb.curselection()
            new_set = set()
            for i in selected_indices: new_set.add(lb.get(i))
            data_set.clear()
            data_set.update(new_set)
            save_func(data_set)
            messagebox.showinfo("Gespeichert", f"{len(data_set)} Arten gespeichert.")
            win.destroy()
        tk.Button(win, text="Speichern & Schließen", command=save_and_close, bg="#ccffcc", height=2).pack(fill="x", padx=10, pady=10)

    def start_monitoring(self):
        path = self.entry_path.get()
        if not path:
            messagebox.showwarning("Fehler", "Bitte wähle zuerst einen Ordner aus.")
            return
        if self.btn_start['text'].startswith("Überwachung Starten") or self.btn_start['text'].startswith("Autostart"):
            self.monitor.start(path, self.recursive_var.get())
            self.btn_start.config(text="Stoppen", bg="#ffcccc")
            infos = []
            if self.rename_var.get(): infos.append("Rename")
            if self.delete_var.get(): infos.append(f"Blacklist ({len(self.blacklist)} Arten)")
            if self.backlog_var.get(): infos.append(f"Backlog ({len(self.backlog)} Arten)")
            if self.greylist_var.get(): infos.append(f"Greylist ({len(self.greylist)} Arten)")
            info_str = ", ".join(infos) if infos else "Nur Erkennung"
            self.update_log(f"Service gestartet: {info_str}")
        else:
            self.monitor.stop()
            self.btn_start.config(text="Überwachung Starten", bg="#dddddd")
            self.update_log("Service gestoppt.")

    def sync_database_orphans(self):
        folder = self.entry_path.get()
        if not folder or not os.path.exists(folder):
            messagebox.showwarning("Fehler", "Bitte Ordner wählen.")
            return
        if not messagebox.askyesno("Sicherheitsabfrage", f"Möchtest du wirklich die Datenbank mit dem Ordner '{folder}' abgleichen?\n\nWARNUNG: Einträge der Greylist (bereits gelöschte Dateien) werden hiermit aus der Statistik entfernt!"):
            return
        self.update_log("Starte Datenbank-Sync...")
        real_files_set = set()
        recursive = self.recursive_var.get()
        extensions = ['*.jpg', '*.jpeg', '*.JPG', '*.png']
        path_obj = Path(folder)
        try:
            for ext in extensions:
                iterator = path_obj.rglob(ext) if recursive else path_obj.glob(ext)
                for f in iterator: real_files_set.add(f.name)
            self.update_log(f"Dateien auf Platte: {len(real_files_set)}")
            conn = sqlite3.connect(DB_FILE, timeout=10)
            c = conn.cursor()
            c.execute("SELECT filename FROM detections")
            db_rows = c.fetchall()
            deleted_count = 0
            for row in db_rows:
                fname = row[0]
                if fname not in real_files_set:
                    c.execute("DELETE FROM detections WHERE filename = ?", (fname,))
                    deleted_count += 1
            conn.commit()
            c.execute("VACUUM")
            conn.close()
            self.update_log(f"Sync fertig: {deleted_count} verwaiste Einträge entfernt.")
            messagebox.showinfo("Sync Fertig", f"{deleted_count} Einträge gelöscht.")
        except Exception as e:
            self.update_log(f"Sync Fehler: {e}")

    def update_size_display(self, size_mb):
        self.current_size_mb = size_mb
        self._refresh_size_label()

    def update_remaining_display(self, remaining):
        self.current_remaining = remaining
        self._refresh_size_label()

    def _refresh_size_label(self):
        def _update():
            color = "green"
            if self.current_size_mb > 100: color = "orange"
            if self.current_size_mb > 500: color = "red"
            
            text = f"Ordnergröße: {self.current_size_mb:.2f} MB"
            if self.current_remaining > 0:
                text += f" ({self.current_remaining} Bilder ausstehend)"
                
            self.lbl_size.config(text=text, fg=color)
        self.root.after(0, _update)

    def update_duration_display(self, duration_ms):
        def _update():
            if duration_ms > 1500:
                color = "red"
            elif duration_ms > 1000:
                color = "orange"
            else:
                color = "gray"
            self.lbl_duration.config(text=f"Bild-Verarbeitung: {duration_ms} ms", fg=color)
        self.root.after(0, _update)

    def update_ram_usage(self):
        try:
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            ram_mb = mem_info.rss / (1024 * 1024)
            self.lbl_ram.config(text=f"RAM: {ram_mb:.1f} MB")
        except:
            pass
        self.root.after(2000, self.update_ram_usage) # Update every 2 seconds

    def open_web(self):
        webbrowser.open(f"http://localhost:{FLASK_PORT}")

    def reset_database(self):
        if messagebox.askyesno("Sicherheitsabfrage", "Möchtest du wirklich ALLE Daten in der Datenbank löschen?"):
            try:
                conn = sqlite3.connect(DB_FILE, timeout=10)
                c = conn.cursor()
                c.execute("DELETE FROM detections")
                conn.commit()
                c.execute("VACUUM")
                conn.close()
                self.update_log("Datenbank geleert.")
                messagebox.showinfo("Erfolg", "Datenbank wurde geleert.")
            except Exception as e: messagebox.showerror("Fehler", f"{e}")

    def clear_log(self):
        self.log_text.config(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state='disabled')

    def update_log(self, message):
        self.log_text.config(state='normal')
        try:
            line_count = int(self.log_text.index('end-1c').split('.')[0])
            if line_count > 100: self.log_text.delete(1.0, tk.END)
        except: pass 
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')
        
    def on_close(self):
        self.monitor.stop()
        self.root.destroy()
        os._exit(0)

if __name__ == "__main__":
    init_db()
    flask_process = multiprocessing.Process(target=run_flask, daemon=True)
    flask_process.start()
    root = tk.Tk()
    app_gui = AppGUI(root)
    root.mainloop()