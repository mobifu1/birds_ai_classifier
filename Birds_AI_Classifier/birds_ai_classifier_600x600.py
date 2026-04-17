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
import requests
import platform

# --- NEU: Pillow für EXIF-Daten ---
from PIL import Image, ExifTags, ImageFilter

# --- NEU: Production Server Import ---
from waitress import serve

# --- WICHTIG: Matplotlib Einstellung ---
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# System

# Web & Data
from flask import Flask, render_template_string, jsonify, request, send_file, url_for, render_template, send_from_directory
import pandas as pd

# AI (TensorFlow)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf

# InceptionV3 Importe
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions # type: ignore
from tensorflow.keras.preprocessing import image as tf_image # type: ignore
from tensorflow.keras.models import load_model 
import numpy as np

# --- KONFIGURATION ---
# DEBUG wird jetzt über settings.json gesteuert (Schlüssel: 'debug_active')
DEBUG_FILE = "debug_log.txt"

DB_FILE = "birds_stats.db"
WEATHER_CONFIG_FILE = "weather_config.json"
GREYLIST_FILE = "greylist.json" 
BLACKLIST_FILE = "blacklist.json" 
BACKLOG_FILE = "backlog.json"
ACTIONLIST_FILE = "actionlist.json"
ACTION_CONFIG_FILE = "action_config.json"
SETTINGS_FILE = "settings.json" 
RECORDS_FILE = "records.json"
FLASK_PORT = 5000
CHECK_INTERVAL_SECONDS = 5 
STATIC_FOLDER = "static" 
LAST_IMG_NAME = "last_detection.jpg" 

# Globals for weather caching
last_weather_check = 0
last_weather_temp = None

# --- MASK PARAMETER ---
MASK_TOP = 0  
MASK_BOTTOM = 0

# --- KAMERA AUFLÖSUNG ---
# Das Bild wird in 3 quadratische Teile (links, mitte, rechts) aufgeteilt,
# die jeweils auf die Modell-Größe skaliert und analysiert werden.

# --- MODELL ZIELGRÖSSE ---
MODEL_TARGET_SIZE = 600  # Das trainierte Modell erwartet 600x600 Pixel

# --- DEBUG: Bildbearbeitung Ergebnis speichern ---
# Wenn True, wird das fertig bearbeitete Bild (nach Letterboxing, Resize und Masking)
# in den Ordner 'debug_live_masking' kopiert, bevor es dem Modell übergeben wird.
# So kann man genau sehen, was das KI-Modell als Eingabe erhält.
debug_result_bildbearbeitung = True

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

def load_records():
    if os.path.exists(RECORDS_FILE):
        try:
            with open(RECORDS_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {}

def save_records(data):
    try:
        with open(RECORDS_FILE, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        print(f"Fehler beim Speichern der Records: {e}")

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
    defaults = {"Unbekannt"}
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

def load_actionlist():
    if os.path.exists(ACTIONLIST_FILE):
        try:
            with open(ACTIONLIST_FILE, 'r', encoding='utf-8') as f:
                return set(json.load(f))
        except:
            return set()
    return set()

def save_actionlist(actionlist_set):
    try:
        with open(ACTIONLIST_FILE, 'w', encoding='utf-8') as f:
            json.dump(list(actionlist_set), f, ensure_ascii=False, indent=2)
    except: pass

def load_action_config():
    defaults = {"start_webhook": "", "stop_webhook": "", "duration": 10, "max_age": 120}
    if os.path.exists(ACTION_CONFIG_FILE):
        try:
            with open(ACTION_CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return {**defaults, **config}
        except:
            return defaults
    return defaults

def save_action_config(config):
    try:
        with open(ACTION_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
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
        self.custom_model_path = "my_birds_modell_600x600.keras"
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
            # Bild laden
            img = Image.open(img_path).convert('RGB')
            w, h = img.size
            target_size = MODEL_TARGET_SIZE
            
            # Bild in 3 quadratische Teile (links, mitte, rechts) aufteilen
            if w > h:
                crop_size = h
                left_img = img.crop((0, 0, crop_size, crop_size))
                mid_img = img.crop(((w - crop_size) // 2, 0, (w + crop_size) // 2, crop_size))
                right_img = img.crop((w - crop_size, 0, w, crop_size))
            else:
                # Fallback, falls das Bild nicht im Querformat ist
                crop_size = w
                left_img = img.crop((0, 0, crop_size, crop_size))
                mid_img = img.crop((0, (h - crop_size) // 2, crop_size, (h + crop_size) // 2))
                right_img = img.crop((0, h - crop_size, crop_size, h))
                
            parts = [left_img, mid_img, right_img]
            part_names = ["links", "mitte", "rechts"]
            
            best_species = "Unbekannt"
            best_conf = 0.0
            best_conf2 = 0.0

            for i, part in enumerate(parts):
                # Teil auf 600x600 (bzw. MODEL_TARGET_SIZE) rezisen
                part_resized = part.resize((target_size, target_size), Image.LANCZOS)
                x = tf_image.img_to_array(part_resized)
                
                # Masking anwenden
                x[:MASK_TOP, :, :] = 0
                img_h = x.shape[0]
                if MASK_BOTTOM > 0:
                    x[img_h-MASK_BOTTOM:, :, :] = 0

                # --- DEBUG: Bearbeitetes Bild speichern ---
                if debug_result_bildbearbeitung:
                    try:
                        debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_live_masking")
                        os.makedirs(debug_dir, exist_ok=True)
                        debug_img = Image.fromarray(x.astype('uint8'))
                        debug_filename = f"debug_{part_names[i]}_{os.path.basename(img_path)}"
                        debug_img.save(os.path.join(debug_dir, debug_filename))
                    except Exception as e:
                        print(f"Debug-Bild {part_names[i]} konnte nicht gespeichert werden: {e}")

                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                
                preds = self.model.predict(x, verbose=0)

                if self.use_custom:
                    sorted_indices = np.argsort(preds[0])[::-1]
                    best_index = sorted_indices[0]
                    confidence = float(preds[0][best_index])
                    confidence2 = float(preds[0][sorted_indices[1]]) if len(preds[0]) > 1 else 0.0
                    label_name = self.labels_map.get(best_index, "Unbekannt")
                    species = label_name.replace('_', ' ').title()
                else:
                    results = decode_predictions(preds, top=2)[0]
                    english_label = results[0][1]
                    confidence = float(results[0][2])
                    confidence2 = float(results[1][2]) if len(results) > 1 else 0.0
                    translated_label = self.translations.get(english_label, english_label)
                    species = translated_label.replace('_', ' ').title()
                    
                # Bestes Ergebnis speichern
                if confidence > best_conf:
                    best_conf = confidence
                    best_conf2 = confidence2
                    best_species = species

            return best_species, best_conf, best_conf2
        except Exception as e:
            return "Fehler", 0.0

# --- HINTERGRUND ÜBERWACHUNG ---
class FolderMonitor:
    def __init__(self, update_log_callback, get_threshold_callback, get_guess_threshold_callback, update_size_callback, 
                 get_delete_callback, 
                 get_greylist_active_callback, get_greylist_callback,
                 get_margin_threshold_callback,
                 get_backlog_active_callback, get_backlog_callback,
                 get_blacklist_callback, update_duration_callback=None,
                 update_remaining_callback=None, get_algo_active_callback=None,
                 get_webhook_active_callback=None,
                 get_actionlist_callback=None, get_action_config_callback=None): 
        self.running = False
        self.folder_path = ""
        self.recursive = False 
        self.ai = None
        self.log_callback = update_log_callback
        self.get_threshold = get_threshold_callback
        self.get_guess_threshold = get_guess_threshold_callback
        self.get_margin_threshold = get_margin_threshold_callback if get_margin_threshold_callback else lambda: 10
        self.update_size_callback = update_size_callback
        self.get_delete_enabled = get_delete_callback
        self.get_greylist_active = get_greylist_active_callback
        self.get_greylist = get_greylist_callback
        self.get_backlog_active = get_backlog_active_callback
        self.get_backlog_callback = get_backlog_callback
        self.get_blacklist = get_blacklist_callback
        self.get_algo_active = get_algo_active_callback if get_algo_active_callback else lambda: False
        self.get_webhook_active = get_webhook_active_callback if get_webhook_active_callback else lambda: True
        self.get_actionlist_callback = get_actionlist_callback if get_actionlist_callback else lambda: set()
        self.get_action_config_callback = get_action_config_callback if get_action_config_callback else lambda: {}
        self.update_duration_callback = update_duration_callback
        self.update_remaining_callback = update_remaining_callback
        self.thread = None
        self.normal_timers = {}
        self.lazy_occupier = None
        self.categories = load_categories()
        self.action_active = False

    def trigger_webhook_action(self, start_url, stop_url, duration):
        try:
            if start_url:
                self.log_callback(f"🚀 Starte Webhook Aktion: {start_url}")
                try: requests.get(start_url, timeout=5)
                except Exception as e: self.log_callback(f"⚠️ Fehler bei Start-Webhook: {e}")
            
            if duration > 0:
                time.sleep(duration)
                
            if stop_url:
                self.log_callback(f"🛑 Stoppe Webhook Aktion: {stop_url}")
                try: requests.get(stop_url, timeout=5)
                except Exception as e: self.log_callback(f"⚠️ Fehler bei Stop-Webhook: {e}")
        finally:
            self.action_active = False

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
                
                # Futterplatz Status exportieren
                try:
                    status = {
                        "occupy": self.lazy_occupier,
                        "time_locks": {}
                    }
                    now = time.time()
                    for sp, last_seen in list(self.normal_timers.items()):
                        remaining = int(120 - (now - last_seen))
                        if remaining > 0:
                            status["time_locks"][sp] = remaining
                    with open("futterplatz_status.json", "w", encoding="utf-8") as f:
                        json.dump(status, f)
                except Exception as e:
                    pass
                    
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
            files_to_process = sorted(new_files, key=lambda f: get_original_date(str(f)))
            current_threshold = self.get_threshold() 
            current_guess_threshold = self.get_guess_threshold()
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
                if not os.path.exists(file_path): 
                    self.log_callback(f"[{file_path.name}] ⚠️ Datei nicht mehr vorhanden. Überspringe...")
                    continue 

                # Prüfen ob die Datei noch beschrieben wird (z.B. von FTP)
                is_locked = True
                for _ in range(4): # max 2 Sekunden warten
                    try:
                        # Exklusiver Schreibzugriff schlägt auf Windows fehl, wenn die Datei gelockt ist
                        with open(file_path, 'a'):
                            is_locked = False
                            break
                    except OSError:
                        time.sleep(0.5)

                if is_locked:
                    self.log_callback(f"[{file_path.name}] ⏳ Datei wird noch hochgeladen (gesperrt). Warte auf nächsten Durchlauf.")
                    continue

                start_time = time.time()
                try:
                    res = self.ai.analyze_image(str(file_path))
                    if len(res) == 3:
                        species, conf, conf2 = res
                    else:
                        species, conf = res
                        conf2 = 0.0
                except Exception as e:
                    self.log_callback(f"[{file_path.name}] ❌ Fehler in analyze_image: {e}")
                    continue
                duration_ms = int((time.time() - start_time) * 1000)
                if self.update_duration_callback:
                    self.update_duration_callback(duration_ms)
                
                if species == "Fehler":
                    self.log_callback(f"[{file_path.name}] ❌ Bild konnte nicht gelesen werden (evtl. defekt oder 0 Bytes).")
                    try:
                        app_dir = Path(os.path.abspath(os.path.dirname(__file__)))
                        error_dir = app_dir / "error_images"
                        error_dir.mkdir(exist_ok=True)
                        error_path = error_dir / file_path.name
                        shutil.move(str(file_path), str(error_path))
                        self.log_callback(f"[{file_path.name}] ➡️ In Ordner 'error_images' verschoben, um Endlosschleife zu verhindern.")
                    except Exception as e:
                        self.log_callback(f"[{file_path.name}] ⚠️ Bild ist gelockt und kann nicht verschoben werden: {e}")
                        pass # Falls es noch geschrieben wird und gelockt ist, bleibt es für den nächsten Versuch
                    continue

                
                gc.collect() 
                
                try:
                    target_img = os.path.join(STATIC_FOLDER, LAST_IMG_NAME)
                    shutil.copy2(file_path, target_img)
                    with open(os.path.join(STATIC_FOLDER, 'last_detection_filename.txt'), 'w', encoding='utf-8') as f:
                        f.write(file_path.name)
                except: pass

                conf_percent = int(conf * 100)
                conf2_percent = int(conf2 * 100)
                margin_percent = conf_percent - conf2_percent
                timestamp = get_original_date(str(file_path))
                
                # 1. Classification (Unbekannt oder Klasse)
                if conf_percent < current_guess_threshold:
                    species = "Unbekannt"
                elif conf_percent < current_threshold:
                    current_margin_threshold = self.get_margin_threshold()
                    if margin_percent >= current_margin_threshold:
                        species = species.replace(" ", "_")
                        self.log_callback(f"[{file_path.name}] 🎯 Vorsprung reicht ({margin_percent}%), akzeptiert als: {species}")
                    else:
                        top1_species = species.replace(" ", "_")
                        species = "Vermutung"
                        self.log_callback(f"[{file_path.name}] 🤔 Vermutung ({top1_species} {conf_percent}%)")
                else:
                    species = species.replace(" ", "_")
                
                # 2. Files umbenennen (Random + Class)
                file_ext = file_path.suffix
                clean_species = species
                while True:
                    rand_id = random.randint(100000, 999999)
                    new_name = f"{rand_id}_{clean_species}{file_ext}"
                    new_full_path = file_path.parent / new_name
                    if not new_full_path.exists():
                        break
                        
                old_name = file_path.name
                try:
                    os.rename(file_path, new_full_path)
                    final_filename = new_name
                    file_path = new_full_path
                    try:
                        tracker_path = os.path.join(STATIC_FOLDER, 'last_detection_filename.txt')
                        if os.path.exists(tracker_path):
                            with open(tracker_path, 'r', encoding='utf-8') as f:
                                tracked = f.read().strip()
                            if tracked == old_name:
                                with open(tracker_path, 'w', encoding='utf-8') as f:
                                    f.write(new_name)
                    except: pass
                    self.log_callback(f"[{old_name}] ✏️ Umbenannt zu: {final_filename}")
                except Exception as e:
                    self.log_callback(f"[{file_path.name}] ❌ Fehler beim Umbenennen: {e}")
                    final_filename = file_path.name

                # --- NEU: Webhook Action Trigger ---
                current_actionlist = self.get_actionlist_callback()
                if species in current_actionlist:
                    if not self.get_webhook_active():
                        self.log_callback(f"[{final_filename}] ⏭️ Webhook deaktiviert in UI.")
                    elif self.action_active:
                        self.log_callback(f"[{final_filename}] ⏭️ Webhook übersprungen: Aktion läuft bereits.")
                    else:
                        try:
                            img_time = time.mktime(datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").timetuple())
                            age_seconds = time.time() - img_time
                            max_age = int(self.get_action_config_callback().get("max_age", 120))
                            if age_seconds <= max_age:
                                current_action_config = self.get_action_config_callback()
                                start_webhook = current_action_config.get("start_webhook", "")
                                stop_webhook = current_action_config.get("stop_webhook", "")
                                try:
                                    duration = int(current_action_config.get("duration", 10))
                                except ValueError:
                                    duration = 10
                                
                                self.action_active = True
                                threading.Thread(target=self.trigger_webhook_action, args=(start_webhook, stop_webhook, duration), daemon=True).start()
                                self.log_callback(f"[{final_filename}] 🚨 {species} erkannt! Aktion gestartet (Laufzeit: {duration}s).")
                            else:
                                self.log_callback(f"[{final_filename}] ⏭️ Webhook übersprungen: Bild ist {int(age_seconds)}s alt (max {max_age}s).")
                        except Exception as e:
                            self.log_callback(f"[{final_filename}] ⚠️ Fehler bei Altersberechnung für Webhook: {e}")

                # 3. Count Algorithm (Algorithmus)
                algo_active = self.get_algo_active()
                if algo_active and conf_percent >= current_threshold and species != "Unbekannt":
                    try:
                        img_time = time.mktime(datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").timetuple())
                    except Exception:
                        img_time = time.time()
                        
                    cat = self.categories.get(species.replace("_", " "), "normal").lower()
                    ignore_type = None
                    
                    if cat == "lazy":
                        if self.lazy_occupier == species:
                            ignore_type = "Occupier"
                            self.log_callback(f"[{final_filename}] ⏳ {species} -> Ignoriert (Algorithmus: Lazy Occupier)")
                        else:
                            self.lazy_occupier = species
                                    
                    elif cat == "normal":
                        self.lazy_occupier = None 
                        last_seen = self.normal_timers.get(species, 0)
                        if (img_time - last_seen) < 120:
                            ignore_type = "Timers"
                            self.log_callback(f"[{final_filename}] ⏳ {species} -> Ignoriert (Algorithmus: Normal < 2 Min)")
                        else:
                            self.normal_timers[species] = img_time
                            
                    elif cat == "hectic":
                        self.lazy_occupier = None 
                        
                    if ignore_type: # "Occupier" or "Timers"
                        old_name = file_path.name
                        species = ignore_type
                        clean_species = species
                        file_ext = file_path.suffix
                        while True:
                            rand_id = random.randint(100000, 999999)
                            new_name = f"{rand_id}_{clean_species}{file_ext}"
                            new_full_path = file_path.parent / new_name
                            if not new_full_path.exists():
                                break
                        try:
                            os.rename(file_path, new_full_path)
                            final_filename = new_name
                            file_path = new_full_path
                            try:
                                tracker_path = os.path.join(STATIC_FOLDER, 'last_detection_filename.txt')
                                if os.path.exists(tracker_path):
                                    with open(tracker_path, 'r', encoding='utf-8') as f:
                                        tracked = f.read().strip()
                                    if tracked == old_name:
                                        with open(tracker_path, 'w', encoding='utf-8') as f:
                                            f.write(new_name)
                            except: pass
                            self.log_callback(f"[{old_name}] ✏️ Umbenannt zu: {final_filename}")
                        except Exception as e:
                            self.log_callback(f"[{file_path.name}] ❌ Fehler beim Umbenennen: {e}")

                # 4. Backlog
                backlog_active = self.get_backlog_active()
                if backlog_active and species in current_backlog:
                    try:
                        app_dir = Path(os.path.abspath(os.path.dirname(__file__)))
                        backlog_dir = app_dir / "backlog"
                        backlog_dir.mkdir(exist_ok=True)
                        target_backlog_path = backlog_dir / file_path.name
                        shutil.move(str(file_path), str(target_backlog_path))
                        
                        try:
                            tracker_path = os.path.join(STATIC_FOLDER, 'last_detection_filename.txt')
                            if os.path.exists(tracker_path):
                                with open(tracker_path, 'r', encoding='utf-8') as f:
                                    tracked = f.read().strip()
                                if tracked == file_path.name:
                                    with open(tracker_path, 'w', encoding='utf-8') as f:
                                        f.write(target_backlog_path.name)
                        except: pass
                        self.log_callback(f"[{final_filename}] ⏳ {species} -> Ins Backlog verschoben")
                    except Exception as e:
                        self.log_callback(f"[{final_filename}] ❌ Fehler beim Verschieben (Backlog): {e}")
                    continue
                
                # 5. Greylist
                greylist_active = self.get_greylist_active()
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
                
                # 6. Blacklist
                is_blacklisted = (species in current_blacklist) 
                should_delete_trash = delete_unsure_active and conf_percent < current_threshold
                if is_blacklisted or should_delete_trash:
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            reason = "Blacklist" if is_blacklisted else "Unsicher"
                            self.log_callback(f"[{final_filename}] 🗑️ {species} -> Gelöscht ({reason})")
                    except Exception as e:
                        self.log_callback(f"[{final_filename}] ❌ Fehler beim Löschen (Trash): {e}")
                    continue 

                # 7. Speichern in DB
                try:
                    c.execute("INSERT INTO detections (filename, species, timestamp, confidence) VALUES (?, ?, ?, ?)",
                              (final_filename, species, timestamp, conf))
                    conn.commit()
                    self.log_callback(f"[{final_filename}] ✅ {species} ({conf_percent}%) -> Gespeichert")
                except sqlite3.IntegrityError: 
                    self.log_callback(f"[{final_filename}] ⚠️ Bild bereits in der Datenbank (Integrity Error).") 
            
            if self.update_remaining_callback:
                self.update_remaining_callback(0)
        conn.close()

# --- NEU: VERSION ---
APP_VERSION = "Version 1.2-RC"

# --- WEB SERVER (FLASK) ---
app = Flask(__name__)

# --- STYLE CSS CONSTANT ---
CSS_STYLE = """
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

# --- VORHERSAGE FUNKTIONEN ---
def load_weather_config():
    if os.path.exists(WEATHER_CONFIG_FILE):
        try:
            with open(WEATHER_CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Fehler beim Laden von {WEATHER_CONFIG_FILE}: {e}")
    return {}

def get_prediction_db_data(days=7):
    conn = sqlite3.connect(DB_FILE, timeout=10)
    cutoff_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d 00:00:00')
    try:
        query = f"""
            SELECT 
                CASE WHEN species = 'IGNORED_LOW_CONFIDENCE' THEN 'Unbekannt' ELSE species END as species,
                timestamp,
                confidence
            FROM detections 
            WHERE timestamp >= '{cutoff_date}'
        """
        df = pd.read_sql_query(query, conn)
        if not df.empty:
            df['datetime'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['datetime'].dt.hour
            df['date'] = df['datetime'].dt.date
    except Exception as e:
        print(f"Datenbankfehler: {e}")
        df = pd.DataFrame()
    finally:
        conn.close()
    return df

def predict_rush_hour(df):
    if df.empty: return None, ""
    unique_days = df['date'].nunique()
    if unique_days == 0: unique_days = 1
    hourly_counts = df.groupby('hour').size() / unique_days
    fig, ax = plt.subplots(figsize=(8, 4), facecolor='#1e1e1e')
    ax.bar(hourly_counts.index, hourly_counts.values, color='#4fc3f7')
    ax.set_title('Durchschnittliche Besuche pro Stunde (Rush-Hour)', color='white')
    ax.set_xlabel('Uhrzeit', color='white')
    ax.set_ylabel('Ø Besuche', color='white')
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45)
    ax.tick_params(colors='white')
    ax.set_facecolor('#1e1e1e')
    plt.tight_layout()
    img = io.BytesIO()
    fig.savefig(img, format='png', facecolor='#1e1e1e')
    img.seek(0)
    chart_url = base64.b64encode(img.getvalue()).decode()
    plt.close(fig)
    busiest_hour = hourly_counts.idxmax() if not hourly_counts.empty else "N/A"
    return busiest_hour, chart_url

def predict_species_probability(df, target_hour=None):
    if df.empty: return {}, ""
    if target_hour is None: target_hour = datetime.datetime.now().hour
    hour_df = df[df['hour'] == target_hour]
    if hour_df.empty: return {}, ""
    species_counts = hour_df['species'].value_counts()
    total = species_counts.sum()
    probabilities = {sp: (count / total * 100) for sp, count in species_counts.items()}
    top5 = species_counts.head(5)
    if top5.sum() < total: top5['Andere'] = total - top5.sum()
    fig, ax = plt.subplots(figsize=(6, 4), facecolor='#1e1e1e')
    ax.pie(top5.values, labels=top5.index, autopct='%1.1f%%', startangle=90, textprops={'color':"w"})
    ax.set_title(f'Wahrscheinlichkeit bis um {target_hour:02d}:00 Uhr', color='white')
    plt.tight_layout()
    img = io.BytesIO()
    fig.savefig(img, format='png', facecolor='#1e1e1e')
    img.seek(0)
    chart_url = base64.b64encode(img.getvalue()).decode()
    plt.close(fig)
    return probabilities, chart_url

def analyze_disturbance(df):
    disturbers = ['Elster', 'Eichelhäher', 'Katze', 'Sperber']
    if df.empty: return "Keine Daten."
    df_sorted = df.sort_values('datetime').reset_index(drop=True)
    events = []
    for i in range(len(df_sorted)):
        row = df_sorted.iloc[i]
        if row['species'] in disturbers:
            for j in range(i+1, len(df_sorted)):
                next_row = df_sorted.iloc[j]
                if next_row['species'] not in disturbers:
                    gap = (next_row['datetime'] - row['datetime']).total_seconds() / 60.0
                    events.append({'disturber': row['species'], 'gap_mins': gap})
                    break
    if not events: return "Bisher keine Störereignisse nachgewiesen."
    avg_gap = sum([e['gap_mins'] for e in events]) / len(events)
    return f"Im Schnitt wird das Futterhaus nach einem Störereignis für {avg_gap:.1f} Minuten gemieden."

def analyze_anomalies(df, df_reference=None):
    """
    Erkennt Anomalien (Massenansturm / Plötzliches Ausbleiben).

    df           – Daten des vom Nutzer gewählten Zeitraums (z.B. 7–14 Tage)
    df_reference – Optionaler längerer Referenz-DataFrame (feste 30 Tage) für
                   stabile Durchschnittswerte. Falls None, wird df selbst verwendet.
    """
    if df.empty:
        return "Keine Daten für eine Anomalie-Erkennung vorhanden."

    SKIP = {"Unbekannt", "IGNORED_LOW_CONFIDENCE", "Hintergrund"}

    # Sicherstellen, dass 'date' vorhanden ist
    for frame in [df, df_reference]:
        if frame is not None and 'date' not in frame.columns:
            frame['date'] = pd.to_datetime(frame['timestamp']).dt.date

    # --- Referenz-DF für Durchschnitte (bevorzugt der lange 30-Tage-DF) ---
    ref = df_reference if (df_reference is not None and not df_reference.empty) else df

    ref_unique_days = ref['date'].nunique()
    if ref_unique_days < 3:
        return "Es werden mindestens 3 Tage an Daten benötigt, um sinnvolle Durchschnittswerte für Anomalien zu berechnen."

    unique_days = df['date'].nunique()

    anomalies = []

    # Tageszählungen aus dem aktuellen (kurzen) DF – für Massenansturm & recent_days
    daily_counts = df.groupby(['date', 'species']).size().unstack(fill_value=0)

    # Tageszählungen aus dem Referenz-DF – für stabile Durchschnitte
    ref_daily_counts = ref.groupby(['date', 'species']).size().unstack(fill_value=0)
    avg_daily_counts = ref_daily_counts.mean()  # stabile Basis über 30 Tage

    # 1. Massenansturm: letzten 2 Tage im aktuellen Fenster prüfen
    recent_days_short = sorted(daily_counts.index)[-2:]

    for day in recent_days_short:
        for species in daily_counts.columns:
            if species in SKIP:
                continue
            count_on_day = daily_counts.loc[day, species]
            avg_count = avg_daily_counts.get(species, 0)
            # Kriterien: typisch >= 1/Tag, heute >= 3× Schnitt, absolut > 10
            if avg_count >= 1 and count_on_day >= (avg_count * 3) and count_on_day > 10:
                day_str = "Heute" if day == datetime.datetime.now().date() else f"Am {day.strftime('%d.%m.')}"
                anomalies.append(
                    f"<strong>🚨 Massenansturm:</strong> {day_str} wurde die Art "
                    f"<em>{species}</em> ungewöhnlich oft gesichtet "
                    f"({int(count_on_day)}x, Normal: ~{int(avg_count)}x pro Tag)."
                )

    # 2. Plötzliches Ausbleiben
    # Braucht mind. 5 Referenztage für eine verlässliche Aussage
    if ref_unique_days >= 5:
        # "Häufige Arten": im Referenz-Durchschnitt mind. 5× pro Tag
        common_species = avg_daily_counts[avg_daily_counts >= 5].index

        # Aktuelle Abwesenheitsprüfung: letzte 7 Tage im kurzen DF
        recent_days_long = sorted(daily_counts.index)[-7:]
        today = datetime.datetime.now().date()

        for species in common_species:
            if species in SKIP:
                continue

            # Prüfe: War die Art in den letzten 7 Tagen an KEINEM Tag sichtbar?
            if species in daily_counts.columns:
                recent_counts = [daily_counts.loc[d, species] for d in recent_days_long]
                species_missing = all(c == 0 for c in recent_counts)
            else:
                # Art kommt im aktuellen DF überhaupt nicht vor → definitiv abwesend
                species_missing = True

            if species_missing:
                # Berechne, seit wie vielen Tagen die Art fehlt (im Referenz-DF)
                if species in ref_daily_counts.columns:
                    all_dates_with_species = ref_daily_counts.index[
                        ref_daily_counts[species] > 0
                    ].tolist()
                    if all_dates_with_species:
                        last_seen = max(all_dates_with_species)
                        days_missing = (today - last_seen).days
                        since_str = f" (zuletzt gesehen am {last_seen.strftime('%d.%m.%Y')}, vor {days_missing} Tag{'en' if days_missing != 1 else ''})"
                    else:
                        since_str = ""
                else:
                    since_str = ""

                anomalies.append(
                    f"<strong>⚠️ Plötzliches Ausbleiben:</strong> Die sonst häufige Art "
                    f"<em>{species}</em> (Referenzdurchschnitt: ~{int(avg_daily_counts[species])}x/Tag) "
                    f"wurde in den letzten {len(recent_days_long)} Tagen gar nicht mehr gesichtet"
                    f"{since_str}."
                )

    if not anomalies:
        return "<span style='color: #69f0ae;'>Keine auffälligen Anomalien erkannt. Das Verhalten entspricht dem Durchschnitt.</span>"

    return "<br><br>".join(anomalies)


def analyze_diversification(df):
    if df.empty:
        return "Bisher keine Sichtungen in diesem Zeitraum erfasst.", "Keine Daten ➖"
    
    daily_species = df.groupby('date')['species'].nunique()
    total_unique = df['species'].nunique()
    
    if len(daily_species) < 2:
        return f"Aktuell <strong>{total_unique}</strong> verschiedene Arten gesichtet.", "Zu wenig Daten für einen Trend ➖"
    
    n_days = len(daily_species)
    half = n_days // 2
    
    first_half_avg = daily_species.iloc[:half].mean()
    second_half_avg = daily_species.iloc[half:].mean()
    
    if pd.isna(first_half_avg) or pd.isna(second_half_avg) or first_half_avg == 0:
        trend_html = "<span style='color: #ffd54f;'>Zu wenig Daten ➖</span>"
    elif second_half_avg > first_half_avg * 1.05:
        trend_html = "<span style='color: #69f0ae;'>Zunehmend 📈</span>"
    elif second_half_avg < first_half_avg * 0.95:
        trend_html = "<span style='color: #ff5252;'>Abnehmend 📉</span>"
    else:
        trend_html = "<span style='color: #ffd54f;'>Gleichbleibend ➖</span>"
        
    text = f"Insgesamt <strong>{total_unique}</strong> verschiedene Arten im ausgewählten Zeitraum ({len(daily_species)} Tage ausgewertet)."
    return text, trend_html

def analyze_absolute_visitors(df):
    if df.empty:
        return "Bisher keine Sichtungen in diesem Zeitraum erfasst.", "Keine Daten ➖"
    
    daily_visitors = df.groupby('date').size()
    total_visitors = int(daily_visitors.sum())
    
    if len(daily_visitors) < 2:
        return f"Aktuell <strong>{total_visitors}</strong> absolute Besucher erfasst.", "Zu wenig Daten für einen Trend ➖"
    
    n_days = len(daily_visitors)
    half = n_days // 2
    
    first_half_avg = daily_visitors.iloc[:half].mean()
    second_half_avg = daily_visitors.iloc[half:].mean()
    
    if pd.isna(first_half_avg) or pd.isna(second_half_avg) or first_half_avg == 0:
        trend_html = "<span style='color: #ffd54f;'>Zu wenig Daten ➖</span>"
    elif second_half_avg > first_half_avg * 1.05:
        trend_html = "<span style='color: #b388ff;'>Zunehmend 📈</span>"
    elif second_half_avg < first_half_avg * 0.95:
        trend_html = "<span style='color: #ff5252;'>Abnehmend 📉</span>"
    else:
        trend_html = "<span style='color: #ffd54f;'>Gleichbleibend ➖</span>"
        
    text = f"Insgesamt <strong>{total_visitors}</strong> absolute Besucher im ausgewählten Zeitraum ({len(daily_visitors)} Tage ausgewertet)."
    return text, trend_html

def analyze_best_observation_time(df):
    """
    Berechnet für jede Art die Tagesstunde mit den meisten historischen Sichtungen.
    Gibt eine HTML-Tabelle zurück, sortiert nach Anzahl der Gesamtsichtungen (häufigste Art zuerst).
    """
    SKIP_SPECIES = {'Unbekannt', 'IGNORED_LOW_CONFIDENCE', 'Hintergrund'}

    if df.empty:
        return "<p>Keine Daten vorhanden.</p>"

    df_filtered = df[~df['species'].isin(SKIP_SPECIES)].copy()
    if df_filtered.empty:
        return "<p>Keine auswertbaren Arten vorhanden.</p>"

    # Sichtungen pro Art und Stunde zählen
    grouped = df_filtered.groupby(['species', 'hour']).size().reset_index(name='count')

    # Gesamt-Sichtungen pro Art (für Sortierung)
    total_per_species = df_filtered.groupby('species').size().rename('total')

    rows = []
    for species, species_df in grouped.groupby('species'):
        species_df = species_df.sort_values('count', ascending=False)
        best_hour   = int(species_df.iloc[0]['hour'])
        best_count  = int(species_df.iloc[0]['count'])
        total       = int(total_per_species.get(species, best_count))

        # Zweite Peakstunde (wenn vorhanden und mind. 60% des Hauptpeaks)
        second_peak = ""
        if len(species_df) > 1:
            second_count = int(species_df.iloc[1]['count'])
            if second_count >= best_count * 0.6:
                second_hour = int(species_df.iloc[1]['hour'])
                second_peak = f" &amp; {second_hour:02d}:00"

        # Balken-Breite als visueller Indikator (relativ zum Maximum)
        bar_pct = min(100, int((best_count / max(total, 1)) * 100 * 3))
        bar_pct = min(bar_pct, 100)

        rows.append({
            'species': species,
            'best_hour': best_hour,
            'second_peak': second_peak,
            'best_count': best_count,
            'total': total,
            'bar_pct': bar_pct,
        })

    # Sortierung: häufigste Art zuerst
    rows.sort(key=lambda r: r['total'], reverse=True)

    html = (
        "<table style='width:100%; border-collapse:collapse; font-size:0.9em;'>"
        "<thead><tr>"
        "<th style='text-align:left; padding:5px 8px; color:#aaa; font-weight:600; border-bottom:1px solid #333;'>Art</th>"
        "<th style='text-align:center; padding:5px 8px; color:#aaa; font-weight:600; border-bottom:1px solid #333;'>Beste Zeit</th>"
        "<th style='text-align:left; padding:5px 8px; color:#aaa; font-weight:600; border-bottom:1px solid #333; min-width:80px;'>Konzentration</th>"
        "<th style='text-align:right; padding:5px 8px; color:#aaa; font-weight:600; border-bottom:1px solid #333;'>Ges.</th>"
        "</tr></thead><tbody>"
    )
    for r in rows:
        bar_html = (
            f"<div style='background:#37474f; border-radius:3px; height:8px;'>"
            f"<div style='background:#ffca28; width:{r['bar_pct']}%; height:8px; border-radius:3px;'></div>"
            f"</div>"
        )
        html += (
            f"<tr>"
            f"<td style='padding:5px 8px; border-bottom:1px solid #2a2a2a;'>{r['species']}</td>"
            f"<td style='padding:5px 8px; border-bottom:1px solid #2a2a2a; text-align:center; "
            f"color:#fff176; font-weight:bold;'>{r['best_hour']:02d}:00{r['second_peak']}</td>"
            f"<td style='padding:5px 8px; border-bottom:1px solid #2a2a2a;'>{bar_html}</td>"
            f"<td style='padding:5px 8px; border-bottom:1px solid #2a2a2a; text-align:right; "
            f"color:#aaa;'>{r['total']}</td>"
            f"</tr>"
        )
    html += "</tbody></table>"
    return html


def predict_weekly_visitors(df):
    """
    Berechnet auf Basis des historischen DataFrames (df) eine Vorhersage
    für die aktuelle laufende Kalenderwoche.
    Gibt zurück:
      - chart_url (base64-PNG)
      - summary_html (kurzer Text mit Ist/Prognose/Gesamt)
      - kw_label (z.B. "KW 11  |  10.03. – 16.03.2026")
    """
    WEEKDAY_NAMES = ['Mo', 'Di', 'Mi', 'Do', 'Fr', 'Sa', 'So']

    today = datetime.datetime.now().date()
    # Montag und Sonntag der aktuellen Woche
    monday = today - datetime.timedelta(days=today.weekday())
    sunday = monday + datetime.timedelta(days=6)
    kw = today.isocalendar()[1]
    kw_label = (f"KW {kw}  |  {monday.strftime('%d.%m.')} – {sunday.strftime('%d.%m.%Y')}")

    if df.empty:
        return None, "Nicht genug Daten für eine Wochenvorhersage.", kw_label

    # --- 1. Historische Tagesdurchschnitte pro Wochentag (0=Mo … 6=So) ---
    df_hist = df.copy()
    df_hist['weekday'] = df_hist['datetime'].dt.weekday   # 0=Mo
    df_hist['date']    = df_hist['datetime'].dt.date

    # Nur Tage, die NICHT in der aktuellen Woche liegen, als Trainingsbasis nehmen
    df_hist = df_hist[df_hist['date'] < monday]

    if df_hist.empty:
        return None, "Noch keine Historik außerhalb der aktuellen Woche vorhanden.", kw_label

    daily_counts_hist = df_hist.groupby(['date', 'weekday']).size().reset_index(name='count')
    avg_by_weekday = daily_counts_hist.groupby('weekday')['count'].mean()

    # --- 2. Ist-Werte der aktuellen Woche (nur abgeschlossene Tage vor heute) ---
    df_this_week = df[df['date'].apply(lambda d: monday <= d < today)]
    actual_by_weekday = df_this_week.groupby(df_this_week['datetime'].dt.weekday).size()

    # --- 3. Daten für das Diagramm zusammenstellen ---
    actuals   = []   # Ist-Werte (0 wenn noch kein Tag)
    forecasts = []   # Prognose-Werte (0 wenn bereits Ist-Wert)
    for wd in range(7):
        day = monday + datetime.timedelta(days=wd)
        if day < today:   # nur abgeschlossene Tage als Ist
            actuals.append(int(actual_by_weekday.get(wd, 0)))
            forecasts.append(0)
        else:             # heute und Zukunft als Prognose
            actuals.append(0)
            forecasts.append(round(avg_by_weekday.get(wd, 0), 1))

    # --- 4. Kennzahlen ---
    total_actual   = sum(actuals)
    total_forecast = sum(forecasts)
    total_combined = total_actual + total_forecast

    # --- 5. Diagramm zeichnen ---
    fig, ax = plt.subplots(figsize=(8, 4), facecolor='#1e1e1e')
    ax.set_facecolor('#1e1e1e')

    x = range(7)
    bar_width = 0.6

    # Ist-Balken (solide blau)
    bars_actual = ax.bar(
        [i for i in x if actuals[i] > 0],
        [actuals[i] for i in x if actuals[i] > 0],
        width=bar_width, color='#4fc3f7', label='Ist', zorder=3
    )
    # Prognose-Balken (violett, schraffiert / transparent)
    bars_forecast = ax.bar(
        [i for i in x if forecasts[i] > 0],
        [forecasts[i] for i in x if forecasts[i] > 0],
        width=bar_width, color='#ce93d8', alpha=0.55, label='Prognose',
        linestyle='--', edgecolor='#ba68c8', linewidth=1.5, zorder=3
    )

    # Heute markieren
    today_wd = today.weekday()
    ax.axvline(today_wd + 0.5, color='#ff9800', linestyle=':', linewidth=1.2, alpha=0.8)

    # Wertbeschriftungen
    for i, (a, f) in enumerate(zip(actuals, forecasts)):
        val = a if a > 0 else f
        if val > 0:
            ax.text(i, val + max(total_combined / 70, 1), f"{int(val)}",
                    ha='center', va='bottom', color='white', fontsize=8.5)

    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [f"{WEEKDAY_NAMES[i]}\n{(monday + datetime.timedelta(days=i)).strftime('%d.%m.')}" for i in x],
        color='white', fontsize=8
    )
    ax.tick_params(axis='y', colors='white')
    ax.set_ylabel('Besucher', color='white', fontsize=9)
    ax.set_title('Besucher-Prognose für die aktuelle Woche', color='white', fontsize=11)
    ax.legend(facecolor='#2e2e2e', labelcolor='white', fontsize=8)
    ax.spines[['top', 'right', 'left', 'bottom']].set_color('#444')
    ax.yaxis.grid(True, color='#333', linestyle='--', linewidth=0.6)
    ax.set_axisbelow(True)

    plt.tight_layout()
    img = io.BytesIO()
    fig.savefig(img, format='png', facecolor='#1e1e1e')
    img.seek(0)
    chart_url = base64.b64encode(img.getvalue()).decode()
    plt.close(fig)

    # --- 6. Zusammenfassungstext ---
    remaining_days = 7 - (today.weekday() + 1)
    summary_html = (
        f"<strong>Bisher diese Woche:</strong> {total_actual} Besucher &nbsp;|&nbsp; "
        f"<strong>Prognose Restwoche</strong> ({remaining_days} Tage): "
        f"<span style='color:#ce93d8'>+{int(total_forecast)}</span> &nbsp;|&nbsp; "
        f"<strong>Gesamtprognose:</strong> <span style='color:#fff176'>~{int(total_combined)}</span>"
    )

    return chart_url, summary_html, kw_label


def fetch_weather_data(config):
    if not config or "API-Key" not in config or "Station-ID" not in config:
        return None, "Wetter-Konfiguration unvollständig."
    api_key = config["API-Key"]
    station_id = config["Station-ID"]
    url = f"https://api.weather.com/v2/pws/observations/current?stationId={station_id}&format=json&units=m&apiKey={api_key}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            obs = data.get("observations", [{}])[0]
            metric = obs.get("metric", {})
            temp = metric.get("temp", "N/A")
            precip = metric.get("precipRate", 0.0)
            impact = "Normaler erwarteter Andrang."
            if isinstance(temp, (int, float)) and temp < 5.0:
                impact = f"Es ist kalt ({temp}°C). Rechne mit 30% mehr Futterhaus-Besuchen."
            elif isinstance(precip, (int, float)) and precip > 0:
                impact = f"Es regnet ({precip} mm/h). Vögel suchen verstärkt Schutz und Nahrung."
            elif isinstance(temp, (int, float)) and temp > 25.0:
                impact = f"Es ist warm ({temp}°C). Aktivität verlagert sich vermutlich in die frühen Morgenstunden."
            return data, impact
        else:
            return None, f"API Fehler {response.status_code}: Bitte überprüfe den API-Key für Station {station_id}."
    except Exception as e:
        return None, f"Verbindungsfehler zur Wetter-API: {e}"

@app.route('/prediction')
def prediction_dashboard():
    try:
        days = int(request.args.get('days', 14))
    except ValueError:
        days = 14
    if days < 1: days = 1
    if days > 30: days = 30
    
    df = get_prediction_db_data(days)
    # Langer Referenz-DF (immer 30 Tage) für stabile Anomalie-Durchschnitte –
    # unabhängig vom gewählten Anzeigebereich des Nutzers.
    df_reference = get_prediction_db_data(30)
    weather_config = load_weather_config()
    
    busiest_hour, rush_hour_chart = predict_rush_hour(df)
    
    next_hour = (datetime.datetime.now().hour + 1) % 24
    probs, prob_chart = predict_species_probability(df, target_hour=next_hour)
    
    weather_text = "Keine Wetterstation konfiguriert."
    weather_data_html = ""
    if weather_config:
        w_data, w_impact = fetch_weather_data(weather_config)
        weather_text = f"<strong>Vorhersage:</strong> {w_impact}"
        if w_data:
            metric = w_data.get("observations", [{}])[0].get("metric", {})
            t = metric.get("temp", "N/A")
            p = metric.get("precipRate", "0.0")
            h = w_data.get("observations", [{}])[0].get("humidity", "N/A")
            weather_data_html = f"<div style='margin-bottom: 10px; font-size: 0.9em; color: #aaa;'>Aktuell: {t}°C | Niederschlag: {p} mm/h | Luftfeuchtigkeit: {h}%</div>"
        else:
            weather_data_html = f"<div style='margin-bottom: 10px; font-size: 0.9em; color: #ff9800;'>Hinweis: {w_impact} <br>(Für echte Daten gültigen API-Key in weather_config.json eintragen)</div>"
            weather_text = "<strong>Muster (Demo):</strong> An regnerischen oder sehr kalten Tagen weicht das Futterverhalten stark von sonnigen Tagen ab. Vögel fressen dann oft mehr und in konzentrierteren Abständen."
    
    disturbance_text = analyze_disturbance(df)
    diversification_text, diversification_trend = analyze_diversification(df)
    absolute_visitors_text, absolute_visitors_trend = analyze_absolute_visitors(df)
    anomaly_text = analyze_anomalies(df, df_reference=df_reference)
    weekly_chart, weekly_summary, weekly_kw_label = predict_weekly_visitors(df)
    best_observation_time_table = analyze_best_observation_time(df)
    
    return render_template('prediction.html', 
                                  days=days, 
                                  busiest_hour=busiest_hour,
                                  rush_hour_chart=rush_hour_chart,
                                  next_hour=next_hour,
                                  probs=probs,
                                  prob_chart=prob_chart,
                                  weather_config=weather_config,
                                  weather_data_html=weather_data_html,
                                  weather_text=weather_text,
                                  disturbance_text=disturbance_text,
                                  diversification_text=diversification_text,
                                  diversification_trend=diversification_trend,
                                  absolute_visitors_text=absolute_visitors_text,
                                  absolute_visitors_trend=absolute_visitors_trend,
                                  anomaly_text=anomaly_text,
                                  weekly_chart=weekly_chart,
                                  weekly_summary=weekly_summary,
                                  weekly_kw_label=weekly_kw_label,
                                  best_observation_time_table=best_observation_time_table,
                                  version=APP_VERSION)

@app.route('/')
def dashboard():
    current_settings = load_settings()
    current_threshold = current_settings.get("threshold", 70)
    
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    
    conn = sqlite3.connect(DB_FILE, timeout=10)
    last_entry = None
    visitors_per_hour = 0
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
        cursor.execute("SELECT species, filename, timestamp, confidence FROM detections WHERE species != 'IGNORED_LOW_CONFIDENCE' ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        if row:
            last_entry = { 'species': row[0], 'filename': row[1], 'timestamp': row[2], 'confidence': int(row[3] * 100) }

        if not df.empty:
            df['species'] = df['species'].replace('IGNORED_LOW_CONFIDENCE', 'Unbekannt')
            df['count'] = pd.to_numeric(df['count'])
            df['today_count'] = pd.to_numeric(df['today_count']).fillna(0).astype(int)
            df = df.sort_values(by='count', ascending=False)
            
        # Check if latest image matches latest db entry
        images_match = False
        if last_entry:
            try:
                tracker_path = os.path.join(STATIC_FOLDER, 'last_detection_filename.txt')
                if os.path.exists(tracker_path):
                    with open(tracker_path, 'r', encoding='utf-8') as f:
                        tracked = f.read().strip()
                    if tracked == last_entry['filename']:
                        images_match = True
            except: pass
            
        # NEU: Neue Arten von heute finden (Erstsichtung)
        cursor.execute(f"SELECT species FROM detections GROUP BY species HAVING MIN(timestamp) LIKE '{today_str}%'")
        new_species_raw = [r[0] for r in cursor.fetchall()]
        new_species_today = [sp for sp in new_species_raw if sp not in ('Unbekannt', 'IGNORED_LOW_CONFIDENCE')]
            
        now = datetime.datetime.now()
        now_str = now.strftime('%Y-%m-%d %H:%M:%S')
        one_hour_ago_str = (now - datetime.timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute(f"SELECT COUNT(*) FROM detections WHERE timestamp >= '{one_hour_ago_str}' AND timestamp <= '{now_str}'")
        row_vph = cursor.fetchone()
        if row_vph:
            visitors_per_hour = row_vph[0]
            
    except: 
        df = pd.DataFrame()
        new_species_today = []
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
        
    futterplatz_occupy = None
    futterplatz_time_locks = {}
    try:
        if os.path.exists("futterplatz_status.json"):
            with open("futterplatz_status.json", "r", encoding="utf-8") as f:
                fstat = json.load(f)
                futterplatz_occupy = fstat.get("occupy", None)
                futterplatz_time_locks = fstat.get("time_locks", {})
    except:
        pass
        
    minutes_passed = 0
    if last_entry and 'timestamp' in last_entry:
        try:
            last_time = datetime.datetime.strptime(last_entry['timestamp'], '%Y-%m-%d %H:%M:%S')
            minutes_passed = int((datetime.datetime.now() - last_time).total_seconds() / 60)
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

    # Chart Generation (IMMER BALKEN) - HEUTE & GESAMT
    chart_url_heute = ""
    chart_url_gesamt = ""
    if not df.empty:
        # 1. Chart Gesamt
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1e1e1e')
        cmap = plt.get_cmap('tab20')
        colors = [('#555555' if sp == 'Unbekannt' else cmap(i % 20)) for i, sp in enumerate(df['species'])]
        ax.bar(df['species'], df['count'], color=colors)
        ax.tick_params(axis='x', colors='white', rotation=45)
        ax.tick_params(axis='y', colors='white')
        ax.set_facecolor('#1e1e1e')
        
        plt.tight_layout()
        img_gesamt = io.BytesIO()
        fig.savefig(img_gesamt, format='png', facecolor='#1e1e1e')
        img_gesamt.seek(0)
        chart_url_gesamt = base64.b64encode(img_gesamt.getvalue()).decode()
        plt.close(fig)

        # 2. Chart Heute (sortiert)
        df_heute = df[df['today_count'] > 0].sort_values(by='today_count', ascending=False)
        fig2, ax2 = plt.subplots(figsize=(10, 6), facecolor='#1e1e1e')
        
        if not df_heute.empty:
            color_dict = {sp: colors[i] for i, sp in enumerate(df['species'])}
            colors_heute = [color_dict[sp] for sp in df_heute['species']]
            ax2.bar(df_heute['species'], df_heute['today_count'], color=colors_heute)
            ax2.tick_params(axis='x', colors='white', rotation=45)
        else:
            ax2.bar(['Keine Daten'], [0], color=['#555555'])
            ax2.tick_params(axis='x', colors='white')
            
        ax2.tick_params(axis='y', colors='white')
        ax2.set_facecolor('#1e1e1e')
        plt.tight_layout()
        img_heute = io.BytesIO()
        fig2.savefig(img_heute, format='png', facecolor='#1e1e1e')
        img_heute.seek(0)
        chart_url_heute = base64.b64encode(img_heute.getvalue()).decode()
        plt.close(fig2)

    timestamp_now = int(time.time())
    
    current_temp = None
    try:
        global last_weather_check, last_weather_temp
        
        if time.time() - last_weather_check > 300: # 5 Minuten Cache
            w_config = load_weather_config()
            if w_config:
                w_data, _ = fetch_weather_data(w_config)
                if w_data:
                    metric = w_data.get("observations", [{}])[0].get("metric", {})
                    last_weather_temp = metric.get("temp", None)
                else:
                    last_weather_temp = None
            last_weather_check = time.time()
            
        current_temp = last_weather_temp
    except Exception:
        current_temp = None

    records = load_records()
    record_visitors_today = False
    record_visitors_per_hour = False
    updated_records = False
    
    max_today = records.get("today_total", 0)
    if today_total > 0 and today_total >= max_today:
        record_visitors_today = True
        if today_total > max_today:
            records["today_total"] = int(today_total)
            updated_records = True
            
    max_vph = records.get("visitors_per_hour", 0)
    if visitors_per_hour > 0 and visitors_per_hour >= max_vph:
        record_visitors_per_hour = True
        if visitors_per_hour > max_vph:
            records["visitors_per_hour"] = int(visitors_per_hour)
            updated_records = True
            
    if updated_records:
        save_records(records)

    return render_template('index.html',
                                  chart_url_heute=chart_url_heute,
                                  chart_url_gesamt=chart_url_gesamt, 
                                  df=df, 
                                  icon_map=icon_map, 
                                  total_count=total_count,
                                  today_total=today_total,
                                  last_entry=last_entry,
                                  images_match=images_match,
                                  ts=timestamp_now,
                                  version=APP_VERSION,
                                  ping_active=ping_active,
                                  camera_online=camera_online,
                                  futterplatz_occupy=futterplatz_occupy,
                                  futterplatz_time_locks=futterplatz_time_locks,
                                  new_species_today=new_species_today,
                                  current_temp=current_temp,
                                  visitors_per_hour=visitors_per_hour,
                                  minutes_passed=minutes_passed,
                                  record_visitors_today=record_visitors_today,
                                  record_visitors_per_hour=record_visitors_per_hour,
                                  unique_species_count=len(df))

@app.route('/weekly')
def weekly_stats():
    # --- SQL Aggregation (Speicheroptimierung) ---
    query = """
    SELECT 
        CASE WHEN species = 'IGNORED_LOW_CONFIDENCE' THEN 'Unbekannt' ELSE species END as species,
        strftime('%Y-', timestamp) || printf('%02d', CAST(strftime('%W', timestamp) AS INTEGER) + 1) as week_sort,
        printf('%02d', CAST(strftime('%W', timestamp) AS INTEGER) + 1) || '<br><small style=''color:#aaa''>''' || substr(strftime('%Y', timestamp), 3, 2) || '</small>' as week_display,
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

    return render_template('weekly.html', table_content=html_table, version=APP_VERSION)

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
        
        first_bird_row = df.sort_values(by='datetime').iloc[0]
        first_bird = first_bird_row['species']
        first_bird_time = first_bird_row['datetime'].strftime('%H:%M:%S')

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

    else:
        first_bird = None
        first_bird_time = None

    return render_template('daily.html', 
                                  chart_url=chart_url, 
                                  version=APP_VERSION,
                                  selected_date_str=selected_date_str,
                                  today_str=today_str,
                                  prev_date=prev_date,
                                  next_date=next_date,
                                  is_today=is_today,
                                  total_birds_day=total_birds_day,
                                  first_bird=first_bird,
                                  first_bird_time=first_bird_time)

def run_flask():
    print(f"Starte Waitress Server auf Port {FLASK_PORT}...")
    serve(app, host='0.0.0.0', port=FLASK_PORT, threads=4)


# --- NEU: Flask Request/Jsonify imports ---
from flask import jsonify, redirect
# 1. Controller definieren
class BirdAppController:
    def __init__(self):
        self.greylist = load_greylist()
        self.backlog = load_backlog()
        self.blacklist = load_blacklist()
        self.actionlist = load_actionlist()
        self.action_config = load_action_config()
        self.settings = load_settings()
        
        self.current_size_mb = 0.0
        self.current_remaining = 0
        self.duration_ms = 0
        self.logs = []
        
        self.monitor = FolderMonitor(
            update_log_callback=self.update_log, 
            get_threshold_callback=lambda: self.settings.get("threshold", 70),
            get_guess_threshold_callback=lambda: self.settings.get("guess_threshold", 40),
            get_margin_threshold_callback=lambda: self.settings.get("margin_threshold", 10),
            update_size_callback=self.update_size_display,
            get_delete_callback=lambda: self.settings.get("delete_active", True),
            get_greylist_active_callback=lambda: self.settings.get("greylist_active", True), 
            get_greylist_callback=lambda: self.greylist,
            get_backlog_active_callback=lambda: self.settings.get("backlog_active", True),
            get_backlog_callback=lambda: self.backlog,
            get_blacklist_callback=lambda: self.blacklist,
            update_duration_callback=self.update_duration_display,
            update_remaining_callback=self.update_remaining_display,
            get_algo_active_callback=lambda: self.settings.get("count_algo_active", True),
            get_webhook_active_callback=lambda: self.settings.get("webhook_active", True),
            get_actionlist_callback=lambda: getattr(self, 'actionlist', set()),
            get_action_config_callback=lambda: getattr(self, 'action_config', {})
        )
        self.schedule_ping()

    def update_settings(self):
        self.settings = load_settings()

    def schedule_ping(self):
        if self.settings.get("ping_active", True):
            ip = self.settings.get("camera_ip", "")
            if ip:
                threading.Thread(target=self._do_ping_task, args=(ip,), daemon=True).start()
            else:
                self.update_log("Ping aktiv, aber keine 'camera_ip' in settings.json.")
                self._write_ping_status(True, False)
        else:
            self._write_ping_status(False, False)
            
        threading.Timer(300.0, self.schedule_ping).start()

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

    def update_log(self, message):
        if len(self.logs) > 100:
            self.logs.pop(0)
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        self.logs.append(log_line)
        if self.settings.get('debug_active', True):
            try:
                ts_full = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(DEBUG_FILE, "a", encoding="utf-8") as f:
                    f.write(f"[{ts_full}] {message}\n")
            except Exception as e:
                print(f"Fehler beim Schreiben in die Debug-Datei: {e}")

    def update_size_display(self, size_mb):
        self.current_size_mb = size_mb

    def update_remaining_display(self, remaining):
        self.current_remaining = remaining

    def update_duration_display(self, duration_ms):
        self.duration_ms = duration_ms

    def start_monitoring(self):
        path = self.settings.get("last_folder", "")
        if not path:
            self.update_log("Fehler: Bitte zuerst einen Ordner in den Einstellungen festlegen.")
            return False
            
        recursive = self.settings.get("recursive", True)
        if not getattr(self.monitor, 'running', False):
            self.monitor.start(path, recursive)
            infos = ["Rename"]
            if self.settings.get("delete_active", True): infos.append(f"Blacklist ({len(self.blacklist)} Arten)")
            if self.settings.get("backlog_active", True): infos.append(f"Backlog ({len(self.backlog)} Arten)")
            if self.settings.get("greylist_active", True): infos.append(f"Greylist ({len(self.greylist)} Arten)")
            info_str = ", ".join(infos) if infos else "Nur Erkennung"
            self.update_log(f"Service gestartet: {info_str}")
            # Bildbearbeitungsmodus im Log anzeigen
            mode_labels = {
                "blur": "Blur-Padding",
                "resize": "Hartes Resize (600×600)",
                "edge": "Replicate / Edge-Padding",
                "crop": "Center-Crop"
            }
            mode_display = mode_labels.get(PADDING_MODE, PADDING_MODE)
            self.update_log(f"Modus Bildbearbeitung: {mode_display}")
            return True
        return False
        
    def stop_monitoring(self):
        if getattr(self.monitor, 'running', False):
            self.monitor.stop()
            self.update_log("Service gestoppt.")
            return True
        return False

    def sort_database_chronologically(self):
        self.update_log("Starte Datenbank-Sortierung...")
        try:
            conn = sqlite3.connect(DB_FILE, timeout=10)
            c = conn.cursor()
            c.execute('''CREATE TABLE detections_new (id INTEGER PRIMARY KEY, filename TEXT UNIQUE, species TEXT, timestamp TEXT, confidence REAL)''')
            c.execute('''INSERT INTO detections_new (filename, species, timestamp, confidence) SELECT filename, species, timestamp, confidence FROM detections ORDER BY timestamp ASC''')
            c.execute('DROP TABLE detections')
            c.execute('ALTER TABLE detections_new RENAME TO detections')
            c.execute('CREATE INDEX IF NOT EXISTS idx_species ON detections(species);')
            c.execute('CREATE INDEX IF NOT EXISTS idx_filename ON detections(filename);') 
            c.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON detections(timestamp);')
            conn.commit()
            c.execute("VACUUM")
            c.execute("SELECT COUNT(*) FROM detections")
            row_count = c.fetchone()[0]
            conn.close()
            self.update_log(f"Sortierung fertig: {row_count} Einträge chronologisch angeordnet.")
            return f"Sortierung Fertig: {row_count} Einträge chronologisch angeordnet."
        except Exception as e:
            msg = f"Sortierung Fehler: {e}"
            self.update_log(msg)
            return msg

    def reset_database(self):
        try:
            conn = sqlite3.connect(DB_FILE, timeout=10)
            c = conn.cursor()
            c.execute("DELETE FROM detections")
            conn.commit()
            c.execute("VACUUM")
            conn.close()
            self.update_log("Datenbank geleert.")
            return "Datenbank wurde geleert."
        except Exception as e: 
            return f"Fehler: {e}"

    def backup_database(self):
        try:
            backup_dir = "backup_db"
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)
            current_date = datetime.datetime.now().strftime("%d.%m.%Y")
            backup_filename = f"birds_stats_{current_date}.db"
            backup_path = os.path.join(backup_dir, backup_filename)
            shutil.copy2(DB_FILE, backup_path)
            self.update_log(f"Datenbank Backup erstellt: {backup_filename}")
            return f"Backup erfolgreich gespeichert: {backup_filename}"
        except Exception as e:
            msg = f"Fehler beim Backup: {e}"
            self.update_log(msg)
            return msg

app_controller = BirdAppController()

# --- WEB ENDPOINTS (SETTINGS & CONTROL) ---

SETTINGS_CSS = """
<style>
    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #121212; color: #e0e0e0; margin: 0; padding: 0; }
    .container { max-width: 900px; margin: 20px auto; background: #1e1e1e; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }
    h1, h2 { color: #4fc3f7; }
    h1 { margin-top: 0; padding-bottom: 10px; border-bottom: 2px solid #333;}
    .section { background: #263238; border-radius: 8px; padding: 20px; margin-bottom: 20px; border: 1px solid #37474f;}
    .row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 15px; }
    label { font-weight: bold; }
    input[type=text], input[type=number] { padding: 10px; border-radius: 6px; border: 1px solid #444; background: #121212; color: #fff; width: 60%; }
    input[type=checkbox] { transform: scale(1.5); margin-right: 10px; cursor: pointer; }
    button, .btn { padding: 10px 20px; border-radius: 6px; border: none; font-weight: bold; cursor: pointer; color: white; display: inline-block; text-decoration: none; text-align: center; }
    .btn-green { background: #2e7d32; } .btn-green:hover { background: #388e3c; }
    .btn-red { background: #c62828; } .btn-red:hover { background: #e53935; }
    .btn-blue { background: #0277bd; } .btn-blue:hover { background: #0288d1; }
    .btn-orange { background: #e65100; } .btn-orange:hover { background: #f57c00; }
    .btn-gray { background: #546e7a; } .btn-gray:hover { background: #607d8b; }
    .log-window { background: #000; border: 1px solid #444; color: #0f0; width: 100%; height: 200px; overflow-y: scroll; padding: 10px; font-family: monospace; border-radius: 6px; box-sizing: border-box; }
    .status-panel { display: flex; justify-content: space-between; background: #1e1e1e; padding: 15px; border-radius: 8px; border: 1px solid #333; margin-bottom: 20px; }
    .status-item { text-align: center; }
    .status-val { font-size: 1.5em; font-weight: bold; color: #81d4fa; }
    textarea.list-editor { width: 100%; height: 150px; background: #121212; color: #fff; border: 1px solid #444; padding: 10px; border-radius: 6px; margin-top: 10px; resize: vertical; }
    .btn-nav { margin-bottom: 20px; }
    .help-text { font-size: 0.8em; color: #999; margin-top: 5px; display: block; }
</style>
"""

@app.route('/settings')
def settings_page():
    s = app_controller.settings
    # Build list of unique species in DB for helpers
    known_species = set()
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT DISTINCT species FROM detections")
        for row in c.fetchall():
            if row[0] == "IGNORED_LOW_CONFIDENCE":
                known_species.add("Unbekannt")
            else:
                known_species.add(row[0])
        conn.close()
    except: pass
    db_species_str = ", ".join(sorted(list(known_species)))

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Einstellungen & Steuerung</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0"> 
        {{{{ css_style|safe }}}}
    </head>
    <body>
        <div class="container">
            <a href="/" class="btn btn-gray btn-nav">&laquo; Zurück zum Dashboard</a>
            <h1>⚙️ Einstellungen & Steuerung</h1>

            <div class="status-panel">
                <div class="status-item"><div>Ordnergröße</div><div class="status-val" id="size-val">0.0 MB</div></div>
                <div class="status-item"><div>Ausstehend</div><div class="status-val" id="rem-val">0</div></div>
                <div class="status-item"><div>Bild-Dauer</div><div class="status-val" id="dur-val">0 ms</div></div>
                <div class="status-item"><div>RAM</div><div class="status-val" id="ram-val">0 MB</div></div>
            </div>

            <div class="section">
                <h2>Steuerung & Datenbank</h2>
                <div style="display:flex; gap:10px; margin-bottom: 20px;">
                    <button class="btn btn-green" onclick="startMonitor()" id="btn-start">Überwachung Starten</button>
                    <button class="btn btn-red" onclick="stopMonitor()" id="btn-stop">Stoppen</button>
                    <button class="btn btn-orange" onclick="dbSync()">Datenbank zeitlich sortieren</button>
                    <button class="btn btn-red" disabled style="opacity: 0.5; cursor: not-allowed;" onclick="dbReset()">DB Reset (Alles löschen)</button>
                    <button class="btn btn-blue" onclick="dbBackup()">DB Backup in backup_db</button>
                    <a href="/manual_entry" class="btn btn-blue" style="text-decoration: none; display: flex; align-items: center; padding: 10px 15px;">➕ Manueller Eintrag</a>
                    <a href="/delete_entry" class="btn btn-blue" style="text-decoration: none; display: flex; align-items: center; padding: 10px 15px;">✏️ Falscherkennung bearbeiten/löschen</a>
                </div>
                <h3>Status-Log</h3>
                <div class="log-window" id="log-window"></div>
            </div>

            <form id="settingsForm">
                <div class="section">
                    <h2>KI Einstellungen</h2>
                    <div class="row">
                        <label>Bilder-Ordner (Absoluter Pfad):</label>
                        <input type="text" id="last_folder" value="{s.get('last_folder', '')}" required>
                    </div>
                    <div class="row" style="justify-content: flex-start;">
                        <input type="checkbox" id="recursive" {'checked' if s.get('recursive', True) else ''}>
                        <label>Unterordner durchsuchen (Rekursiv)</label>
                    </div>
                    
                    <div class="row">
                        <label>Kamera IP (für Ping):</label>
                        <input type="text" id="camera_ip" value="{s.get('camera_ip', '')}">
                    </div>
                    
                    <div class="row">
                        <label>Mindest-Wahrscheinlichkeit Konfidenz (Threshold %):</label>
                        <input type="number" id="threshold" value="{s.get('threshold', 70)}" min="0" max="100">
                    </div>
                    
                    <div class="row">
                        <label>Vorsprung der Konfidenz (Threshold %):</label>
                        <input type="number" id="margin_threshold" value="{s.get('margin_threshold', 10)}" min="0" max="100">
                    </div>
                    
                    <div class="row">
                        <label>Mindest-Wahrscheinlichkeit Vermutung (Guess Threshold %):</label>
                        <input type="number" id="guess_threshold" value="{s.get('guess_threshold', 40)}" min="0" max="100">
                    </div>
                    
                    <div style="margin-top: 20px;">
                        <div class="row" style="justify-content: flex-start;"><input type="checkbox" id="ping_active" {'checked' if s.get('ping_active', True) else ''}> <label style="color:darkgreen;">Camera: ping aktiv</label></div>
                        <div class="row" style="justify-content: flex-start;"><input type="checkbox" id="debug_active" {'checked' if s.get('debug_active', True) else ''}> <label style="color:orange;">Debug Log</label></div>
                        <div class="row" style="justify-content: flex-start;"><input type="checkbox" id="count_algo_active" {'checked' if s.get('count_algo_active', True) else ''}> <label style="color:purple;">Count Algorithm (Hectic/Normal/Lazy)</label></div>
                    </div>
                </div>

                <div class="section">
                    <h2>Aktions-Einstellungen (Webhook)</h2>
                    <div class="row">
                        <label>Start-Webhook URL (GET):</label>
                        <input type="text" id="action_start_webhook" value="{app_controller.action_config.get('start_webhook', '')}" placeholder="http://tasmota-ip/cm?cmnd=Power%20On" style="width:100%; max-width:400px;">
                    </div>
                    <div class="row">
                        <label>Stop-Webhook URL (GET):</label>
                        <input type="text" id="action_stop_webhook" value="{app_controller.action_config.get('stop_webhook', '')}" placeholder="http://tasmota-ip/cm?cmnd=Power%20Off" style="width:100%; max-width:400px;">
                    </div>
                    <div class="row">
                        <label>Laufzeit in Sekunden:</label>
                        <input type="number" id="action_duration" value="{app_controller.action_config.get('duration', 10)}" min="1" max="3600" style="max-width:100px;">
                    </div>
                    <div class="row">
                        <label>Max. Bildalter in Sekunden:</label>
                        <input type="number" id="action_max_age" value="{app_controller.action_config.get('max_age', 120)}" min="1" max="3600" style="max-width:100px;">
                    </div>
                    <div class="row" style="gap:10px; margin-top:10px;">
                        <button type="button" class="btn btn-orange" onclick="testWebhookStart()">Test Start-Webhook</button>
                        <button type="button" class="btn btn-red" onclick="testWebhookStop()">Test Stop-Webhook</button>
                    </div>
                </div>

                <div class="section">
                    <h2>Listen Konfiguration (Filter)</h2>
                    
                    <div class="row" style="justify-content: flex-start;">
                        <input type="checkbox" id="webhook_active" {'checked' if s.get('webhook_active', True) else ''}>
                        <label style="color:#009688; font-weight:bold;">Aktions-Trigger aktiv: Bei diesen Arten wird die oben konfigurierte Aktion ausgelöst</label>
                    </div>
                    <label>Aktions-Arten (kommagetrennt):</label>
                    <textarea class="list-editor" id="actionlist">{','.join(app_controller.actionlist)}</textarea>

                    <hr style="border-color: #444; margin:20px 0;">
                    
                    <div class="row" style="justify-content: flex-start;">
                        <input type="checkbox" id="backlog_active" {'checked' if s.get('backlog_active', True) else ''}>
                        <label style="color:#d2691e;">Backlog: Verschieben - KEIN Datenbankeintrag (für unbedeutende Arten / Algo-Ignores)</label>
                    </div>
                    <label>Backlog Arten (kommagetrennt):</label>
                    <textarea class="list-editor" id="backlog">{','.join(app_controller.backlog)}</textarea>

                    <hr style="border-color: #444; margin:20px 0;">
                    
                    <div class="row" style="justify-content: flex-start;">
                        <input type="checkbox" id="greylist_active" {'checked' if s.get('greylist_active', True) else ''}>
                        <label style="color:indianred;">Greylist: Löschen + Datenbankeintrag</label>
                    </div>
                    <label>Greylist Arten (kommagetrennt):</label>
                    <textarea class="list-editor" id="greylist">{','.join(app_controller.greylist)}</textarea>

                    <hr style="border-color: #444; margin:20px 0;">

                    <div class="row" style="justify-content: flex-start;">
                        <input type="checkbox" id="delete_active" {'checked' if s.get('delete_active', True) else ''}>
                        <label style="color:red;">Blacklist: Löschen - KEIN Datenbankeintrag (z.B. Hintergrund, Unbekannt)</label>
                    </div>
                    <label>Blacklist Arten (kommagetrennt):</label>
                    <textarea class="list-editor" id="blacklist">{','.join(app_controller.blacklist)}</textarea>

                    <span class="help-text">Verfügbare Arten in DB: {db_species_str}</span>
                </div>
                
                <button type="button" class="btn btn-blue" style="width:100%; padding: 15px; font-size:1.2em;" onclick="saveSettings()">Einstellungen Speichern</button>
            </form>
        </div>

        <script>
            // Initiale UI
            updateButtons();

            function startMonitor() {{
                fetch('/api/control/start', {{ method: 'POST' }})
                .then(r => r.json())
                .then(d => {{ if(d.error) alert(d.error); updateButtons(); }});
            }}
            
            function stopMonitor() {{
                fetch('/api/control/stop', {{ method: 'POST' }})
                .then(r => r.json())
                .then(d => updateButtons());
            }}
            
            function dbSync() {{
                if(confirm("Möchtest du wirklich die Datenbank zeitlich neu sortieren?")) {{
                    fetch('/api/control/dbsync', {{ method: 'POST' }}).then(r=>r.json()).then(d=>alert(d.msg));
                }}
            }}
            
            function dbReset() {{
                if(confirm("WARNUNG! Möchtest du wirklich ALLE Daten der Datenbank und Statistiken verwerfen?")) {{
                    if(confirm("Sicher? Letzte Warnung!")) {{
                        fetch('/api/control/dbreset', {{ method: 'POST' }}).then(r=>r.json()).then(d=>alert(d.msg));
                    }}
                }}
            }}

            function dbBackup() {{
                fetch('/api/control/dbbackup', {{ method: 'POST' }}).then(r=>r.json()).then(d=>alert(d.msg));
            }}

            function testWebhookStart() {{
                fetch('/api/control/test_webhook_start', {{ method: 'POST' }}).then(r => r.json()).then(d => {{
                    if(d.error) alert(d.error); else alert(d.msg);
                }});
            }}

            function testWebhookStop() {{
                fetch('/api/control/test_webhook_stop', {{ method: 'POST' }}).then(r => r.json()).then(d => {{
                    if(d.error) alert(d.error); else alert(d.msg);
                }});
            }}

            function saveSettings() {{
                const data = {{
                    settings: {{
                        last_folder: document.getElementById('last_folder').value,
                        recursive: document.getElementById('recursive').checked,
                        camera_ip: document.getElementById('camera_ip').value,
                        threshold: parseInt(document.getElementById('threshold').value),
                        margin_threshold: parseInt(document.getElementById('margin_threshold').value),
                        guess_threshold: parseInt(document.getElementById('guess_threshold').value),
                        ping_active: document.getElementById('ping_active').checked,
                        debug_active: document.getElementById('debug_active').checked,
                        count_algo_active: document.getElementById('count_algo_active').checked,
                        webhook_active: document.getElementById('webhook_active').checked,
                        backlog_active: document.getElementById('backlog_active').checked,
                        greylist_active: document.getElementById('greylist_active').checked,
                        delete_active: document.getElementById('delete_active').checked
                    }},
                    lists: {{
                        backlog: document.getElementById('backlog').value.split(',').map(s => s.trim()).filter(s => s),
                        greylist: document.getElementById('greylist').value.split(',').map(s => s.trim()).filter(s => s),
                        blacklist: document.getElementById('blacklist').value.split(',').map(s => s.trim()).filter(s => s),
                        actionlist: document.getElementById('actionlist').value.split(',').map(s => s.trim()).filter(s => s)
                    }},
                    action_config: {{
                        start_webhook: document.getElementById('action_start_webhook').value.trim(),
                        stop_webhook: document.getElementById('action_stop_webhook').value.trim(),
                        duration: parseInt(document.getElementById('action_duration').value) || 10,
                        max_age: parseInt(document.getElementById('action_max_age').value) || 120
                    }}
                }};
                
                fetch('/api/settings/save', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify(data)
                }})
                .then(res => res.json())
                .then(data => alert(data.msg))
                .catch(err => alert('Fehler beim Speichern!'));
            }}

            function updateButtons() {{
                fetch('/api/status')
                .then(res => res.json())
                .then(data => {{
                    if (data.running) {{
                        document.getElementById('btn-start').style.display = 'none';
                        document.getElementById('btn-stop').style.display = 'inline-block';
                    }} else {{
                        document.getElementById('btn-start').style.display = 'inline-block';
                        document.getElementById('btn-stop').style.display = 'none';
                    }}
                    
                    document.getElementById('size-val').innerText = data.size_mb.toFixed(2) + " MB";
                    document.getElementById('rem-val').innerText = data.remaining;
                    document.getElementById('dur-val').innerText = data.duration_ms + " ms";
                    document.getElementById('ram-val').innerText = data.ram_mb.toFixed(1) + " MB";
                    
                    const logWin = document.getElementById('log-window');
                    logWin.innerHTML = data.logs.join('<br>');
                    logWin.scrollTop = logWin.scrollHeight;
                }});
            }}

            // Poll Status Every 1.5 seconds
            setInterval(updateButtons, 1500);
        </script>
    </body>
    </html>
    """
    return render_template_string(html, css_style=SETTINGS_CSS)

@app.route('/api/control/start', methods=['POST'])
def api_start():
    ok = app_controller.start_monitoring()
    if ok: return jsonify({"msg": "Started"})
    return jsonify({"error": "Fehler beim Start. Überprüfe den Bilder-Ordner."})

@app.route('/api/control/stop', methods=['POST'])
def api_stop():
    app_controller.stop_monitoring()
    return jsonify({"msg": "Stopped"})

@app.route('/api/control/dbsync', methods=['POST'])
def api_dbsync():
    msg = app_controller.sort_database_chronologically()
    return jsonify({"msg": msg})

@app.route('/api/control/dbreset', methods=['POST'])
def api_dbreset():
    msg = app_controller.reset_database()
    return jsonify({"msg": msg})

@app.route('/api/control/dbbackup', methods=['POST'])
def api_dbbackup():
    msg = app_controller.backup_database()
    return jsonify({"msg": msg})

@app.route('/api/settings/save', methods=['POST'])
def api_settings_save():
    data = request.json
    new_settings = data.get('settings', {})
    
    # Save settings directly to file
    current_settings = load_settings()
    for k, v in new_settings.items():
        current_settings[k] = v
        save_setting(k, v)
        
    app_controller.update_settings()
    
    lists = data.get('lists', {})
    if 'backlog' in lists:
        app_controller.backlog = set(lists['backlog'])
        save_backlog(app_controller.backlog)
    if 'greylist' in lists:
        app_controller.greylist = set(lists['greylist'])
        save_greylist(app_controller.greylist)
    if 'blacklist' in lists:
        app_controller.blacklist = set(lists['blacklist'])
        save_blacklist(app_controller.blacklist)
    if 'actionlist' in lists:
        app_controller.actionlist = set(lists['actionlist'])
        save_actionlist(app_controller.actionlist)
        
    action_config = data.get('action_config')
    if action_config is not None:
        app_controller.action_config = action_config
        save_action_config(action_config)

    return jsonify({"msg": "Einstellungen und Listen erfolgreich gespeichert!"})

@app.route('/api/control/test_webhook_start', methods=['POST'])
def api_test_webhook_start():
    url = app_controller.action_config.get("start_webhook", "")
    if not url: return jsonify({"error": "Keine Start-URL konfiguriert."})
    try:
        requests.get(url, timeout=5)
        return jsonify({"msg": f"Start-Webhook an {url} gesendet."})
    except Exception as e:
        return jsonify({"error": f"Fehler beim Senden: {e}"})

@app.route('/api/control/test_webhook_stop', methods=['POST'])
def api_test_webhook_stop():
    url = app_controller.action_config.get("stop_webhook", "")
    if not url: return jsonify({"error": "Keine Stop-URL konfiguriert."})
    try:
        requests.get(url, timeout=5)
        return jsonify({"msg": f"Stop-Webhook an {url} gesendet."})
    except Exception as e:
        return jsonify({"error": f"Fehler beim Senden: {e}"})

@app.route('/api/status', methods=['GET'])
def api_status():
    import psutil
    try:
        process = psutil.Process(os.getpid())
        ram_mb = process.memory_info().rss / (1024 * 1024)
    except:
        ram_mb = 0.0
        
    return jsonify({
        "running": getattr(app_controller.monitor, 'running', False),
        "size_mb": app_controller.current_size_mb,
        "remaining": app_controller.current_remaining,
        "duration_ms": app_controller.duration_ms,
        "ram_mb": ram_mb,
        "logs": app_controller.logs
    })

@app.route('/wiki')
def wiki_page():
    wiki_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wiki')
    images = []
    if os.path.exists(wiki_folder):
        images = sorted([
            f for f in os.listdir(wiki_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp'))
        ])
    return render_template('wiki.html', version=APP_VERSION, wiki_images=images)

@app.route('/wiki/images/<path:filename>')
def wiki_image(filename):
    wiki_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wiki')
    return send_from_directory(wiki_folder, filename)

@app.route('/manual_entry')
def manual_entry_page():
    known_species = set()
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT DISTINCT species FROM detections")
        for row in c.fetchall():
            if row[0] == "IGNORED_LOW_CONFIDENCE":
                known_species.add("Unbekannt")
            else:
                known_species.add(row[0])
        conn.close()
    except: pass
    
    try:
        if os.path.exists("species_categories.json"):
            with open("species_categories.json", "r") as f:
                import json
                categories = json.load(f)
                for sp in categories.keys():
                    known_species.add(sp)
    except: pass
    
    species_list = sorted(list(known_species))
    return render_template('manual_entry.html', species_list=species_list)

@app.route('/api/manual_entry/save', methods=['POST'])
def api_manual_entry_save():
    import uuid
    data = request.json
    species = data.get('species')
    date_str = data.get('date')
    time_str = data.get('time')
    
    if not species or not date_str or not time_str:
        return jsonify({"success": False, "error": "Fehlende Daten"})
        
    timestamp = f"{date_str} {time_str}:00"
    dummy_filename = f"manual_{uuid.uuid4().hex[:8]}.jpg"
    
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        c = conn.cursor()
        c.execute("INSERT INTO detections (filename, species, timestamp, confidence) VALUES (?, ?, ?, ?)", 
                  (dummy_filename, species, timestamp, 1.0))
        conn.commit()
        conn.close()
        app_controller.update_log(f"Manueller Eintrag: {species} am {timestamp}")
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/delete_entry')
def delete_entry_page():
    known_species = set()
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT DISTINCT species FROM detections")
        for row in c.fetchall():
            if row[0] == "IGNORED_LOW_CONFIDENCE":
                known_species.add("Unbekannt")
            else:
                known_species.add(row[0])
        conn.close()
    except: pass
    
    try:
        if os.path.exists("species_categories.json"):
            with open("species_categories.json", "r") as f:
                import json
                categories = json.load(f)
                for sp in categories.keys():
                    known_species.add(sp)
    except: pass
    
    species_list = sorted(list(known_species))
    return render_template('delete_entry.html', species_list=species_list)

@app.route('/api/detections/by_date', methods=['GET'])
def api_detections_by_date():
    date_str = request.args.get('date')
    if not date_str:
        return jsonify({"success": False, "error": "Fehlendes Datum"}), 400
        
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        c = conn.cursor()
        search_pattern = f"{date_str}%"
        c.execute("SELECT id, timestamp, species, confidence, filename FROM detections WHERE timestamp LIKE ? ORDER BY timestamp DESC", (search_pattern,))
        rows = c.fetchall()
        
        entries = []
        for row in rows:
            time_part = row[1].split(' ')[1] if ' ' in row[1] else row[1]
            entries.append({
                "id": row[0],
                "timestamp": row[1],
                "time": time_part,
                "species": row[2],
                "confidence": round(row[3] * 100, 1) if row[3] else 0.0,
                "filename": row[4]
            })
            
        conn.close()
        return jsonify({"success": True, "entries": entries})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/detections/delete/<int:entry_id>', methods=['DELETE'])
def api_delete_detection(entry_id):
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        c = conn.cursor()
        
        # Get info before deleting for log
        c.execute("SELECT species, timestamp FROM detections WHERE id = ?", (entry_id,))
        row = c.fetchone()
        
        if not row:
            conn.close()
            return jsonify({"success": False, "error": "Eintrag nicht gefunden"}), 404
            
        species, timestamp = row
        
        c.execute("DELETE FROM detections WHERE id = ?", (entry_id,))
        conn.commit()
        conn.close()
        
        app_controller.update_log(f"Fehlerhafter DB-Eintrag gelöscht: {species} am {timestamp}")
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/detections/update/<int:entry_id>', methods=['POST'])
def api_update_detection(entry_id):
    data = request.json
    new_species = data.get('species')
    if not new_species:
         return jsonify({"success": False, "error": "Keine Art angegeben"}), 400
         
    try:
        conn = sqlite3.connect(DB_FILE, timeout=10)
        c = conn.cursor()
        
        # Get old info for log
        c.execute("SELECT species, timestamp FROM detections WHERE id = ?", (entry_id,))
        row = c.fetchone()
        
        if not row:
            conn.close()
            return jsonify({"success": False, "error": "Eintrag nicht gefunden"}), 404
            
        old_species, timestamp = row
        
        c.execute("UPDATE detections SET species = ? WHERE id = ?", (new_species, entry_id))
        conn.commit()
        conn.close()
        
        app_controller.update_log(f"Eintrag korrigiert: '{old_species}' -> '{new_species}' am {timestamp}")
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# Main entry point update
if __name__ == "__main__":
    init_db()
    # Check ob wir einen Autostart brauchen
    s = load_settings()
    last_folder = s.get("last_folder", "")
    if last_folder and os.path.exists(last_folder):
        print(f"Versuche Autostart für Überwachung auf: {last_folder}")
        def delayed_start():
            import time
            time.sleep(5)
            app_controller.start_monitoring()
        threading.Thread(target=delayed_start, daemon=True).start()
    
    run_flask()
