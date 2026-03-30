# --- NEU: Flask Request/Jsonify imports ---
from flask import Flask, render_template_string, request, url_for, jsonify, render_template, send_from_directory
# 1. Controller definieren
class BirdAppController:
    def __init__(self):
        self.greylist = load_greylist()
        self.backlog = load_backlog()
        self.blacklist = load_blacklist()
        self.settings = load_settings()
        
        self.current_size_mb = 0.0
        self.current_remaining = 0
        self.duration_ms = 0
        self.logs = []
        
        self.monitor = FolderMonitor(
            update_log_callback=self.update_log, 
            get_threshold_callback=lambda: self.settings.get("threshold", 70),
            get_guess_threshold_callback=lambda: self.settings.get("guess_threshold", 40),
            update_size_callback=self.update_size_display,
            get_rename_callback=lambda: self.settings.get("rename_active", True),
            get_delete_callback=lambda: self.settings.get("delete_active", True),
            get_greylist_active_callback=lambda: self.settings.get("greylist_active", True), 
            get_greylist_callback=lambda: self.greylist,
            get_backlog_active_callback=lambda: self.settings.get("backlog_active", True),
            get_backlog_callback=lambda: self.backlog,
            get_blacklist_callback=lambda: self.blacklist,
            update_duration_callback=self.update_duration_display,
            update_remaining_callback=self.update_remaining_display,
            get_algo_active_callback=lambda: self.settings.get("count_algo_active", True)
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
        if DEBUG:
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
            infos = []
            if self.settings.get("rename_active", True): infos.append("Rename")
            if self.settings.get("delete_active", True): infos.append(f"Blacklist ({len(self.blacklist)} Arten)")
            if self.settings.get("backlog_active", True): infos.append(f"Backlog ({len(self.backlog)} Arten)")
            if self.settings.get("greylist_active", True): infos.append(f"Greylist ({len(self.greylist)} Arten)")
            info_str = ", ".join(infos) if infos else "Nur Erkennung"
            self.update_log(f"Service gestartet: {info_str}")
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

app_controller = BirdAppController()

# --- WEB ENDPOINTS (SETTINGS & CONTROL) ---

SETTINGS_CSS = "" # Not used anymore

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

    return render_template('settings.html', s=s, db_species_str=db_species_str, app_controller=app_controller)

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

    return jsonify({"msg": "Einstellungen und Listen erfolgreich gespeichert!"})

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
