# --- NEU: Flask Request/Jsonify imports ---
from flask import jsonify, redirect
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
                    <button class="btn btn-red" onclick="dbReset()">DB Reset (Alles löschen)</button>
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
                        <label>Mindest-Wahrscheinlichkeit (Threshold %):</label>
                        <input type="number" id="threshold" value="{s.get('threshold', 70)}" min="0" max="100">
                    </div>
                    
                    <div style="margin-top: 20px;">
                        <div class="row" style="justify-content: flex-start;"><input type="checkbox" id="rename_active" {'checked' if s.get('rename_active', True) else ''}> <label>Dateien umbenennen (Random + Class)</label></div>
                        <div class="row" style="justify-content: flex-start;"><input type="checkbox" id="ping_active" {'checked' if s.get('ping_active', True) else ''}> <label style="color:darkgreen;">Camera: ping aktiv</label></div>
                        <div class="row" style="justify-content: flex-start;"><input type="checkbox" id="count_algo_active" {'checked' if s.get('count_algo_active', True) else ''}> <label style="color:purple;">Count Algorithm (Hectic/Normal/Lazy)</label></div>
                    </div>
                </div>

                <div class="section">
                    <h2>Listen Konfiguration (Filter)</h2>
                    
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
                        <label style="color:red;">Blacklist (Trash): Löschen - KEIN Datenbankeintrag (z.B. Hintergrund, Unbekannt)</label>
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

            function saveSettings() {{
                const data = {{
                    settings: {{
                        last_folder: document.getElementById('last_folder').value,
                        recursive: document.getElementById('recursive').checked,
                        camera_ip: document.getElementById('camera_ip').value,
                        threshold: parseInt(document.getElementById('threshold').value),
                        rename_active: document.getElementById('rename_active').checked,
                        ping_active: document.getElementById('ping_active').checked,
                        count_algo_active: document.getElementById('count_algo_active').checked,
                        backlog_active: document.getElementById('backlog_active').checked,
                        greylist_active: document.getElementById('greylist_active').checked,
                        delete_active: document.getElementById('delete_active').checked
                    }},
                    lists: {{
                        backlog: document.getElementById('backlog').value.split(',').map(s => s.trim()).filter(s => s),
                        greylist: document.getElementById('greylist').value.split(',').map(s => s.trim()).filter(s => s),
                        blacklist: document.getElementById('blacklist').value.split(',').map(s => s.trim()).filter(s => s)
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
