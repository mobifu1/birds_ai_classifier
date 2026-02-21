import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import scrolledtext
import sqlite3

# --- Logik-Funktionen ---

def log(message):
    """Hilfsfunktion, um Nachrichten im Log-Fenster anzuzeigen."""
    log_window.config(state='normal')
    log_window.insert(tk.END, message + "\n")
    log_window.see(tk.END)
    log_window.config(state='disabled')

def select_database():
    """Öffnet einen Dialog zur Auswahl der Datenbankdatei."""
    filepath = filedialog.askopenfilename(
        title="Datenbank auswählen",
        filetypes=(("SQLite Datenbanken", "*.db"), ("Alle Dateien", "*.*"))
    )
    if filepath:
        db_path_var.set(filepath)
        log(f"[*] Datenbank ausgewählt: {filepath}")
        log("Tipp: Du kannst dir jetzt alle vorhandenen Keywords anzeigen lassen.")

def show_unique_keywords():
    """Liest alle einzigartigen Einträge aus der Spalte 'species' aus und zeigt sie im Log."""
    db_path = db_path_var.get().strip()
    
    if not db_path:
        messagebox.showwarning("Fehler", "Bitte wähle zuerst eine Datenbank aus.")
        return

    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # SELECT DISTINCT holt jeden Wert nur genau einmal aus der Datenbank.
        # ORDER BY sortiert die Liste alphabetisch, was das Lesen erleichtert.
        cursor.execute("SELECT DISTINCT species FROM detections WHERE species IS NOT NULL AND species != '' ORDER BY species")
        unique_species = cursor.fetchall()
        
        log("\n--- Liste aller vorhandenen Keywords (Species) ---")
        if not unique_species:
            log("[i] Die Datenbank enthält keine Einträge oder die Spalte ist leer.")
        else:
            # unique_species ist eine Liste von Tupeln, z.B. [('Amsel',), ('Hintergrund',)]. 
            # Wir wandeln das in eine schöne, durch Kommas getrennte Textzeile um.
            keywords_list = [item[0] for item in unique_species]
            log(", ".join(keywords_list))
        log("--------------------------------------------------\n")
            
    except sqlite3.Error as e:
        log(f"[ERROR] Datenbankfehler: {e}")
    finally:
        if conn:
            conn.close()

def execute_action():
    """Führt den Testlauf oder das tatsächliche Löschen aus."""
    db_path = db_path_var.get().strip()
    keyword = keyword_entry.get().strip()
    is_real_run = delete_checkbox_var.get()

    if not db_path:
        messagebox.showwarning("Fehler", "Bitte wähle zuerst eine Datenbank aus.")
        return
    if not keyword:
        messagebox.showwarning("Fehler", "Bitte gib ein Keyword ein.")
        return

    log(f"\n--- Starte Suche nach '{keyword}' ---")
    
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        search_pattern = f"%{keyword}%"
        
        # Zuerst herausfinden, WELCHE Einträge betroffen sind
        cursor.execute("SELECT id, filename, species FROM detections WHERE species LIKE ?", (search_pattern,))
        matches = cursor.fetchall()
        count = len(matches)
        
        if count == 0:
            log(f"[i] Keine Einträge mit dem Keyword '{keyword}' gefunden.")
        else:
            log(f"[i] Es wurden {count} betroffene Einträge gefunden:")
            for match in matches[:10]:
                log(f"    -> ID: {match[0]} | Datei: {match[1]} | Species: {match[2]}")
            
            if count > 10:
                log(f"    -> ... und {count - 10} weitere Einträge.")

            if not is_real_run:
                log("[!] TESTLAUF: Es wurden keine Daten gelöscht.")
                log("    Aktiviere die Checkbox 'Echtes Löschen durchführen', um diese Einträge zu entfernen.")
            else:
                # Echtes Löschen
                cursor.execute("DELETE FROM detections WHERE species LIKE ?", (search_pattern,))
                conn.commit()
                log(f"[SUCCESS] {count} Einträge wurden erfolgreich und restlos gelöscht.")
                delete_checkbox_var.set(False)

    except sqlite3.Error as e:
        log(f"[ERROR] Datenbankfehler: {e}")
    finally:
        if conn:
            conn.close()

# --- Aufbau der grafischen Benutzeroberfläche (GUI) ---

root = tk.Tk()
root.title("Datenbank Bereinigung - Pro Version")
root.geometry("650x650") # Fenster etwas vergrößert für die neuen Elemente

db_path_var = tk.StringVar()
delete_checkbox_var = tk.BooleanVar(value=False)

# --- Abschnitt 1: Datenbank Auswahl ---
frame_db = tk.LabelFrame(root, text="1. Datenbank auswählen", padx=10, pady=10)
frame_db.pack(padx=10, pady=10, fill="x")

db_entry = tk.Entry(frame_db, textvariable=db_path_var, state="readonly", width=50)
db_entry.pack(side="left", padx=5)

db_button = tk.Button(frame_db, text="Durchsuchen...", command=select_database)
db_button.pack(side="left", padx=5)

# NEU: Button, um die verfügbaren Keywords anzuzeigen
show_kw_button = tk.Button(frame_db, text="Vorhandene Keywords anzeigen", command=show_unique_keywords, bg="#e0e0e0")
show_kw_button.pack(side="left", padx=10)

# --- Abschnitt 2: Keyword und Optionen ---
frame_action = tk.LabelFrame(root, text="2. Suchkriterien & Aktion", padx=10, pady=10)
frame_action.pack(padx=10, pady=5, fill="x")

tk.Label(frame_action, text="Keyword (in 'species'):").grid(row=0, column=0, sticky="w", pady=5)
keyword_entry = tk.Entry(frame_action, width=30)
keyword_entry.grid(row=0, column=1, sticky="w", pady=5, padx=5)

delete_checkbox = tk.Checkbutton(
    frame_action, 
    text="Echtes Löschen durchführen (Wenn nicht markiert = Nur Testlauf)", 
    variable=delete_checkbox_var,
    fg="red"
)
delete_checkbox.grid(row=1, column=0, columnspan=2, sticky="w", pady=5)

execute_button = tk.Button(
    frame_action, 
    text="Aktion starten", 
    command=execute_action, 
    bg="blue", 
    fg="white", 
    font=("Arial", 10, "bold")
)
execute_button.grid(row=2, column=0, columnspan=2, pady=10)

# --- Abschnitt 3: Log Fenster ---
frame_log = tk.LabelFrame(root, text="3. System-Log", padx=10, pady=10)
frame_log.pack(padx=10, pady=10, fill="both", expand=True)

log_window = scrolledtext.ScrolledText(frame_log, wrap=tk.WORD, height=15, state='disabled', bg="#f4f4f4")
log_window.pack(fill="both", expand=True)

log("Willkommen! Bitte wähle eine Datenbank aus.")

root.mainloop()