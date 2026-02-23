import tkinter as tk
from tkinter import ttk, filedialog, messagebox
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
        cursor.execute("SELECT DISTINCT species FROM detections WHERE species IS NOT NULL AND species != '' ORDER BY species")
        unique_species = cursor.fetchall()
        
        log("\n--- Liste aller vorhandenen Keywords (Species) ---")
        if not unique_species:
            log("[i] Die Datenbank enthält keine Einträge oder die Spalte ist leer.")
        else:
            keywords_list = [item[0] for item in unique_species]
            log(", ".join(keywords_list))
        log("--------------------------------------------------\n")
            
    except sqlite3.Error as e:
        log(f"[ERROR] Datenbankfehler: {e}")
    finally:
        if conn:
            conn.close()

def search_entries():
    """Sucht nach Einträgen und zeigt sie in der Liste an."""
    db_path = db_path_var.get().strip()
    keyword = keyword_entry.get().strip()

    if not db_path:
        messagebox.showwarning("Fehler", "Bitte wähle zuerst eine Datenbank aus.")
        return
    if not keyword:
        messagebox.showwarning("Fehler", "Bitte gib ein Keyword ein.")
        return

    # Clear treeview
    for row in tree.get_children():
        tree.delete(row)

    log(f"\n--- Starte Suche nach '{keyword}' ---")
    
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Split keywords by comma
        keywords = [k.strip() for k in keyword.split(',') if k.strip()]
        
        all_matches = []
        for kw in keywords:
            search_pattern = f"%{kw}%"
            cursor.execute("SELECT id, filename, species FROM detections WHERE species LIKE ?", (search_pattern,))
            all_matches.extend(cursor.fetchall())
            
        # Remove duplicates if any
        all_matches = list(set(all_matches))
        
        count = len(all_matches)
        
        if count == 0:
            log(f"[i] Keine Einträge gefunden.")
        else:
            log(f"[i] Es wurden {count} betroffene Einträge gefunden. Wähle in der Liste aus, welche gelöscht werden sollen.")
            # Sort by ID
            for match in sorted(all_matches, key=lambda x: x[0]): 
                tree.insert("", tk.END, values=(match[0], match[1], match[2]))

    except sqlite3.Error as e:
        log(f"[ERROR] Datenbankfehler: {e}")
    finally:
        if conn:
            conn.close()

def execute_delete(items_to_delete):
    """Führt das tatsächliche Löschen der übergebenen Treeview-Items aus."""
    db_path = db_path_var.get().strip()
    
    if not delete_checkbox_var.get():
        log("[!] TESTLAUF: Es wurden keine Daten gelöscht.")
        log("    Aktiviere die Checkbox 'Echtes Löschen durchführen', um diese Einträge zu entfernen.")
        return

    if not messagebox.askyesno("Löschen bestätigen", f"Möchtest du wirklich {len(items_to_delete)} Einträge final löschen?"):
        return

    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        deleted_count = 0
        for item in items_to_delete:
            values = tree.item(item, 'values')
            db_id = values[0]
            cursor.execute("DELETE FROM detections WHERE id = ?", (db_id,))
            deleted_count += 1
            tree.delete(item)
            
        conn.commit()
        log(f"[SUCCESS] {deleted_count} Einträge wurden erfolgreich und restlos gelöscht.")
        delete_checkbox_var.set(False)

    except sqlite3.Error as e:
        log(f"[ERROR] Datenbankfehler: {e}")
    finally:
        if conn:
            conn.close()

def delete_selected():
    """Löscht die in der Liste ausgewählten Einträge."""
    selected_items = tree.selection()
    if not selected_items:
        messagebox.showwarning("Fehler", "Bitte wähle zuerst Einträge in der Liste aus! (LMT oder Strg / Shift + Klick)")
        return
    execute_delete(selected_items)

def delete_all_found():
    """Löscht alle aktuell in der Liste angezeigten Einträge."""
    all_items = tree.get_children()
    if not all_items:
        messagebox.showwarning("Fehler", "Die Liste ist leer.")
        return
    execute_delete(all_items)


# --- Aufbau der grafischen Benutzeroberfläche (GUI) ---

root = tk.Tk()
root.title("Datenbank Bereinigung - Pro Version")
root.geometry("800x750")

db_path_var = tk.StringVar()
delete_checkbox_var = tk.BooleanVar(value=False)

# --- Abschnitt 1: Datenbank Auswahl ---
frame_db = tk.LabelFrame(root, text="1. Datenbank auswählen", padx=10, pady=10)
frame_db.pack(padx=10, pady=5, fill="x")

db_entry = tk.Entry(frame_db, textvariable=db_path_var, state="readonly", width=50)
db_entry.pack(side="left", padx=5)

db_button = tk.Button(frame_db, text="Durchsuchen...", command=select_database)
db_button.pack(side="left", padx=5)

show_kw_button = tk.Button(frame_db, text="Vorhandene Keywords anzeigen", command=show_unique_keywords, bg="#e0e0e0")
show_kw_button.pack(side="left", padx=10)

# --- Abschnitt 2: Suche und Auswahl ---
frame_action = tk.LabelFrame(root, text="2. Suchkriterien & Ergebnisse", padx=10, pady=10)
frame_action.pack(padx=10, pady=5, fill="both", expand=True)

search_frame = tk.Frame(frame_action)
search_frame.pack(fill="x", pady=5)

tk.Label(search_frame, text="Keyword(s) (kommagetrennt für mehrere):").pack(side="left")
keyword_entry = tk.Entry(search_frame, width=30)
keyword_entry.pack(side="left", padx=5)

search_button = tk.Button(search_frame, text="Suchen", command=search_entries, bg="#e0e0e0")
search_button.pack(side="left", padx=5)

# Treeview für die Suchergebnisse
tree_frame = tk.Frame(frame_action)
tree_frame.pack(fill="both", expand=True, pady=5)

columns = ('id', 'filename', 'species')
tree = ttk.Treeview(tree_frame, columns=columns, show='headings', selectmode='extended')
tree.heading('id', text='ID')
tree.heading('filename', text='Dateiname')
tree.heading('species', text='Species')
tree.column('id', width=50, stretch=tk.NO)
tree.column('filename', width=450)
tree.column('species', width=150)

tree.pack(side="left", fill="both", expand=True)

scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=tree.yview)
tree.configure(yscroll=scrollbar.set)
scrollbar.pack(side="right", fill="y")

# Löschen-Optionen
delete_frame = tk.Frame(frame_action)
delete_frame.pack(fill="x", pady=10)

delete_checkbox = tk.Checkbutton(
    delete_frame, 
    text="Echtes Löschen durchführen", 
    variable=delete_checkbox_var,
    fg="red"
)
delete_checkbox.pack(side="left", padx=10)

btn_del_selected = tk.Button(delete_frame, text="Ausgewählte löschen", command=delete_selected, bg="orange", font=("Arial", 9, "bold"))
btn_del_selected.pack(side="left", padx=5)

btn_del_all = tk.Button(delete_frame, text="Alle Ergebnisse löschen", command=delete_all_found, bg="red", fg="white", font=("Arial", 9, "bold"))
btn_del_all.pack(side="left", padx=5)

# --- Abschnitt 3: Log Fenster ---
frame_log = tk.LabelFrame(root, text="3. System-Log", padx=10, pady=10)
frame_log.pack(padx=10, pady=5, fill="both")

log_window = scrolledtext.ScrolledText(frame_log, wrap=tk.WORD, height=10, state='disabled', bg="#f4f4f4")
log_window.pack(fill="both", expand=True)

log("Willkommen! Bitte wähle eine Datenbank aus.")

root.mainloop()