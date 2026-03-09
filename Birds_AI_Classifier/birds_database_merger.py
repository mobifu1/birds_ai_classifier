import sqlite3
import tkinter as tk
from tkinter import filedialog, messagebox

def merge_db_files():
    main_db = filedialog.askopenfilename(
        title="Wähle das Haupt-Datenbank File (Main file)", 
        filetypes=[("SQLite Database", "*.db"), ("All files", "*.*")]
    )
    if not main_db:
        return
        
    sub_db = filedialog.askopenfilename(
        title="Wähle das Sub-Datenbank File (Sub file, welches in Main integriert wird)", 
        filetypes=[("SQLite Database", "*.db"), ("All files", "*.*")]
    )
    if not sub_db:
        return
        
    try:
        # Verbindung zur Hauptdatenbank herstellen
        conn = sqlite3.connect(main_db)
        cursor = conn.cursor()
        # Zählen wie viele Einträge vorher da waren
        cursor.execute("SELECT COUNT(*) FROM detections")
        count_before = cursor.fetchone()[0]
        
        # Sub-Datenbank einhängen (ATTACH DATABASE)
        cursor.execute(f"ATTACH DATABASE '{sub_db}' AS sub_db")
        
        # 1. Temporäre Tabelle erstellen
        cursor.execute("CREATE TEMP TABLE temp_merge (filename TEXT UNIQUE, species TEXT, timestamp TEXT, confidence REAL)")
        
        # 2. Main-Datenbank Daten reinkopieren
        cursor.execute("INSERT OR IGNORE INTO temp_merge SELECT filename, species, timestamp, confidence FROM detections")
        
        # 3. Sub-Datenbank Daten reinkopieren (ignoriert Duplikate anhand vom Dateinamen)
        cursor.execute("INSERT OR IGNORE INTO temp_merge SELECT filename, species, timestamp, confidence FROM sub_db.detections")
        
        # 4. Alte Tabelle komplett ersetzen
        cursor.execute("DROP TABLE detections")
        cursor.execute('''
            CREATE TABLE detections 
            (id INTEGER PRIMARY KEY, filename TEXT UNIQUE, species TEXT, timestamp TEXT, confidence REAL)
        ''')
        
        # 5. Daten aus temporärer Tabelle wieder einfügen: diesmal SORTIERT nach Zeitstempel
        cursor.execute("""
            INSERT INTO detections (filename, species, timestamp, confidence)
            SELECT filename, species, timestamp, confidence FROM temp_merge ORDER BY timestamp ASC
        """)
        
        # 6. Indizes wieder anlegen (wie in deinem Original-Script)
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_species ON detections(species);')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_filename ON detections(filename);') 
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON detections(timestamp);')
        
        # Berechnen wie viele Einträge neu hinzugekommen sind
        cursor.execute("SELECT COUNT(*) FROM detections")
        count_after = cursor.fetchone()[0]
        changes = count_after - count_before
        
        conn.commit()
        
        # Verbindung zur Sub-DB wieder trennen
        cursor.execute("DETACH DATABASE sub_db")
        conn.close()
        
        messagebox.showinfo(
            "Ergebnis", 
            f"Merge abgeschlossen!\n\nEs wurden {changes} neue Bilder/Erkennungen in das Main-File hinzugefügt.\n(Duplikate anhand des Dateinamens wurden ignoriert)."
        )
        
    except sqlite3.Error as e:
        messagebox.showerror("Datenbankfehler", f"Ein Fehler ist mit der Datenbank aufgetreten:\n{e}")
    except Exception as e:
        messagebox.showerror("Allgemeiner Fehler", f"Ein unbekannter Fehler ist aufgetreten:\n{e}")

# --- GUI Setup ---
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Birds Stat DB Merger")
    root.geometry("400x200")
    root.resizable(False, False)
    
    # Styling
    root.configure(bg="#f0f0f0")
    
    # Header
    lbl_title = tk.Label(root, text="Vögel Datenbank Merger", font=("Arial", 16, "bold"), bg="#f0f0f0")
    lbl_title.pack(pady=15)
    
    desc_text = "Fügt Datensätze aus einer Sub-Datenbank in eine\nHaupt-Datenbank ein. Duplikate werden übersprungen."
    lbl_desc = tk.Label(root, text=desc_text, font=("Arial", 10), bg="#f0f0f0", justify="center")
    lbl_desc.pack(pady=5)
    
    # Button
    btn_merge = tk.Button(
        root, 
        text="Start Merge", 
        font=("Arial", 12, "bold"), 
        bg="#4CAF50", 
        fg="white", 
        command=merge_db_files,
        padx=10,
        pady=5
    )
    btn_merge.pack(pady=15)
    
    root.mainloop()
