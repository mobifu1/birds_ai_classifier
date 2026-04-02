import sqlite3
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import random

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
        
    output_db = filedialog.asksaveasfilename(
        title="Speicherort für die NEUE zusammengeführte Datenbank",
        defaultextension=".db",
        initialfile="birds_stats_merged.db",
        filetypes=[("SQLite Database", "*.db"), ("All files", "*.*")]
    )
    if not output_db:
        return
        
    try:
        # Verbindung zur NEUEN Hauptdatenbank herstellen
        conn = sqlite3.connect(output_db)
        cursor = conn.cursor()
        
        # Beide Quell-Datenbanken an die neue DB anhängen
        cursor.execute(f"ATTACH DATABASE '{main_db}' AS main_db")
        cursor.execute(f"ATTACH DATABASE '{sub_db}' AS sub_db")
        
        # 1. Temporäre Tabelle in der NEUEN DB erstellen
        cursor.execute("CREATE TEMP TABLE temp_merge (filename TEXT UNIQUE, species TEXT, timestamp TEXT, confidence REAL)")
        
        
        # 2. Main-Datenbank Daten reinkopieren
        cursor.execute("INSERT OR IGNORE INTO temp_merge SELECT filename, species, timestamp, confidence FROM main_db.detections")
        
        # 3. Sub-Datenbank Daten reinkopieren (Kollisionen umbenennen)
        cursor.execute("SELECT filename, species, timestamp, confidence FROM sub_db.detections")
        for row in cursor.fetchall():
            filename, species, timestamp, confidence = row
            try:
                cursor.execute("INSERT INTO temp_merge (filename, species, timestamp, confidence) VALUES (?, ?, ?, ?)",
                               (filename, species, timestamp, confidence))
            except sqlite3.IntegrityError:
                # Check if it's an exact duplicate
                cursor.execute("SELECT species, timestamp FROM temp_merge WHERE filename = ?", (filename,))
                existing_row = cursor.fetchone()
                
                if existing_row and existing_row[0] == species and existing_row[1] == timestamp:
                    # It's a duplicate entry, skip inserting it again
                    continue
                
                # Collision: different entry with the same filename. Rename to preserve both.
                name_part, ext = os.path.splitext(filename)
                inserted = False
                while not inserted:
                    new_filename = f"{name_part}_merge_{random.randint(100000, 999999)}{ext}"
                    try:
                        cursor.execute("INSERT INTO temp_merge (filename, species, timestamp, confidence) VALUES (?, ?, ?, ?)",
                                       (new_filename, species, timestamp, confidence))
                        inserted = True
                    except sqlite3.IntegrityError:
                        pass
        
        # 4. In der neuen Tabelle speichern (detections)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections 
            (id INTEGER PRIMARY KEY, filename TEXT UNIQUE, species TEXT, timestamp TEXT, confidence REAL)
        ''')
        
        # 5. Daten aus temporärer Tabelle einfügen, sortiert nach Zeitstempel
        cursor.execute("""
            INSERT INTO detections (filename, species, timestamp, confidence)
            SELECT filename, species, timestamp, confidence FROM temp_merge ORDER BY timestamp ASC
        """)
        
        # 6. Indizes anlegen
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_species ON detections(species);')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_filename ON detections(filename);') 
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON detections(timestamp);')
        
        # Berechnen wie viele Einträge in der neuen DB sind
        cursor.execute("SELECT COUNT(*) FROM detections")
        count_after = cursor.fetchone()[0]
        
        conn.commit()
        
        # Verbindungen wieder trennen
        cursor.execute("DETACH DATABASE main_db")
        cursor.execute("DETACH DATABASE sub_db")
        conn.close()
        
        messagebox.showinfo(
            "Ergebnis", 
            f"Merge erfolgreich in neue Datei abgeschlossen!\n\nDie neue Datenbank enthält {count_after} Bilder/Erkennungen.\n(Kollidierende Dateinamen wurden automatisch umbenannt)."
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
