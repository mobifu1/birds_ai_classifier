import os
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

class FolderNameRenamer:
    def __init__(self, root):
        self.root = root
        self.root.title("Bilder-Umbenenner: 0001_Ordnername")
        self.root.geometry("600x450")

        # UI Elemente
        title = tk.Label(root, text="Schema: 0001_Ordnername.jpg", font=("Arial", 14, "bold"))
        title.pack(pady=10)

        desc = tk.Label(root, text="Jeder Unterordner startet neu bei 0001.\nDer Name des Ordners wird Teil des Dateinamens.")
        desc.pack(pady=5)

        self.btn_select = tk.Button(root, text="Ordner auswählen & Starten", 
                                    command=self.process_renaming, 
                                    bg="#008CBA", fg="white", 
                                    font=("Arial", 11), padx=20, pady=10)
        self.btn_select.pack(pady=15)

        # Log-Bereich
        self.log_text = scrolledtext.ScrolledText(root, height=15, width=70, state='disabled')
        self.log_text.pack(pady=5, padx=10)

        self.status_label = tk.Label(root, text="Warte auf Start...", fg="grey")
        self.status_label.pack(pady=5)

    def log(self, message):
        """Schreibt Nachrichten in das Log-Fenster"""
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')
        self.root.update()

    def process_renaming(self):
        root_path = filedialog.askdirectory()
        if not root_path:
            return

        self.log_text.config(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state='disabled')
        
        self.log(f"Starte im Ordner: {root_path}")
        
        folders_processed = 0
        images_renamed = 0

        # Rekursiv durch alle Ordner gehen
        for current_root, dirs, files in os.walk(root_path):
            
            # Bilder filtern
            images = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if not images:
                continue

            # Sortieren, damit die Reihenfolge konstant bleibt
            images.sort()
            
            # Ordnernamen ermitteln (für den Dateinamen)
            # os.path.basename holt den letzten Teil des Pfades (den Ordnernamen)
            folder_name = os.path.basename(current_root)
            
            # Falls Leerzeichen im Ordnernamen sind, ersetzen wir sie evtl. (optional, hier lassen wir sie)
            # folder_name = folder_name.replace(" ", "_") 

            self.log(f"-> Bearbeite: '{folder_name}' ({len(images)} Bilder)")

            # Zähler resetten für diesen Ordner
            count = 1

            # --- SCHRITT 1: Temporäres Umbenennen ---
            # Das verhindert Fehler, falls '0001_Urlaub.jpg' schon existiert
            temp_files = []
            for filename in images:
                extension = os.path.splitext(filename)[1]
                old_path = os.path.join(current_root, filename)
                
                # Temporärer Name
                temp_name = f"__temp_{count}_{filename}"
                temp_path = os.path.join(current_root, temp_name)
                
                try:
                    os.rename(old_path, temp_path)
                    temp_files.append(temp_path)
                    count += 1
                except Exception as e:
                    self.log(f"   Fehler (Temp): {e}")

            # --- SCHRITT 2: Finales Umbenennen ---
            # Schema: 0001_Ordnername.jpg
            final_count = 1
            for temp_path in temp_files:
                extension = os.path.splitext(temp_path)[1]
                
                # Hier entsteht der neue Name:
                # {final_count:04d} -> 0001
                # folder_name -> Name des aktuellen Ordners
                new_filename = f"{final_count:04d}_{folder_name}{extension}"
                
                final_path = os.path.join(current_root, new_filename)

                try:
                    os.rename(temp_path, final_path)
                    final_count += 1
                    images_renamed += 1
                except Exception as e:
                    self.log(f"   Fehler (Final): {e}")

            folders_processed += 1

        self.status_label.config(text="Fertig!", fg="green")
        messagebox.showinfo("Abschluss", f"Fertig!\n{folders_processed} Ordner bearbeitet.\n{images_renamed} Bilder umbenannt.")

if __name__ == "__main__":
    root = tk.Tk()
    app = FolderNameRenamer(root)
    root.mainloop()