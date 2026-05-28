import sqlite3
import tkinter as tk
from tkinter import ttk, messagebox
import os
from datetime import datetime

# --- Konfiguration ---
DB_FILENAME = "birds_stats.db"
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), DB_FILENAME)


class BirdsDatabaseEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Birds Database Editor")
        self.root.geometry("820x650")
        self.root.resizable(True, True)
        self.root.minsize(700, 550)
        self.root.configure(bg="#1e1e2e")

        # Styling
        self.colors = {
            "bg": "#1e1e2e",
            "surface": "#2a2a3d",
            "surface_light": "#35354d",
            "accent": "#7c3aed",
            "accent_hover": "#6d28d9",
            "danger": "#ef4444",
            "danger_hover": "#dc2626",
            "text": "#e2e8f0",
            "text_dim": "#94a3b8",
            "border": "#3f3f5c",
            "success": "#22c55e",
            "warning": "#f59e0b",
        }

        self._setup_styles()
        self._build_ui()
        self._load_dates()

    def _setup_styles(self):
        """Konfiguriert ttk Styles für die App."""
        self.style = ttk.Style()
        self.style.theme_use("clam")

        # Treeview Style
        self.style.configure(
            "Custom.Treeview",
            background=self.colors["surface"],
            foreground=self.colors["text"],
            fieldbackground=self.colors["surface"],
            borderwidth=0,
            rowheight=28,
            font=("Segoe UI", 10),
        )
        self.style.configure(
            "Custom.Treeview.Heading",
            background=self.colors["surface_light"],
            foreground=self.colors["text"],
            borderwidth=0,
            font=("Segoe UI", 10, "bold"),
        )
        self.style.map(
            "Custom.Treeview",
            background=[("selected", self.colors["accent"])],
            foreground=[("selected", "#ffffff")],
        )

    def _build_ui(self):
        """Erstellt die gesamte Benutzeroberfläche."""
        # --- Header ---
        header_frame = tk.Frame(self.root, bg=self.colors["bg"])
        header_frame.pack(fill="x", padx=25, pady=(20, 5))

        tk.Label(
            header_frame,
            text="🐦  Birds Database Editor",
            font=("Segoe UI", 20, "bold"),
            bg=self.colors["bg"],
            fg=self.colors["text"],
        ).pack(side="left")

        # DB-Status Label
        self.lbl_db_status = tk.Label(
            header_frame,
            text="",
            font=("Segoe UI", 9),
            bg=self.colors["bg"],
            fg=self.colors["text_dim"],
        )
        self.lbl_db_status.pack(side="right", pady=(8, 0))

        # Trennlinie
        tk.Frame(self.root, bg=self.colors["border"], height=1).pack(fill="x", padx=25, pady=(5, 15))

        # --- Oberer Bereich: Datumsauswahl ---
        select_frame = tk.Frame(self.root, bg=self.colors["surface"], highlightbackground=self.colors["border"], highlightthickness=1)
        select_frame.pack(fill="x", padx=25, pady=(0, 10))

        inner_select = tk.Frame(select_frame, bg=self.colors["surface"])
        inner_select.pack(fill="x", padx=20, pady=15)

        tk.Label(
            inner_select,
            text="Tag auswählen:",
            font=("Segoe UI", 12, "bold"),
            bg=self.colors["surface"],
            fg=self.colors["text"],
        ).pack(side="left", padx=(0, 15))

        # Datum-Dropdown
        self.date_var = tk.StringVar()
        self.date_combo = ttk.Combobox(
            inner_select,
            textvariable=self.date_var,
            state="readonly",
            width=30,
            font=("Segoe UI", 11),
        )
        self.date_combo.pack(side="left", padx=(0, 15))
        self.date_combo.bind("<<ComboboxSelected>>", self._on_date_selected)

        # Vorschau-Button
        self.btn_preview = tk.Button(
            inner_select,
            text="📋  Einträge anzeigen",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors["accent"],
            fg="white",
            activebackground=self.colors["accent_hover"],
            activeforeground="white",
            relief="flat",
            cursor="hand2",
            padx=15,
            pady=6,
            command=self._preview_entries,
        )
        self.btn_preview.pack(side="left", padx=(0, 10))

        # Aktualisieren-Button
        self.btn_refresh = tk.Button(
            inner_select,
            text="🔄",
            font=("Segoe UI", 12),
            bg=self.colors["surface_light"],
            fg=self.colors["text"],
            activebackground=self.colors["border"],
            activeforeground=self.colors["text"],
            relief="flat",
            cursor="hand2",
            padx=6,
            pady=2,
            command=self._load_dates,
        )
        self.btn_refresh.pack(side="left")

        # --- Info-Leiste ---
        info_frame = tk.Frame(self.root, bg=self.colors["bg"])
        info_frame.pack(fill="x", padx=25, pady=(0, 5))

        self.lbl_info = tk.Label(
            info_frame,
            text="Bitte wähle einen Tag aus der Liste.",
            font=("Segoe UI", 10),
            bg=self.colors["bg"],
            fg=self.colors["text_dim"],
            anchor="w",
        )
        self.lbl_info.pack(side="left")

        self.lbl_count = tk.Label(
            info_frame,
            text="",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors["bg"],
            fg=self.colors["warning"],
            anchor="e",
        )
        self.lbl_count.pack(side="right")

        # --- Treeview für Vorschau ---
        tree_container = tk.Frame(self.root, bg=self.colors["border"])
        tree_container.pack(fill="both", expand=True, padx=25, pady=(0, 10))

        tree_inner = tk.Frame(tree_container, bg=self.colors["surface"])
        tree_inner.pack(fill="both", expand=True, padx=1, pady=1)

        columns = ("id", "filename", "species", "timestamp", "confidence")
        self.tree = ttk.Treeview(
            tree_inner,
            columns=columns,
            show="headings",
            style="Custom.Treeview",
            selectmode="none",
        )

        # Spalten definieren
        self.tree.heading("id", text="ID")
        self.tree.heading("filename", text="Dateiname")
        self.tree.heading("species", text="Vogelart")
        self.tree.heading("timestamp", text="Zeitstempel")
        self.tree.heading("confidence", text="Konfidenz")

        self.tree.column("id", width=60, minwidth=50, anchor="center")
        self.tree.column("filename", width=220, minwidth=150)
        self.tree.column("species", width=150, minwidth=100)
        self.tree.column("timestamp", width=170, minwidth=130, anchor="center")
        self.tree.column("confidence", width=100, minwidth=80, anchor="center")

        # Scrollbar
        scrollbar = ttk.Scrollbar(tree_inner, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.tree.pack(fill="both", expand=True)

        # --- Unterer Bereich: Löschen-Button ---
        bottom_frame = tk.Frame(self.root, bg=self.colors["bg"])
        bottom_frame.pack(fill="x", padx=25, pady=(0, 20))

        self.btn_delete = tk.Button(
            bottom_frame,
            text="🗑️  Alle Einträge für diesen Tag löschen",
            font=("Segoe UI", 12, "bold"),
            bg=self.colors["danger"],
            fg="white",
            activebackground=self.colors["danger_hover"],
            activeforeground="white",
            relief="flat",
            cursor="hand2",
            padx=20,
            pady=10,
            state="disabled",
            command=self._delete_entries,
        )
        self.btn_delete.pack(side="right")

        # Gesamtanzahl Label
        self.lbl_total = tk.Label(
            bottom_frame,
            text="",
            font=("Segoe UI", 10),
            bg=self.colors["bg"],
            fg=self.colors["text_dim"],
            anchor="w",
        )
        self.lbl_total.pack(side="left", pady=(8, 0))

    def _get_connection(self):
        """Stellt eine Verbindung zur Datenbank her."""
        if not os.path.exists(DB_PATH):
            messagebox.showerror("Fehler", f"Datenbank nicht gefunden:\n{DB_PATH}")
            return None
        return sqlite3.connect(DB_PATH)

    def _load_dates(self):
        """Lädt alle verfügbaren Tage aus der Datenbank in das Dropdown."""
        conn = self._get_connection()
        if not conn:
            return

        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT DISTINCT DATE(timestamp) as d, COUNT(*) as c "
                "FROM detections GROUP BY d ORDER BY d DESC"
            )
            rows = cursor.fetchall()

            # Gesamtanzahl
            cursor.execute("SELECT COUNT(*) FROM detections")
            total = cursor.fetchone()[0]
            self.lbl_total.config(text=f"Gesamt: {total:,} Einträge in der Datenbank".replace(",", "."))
            self.lbl_db_status.config(text=f"📂 {DB_FILENAME}", fg=self.colors["success"])

            # Dropdown befüllen: "2026-04-25  (25 Einträge)"
            date_entries = []
            for date_str, count in rows:
                if date_str:
                    # Datum umformatieren: YYYY-MM-DD -> DD.MM.YYYY
                    try:
                        dt = datetime.strptime(date_str, "%Y-%m-%d")
                        display_date = dt.strftime("%d.%m.%Y")
                    except ValueError:
                        display_date = date_str
                    date_entries.append(f"{display_date}   ({count} Einträge)")

            self.date_combo["values"] = date_entries
            if date_entries:
                self.date_combo.current(0)

            # Vorschau und Löschen zurücksetzen
            self.tree.delete(*self.tree.get_children())
            self.btn_delete.config(state="disabled")
            self.lbl_info.config(text="Bitte wähle einen Tag aus der Liste.", fg=self.colors["text_dim"])
            self.lbl_count.config(text="")

        except sqlite3.Error as e:
            messagebox.showerror("Datenbankfehler", f"Fehler beim Laden der Daten:\n{e}")
        finally:
            conn.close()

    def _get_selected_date_iso(self):
        """Gibt das aktuell ausgewählte Datum im ISO-Format (YYYY-MM-DD) zurück."""
        selection = self.date_var.get()
        if not selection:
            return None
        # Extrahiert DD.MM.YYYY aus "DD.MM.YYYY   (X Einträge)"
        date_part = selection.split("   ")[0].strip()
        try:
            dt = datetime.strptime(date_part, "%d.%m.%Y")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return None

    def _on_date_selected(self, event=None):
        """Wird aufgerufen, wenn ein Datum im Dropdown ausgewählt wird."""
        self.tree.delete(*self.tree.get_children())
        self.btn_delete.config(state="disabled")
        self.lbl_count.config(text="")
        date_iso = self._get_selected_date_iso()
        if date_iso:
            display = self.date_var.get().split("   ")[0].strip()
            self.lbl_info.config(
                text=f"Tag {display} ausgewählt – klicke auf 'Einträge anzeigen' für eine Vorschau.",
                fg=self.colors["text"],
            )

    def _preview_entries(self):
        """Zeigt alle Einträge für den ausgewählten Tag in der Treeview an."""
        date_iso = self._get_selected_date_iso()
        if not date_iso:
            messagebox.showwarning("Hinweis", "Bitte wähle zuerst einen Tag aus.")
            return

        conn = self._get_connection()
        if not conn:
            return

        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, filename, species, timestamp, confidence "
                "FROM detections WHERE DATE(timestamp) = ? ORDER BY timestamp ASC",
                (date_iso,),
            )
            rows = cursor.fetchall()

            # Treeview leeren
            self.tree.delete(*self.tree.get_children())

            # Zebra-Farben für bessere Lesbarkeit
            self.tree.tag_configure("even", background=self.colors["surface"])
            self.tree.tag_configure("odd", background=self.colors["surface_light"])

            for i, row in enumerate(rows):
                tag = "even" if i % 2 == 0 else "odd"
                confidence_display = f"{row[4]:.2%}" if row[4] is not None else "–"
                self.tree.insert(
                    "",
                    "end",
                    values=(row[0], row[1], row[2], row[3], confidence_display),
                    tags=(tag,),
                )

            count = len(rows)
            display_date = self.date_var.get().split("   ")[0].strip()

            if count > 0:
                self.lbl_info.config(
                    text=f"Vorschau für {display_date}:",
                    fg=self.colors["text"],
                )
                self.lbl_count.config(text=f"{count} Einträge gefunden")
                self.btn_delete.config(state="normal")
            else:
                self.lbl_info.config(
                    text=f"Keine Einträge für {display_date} gefunden.",
                    fg=self.colors["warning"],
                )
                self.lbl_count.config(text="")
                self.btn_delete.config(state="disabled")

        except sqlite3.Error as e:
            messagebox.showerror("Datenbankfehler", f"Fehler beim Laden der Vorschau:\n{e}")
        finally:
            conn.close()

    def _delete_entries(self):
        """Löscht alle Einträge für den ausgewählten Tag nach Bestätigung."""
        date_iso = self._get_selected_date_iso()
        if not date_iso:
            return

        display_date = self.date_var.get().split("   ")[0].strip()

        # Anzahl ermitteln
        conn = self._get_connection()
        if not conn:
            return

        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM detections WHERE DATE(timestamp) = ?",
                (date_iso,),
            )
            count = cursor.fetchone()[0]
        except sqlite3.Error as e:
            messagebox.showerror("Datenbankfehler", str(e))
            conn.close()
            return

        conn.close()

        if count == 0:
            messagebox.showinfo("Info", "Es gibt keine Einträge für diesen Tag.")
            return

        # Sicherheitsabfrage
        confirm = messagebox.askyesno(
            "⚠️ Löschen bestätigen",
            f"Möchtest du wirklich ALLE {count} Einträge\n"
            f"vom {display_date} unwiderruflich löschen?\n\n"
            f"Diese Aktion kann nicht rückgängig gemacht werden!",
            icon="warning",
        )

        if not confirm:
            return

        # Zweite Sicherheitsabfrage bei vielen Einträgen
        if count > 100:
            confirm2 = messagebox.askyesno(
                "⚠️ Zusätzliche Bestätigung",
                f"ACHTUNG: Es werden {count} Einträge gelöscht!\n\n"
                f"Bist du dir absolut sicher?",
                icon="warning",
            )
            if not confirm2:
                return

        # Löschen durchführen
        conn = self._get_connection()
        if not conn:
            return

        try:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM detections WHERE DATE(timestamp) = ?",
                (date_iso,),
            )
            deleted = cursor.rowcount
            conn.commit()

            messagebox.showinfo(
                "✅ Erfolgreich gelöscht",
                f"{deleted} Einträge vom {display_date}\n"
                f"wurden erfolgreich aus der Datenbank entfernt.",
            )

            # UI aktualisieren
            self._load_dates()

        except sqlite3.Error as e:
            conn.rollback()
            messagebox.showerror("Datenbankfehler", f"Fehler beim Löschen:\n{e}")
        finally:
            conn.close()


# --- Hauptprogramm starten ---
if __name__ == "__main__":
    root = tk.Tk()
    app = BirdsDatabaseEditor(root)
    root.mainloop()
