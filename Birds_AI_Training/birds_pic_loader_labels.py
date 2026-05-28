import os
import json
import time
import requests
from io import BytesIO
from PIL import Image
from duckduckgo_search import DDGS

# --- KONFIGURATION ---
LABELS_DATEI = "model_labels.json"    # JSON-Datei mit allen Vogelarten
ANZAHL_PRO_SUCHE = 30                 # Bilder pro Suchbegriff
DOWNLOAD_ORDNER = "neue_bilder_downloads"
pic_min_size = 600                    # Minimale Kantenlänge (Breite UND Höhe) in Pixeln

def load_labels(json_path):
    """Lädt alle Vogelarten aus der model_labels.json Datei."""
    with open(json_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    # Werte (Artnamen) als sortierte Liste zurückgeben
    arten = sorted(labels.values())
    return arten

def clean_filename(query, index):
    """Erstellt saubere Dateinamen: Stieglitz_Vogel_DDG_01.jpg"""
    # Leerzeichen durch Unterstriche ersetzen
    clean_query = query.replace(" ", "_")
    # Umlaute ersetzen (optional, aber sicherer für manche Windows-Systeme)
    clean_query = clean_query.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    
    return f"{clean_query}_DDG_{index}.jpg"

def download_images(query, folder, max_images, min_size=pic_min_size):
    """Lädt Bilder via DuckDuckGo herunter (nur wenn beide Kanten >= min_size)"""
    print(f"   > Suche nach: '{query}'...")
    
    count = 0
    try:
        with DDGS() as ddgs:
            # Suche ausführen
            results = ddgs.images(
                query, 
                region="de-de", # WICHTIG: Region auf Deutschland gesetzt für bessere lokale Ergebnisse
                safesearch="off", 
                max_results=max_images
            )
            
            for result in results:
                image_url = result.get('image')
                if not image_url: continue
                
                try:
                    # Request mit Timeout
                    response = requests.get(image_url, timeout=5)
                    
                    if response.status_code == 200:
                        # Bildgröße prüfen
                        try:
                            img = Image.open(BytesIO(response.content))
                            width, height = img.size
                        except Exception:
                            print(f"      - Übersprungen (ungültiges Bild)")
                            continue
                        
                        if width < min_size or height < min_size:
                            print(f"      - Übersprungen: {width}x{height} (min. {min_size}x{min_size} erforderlich)")
                            continue
                        
                        # Dateiendung ermitteln
                        file_ext = os.path.splitext(image_url)[1].lower()
                        if file_ext not in ['.jpg', '.jpeg', '.png']:
                            file_ext = '.jpg'
                        
                        # Dateinamen erstellen
                        filename = clean_filename(query, count)
                        filepath = os.path.join(folder, filename)
                        
                        # Speichern
                        with open(filepath, 'wb') as f:
                            f.write(response.content)
                        
                        print(f"      + Gespeichert: {filename} ({width}x{height})")
                        count += 1
                        
                except Exception:
                    continue

    except Exception as e:
        print(f"   ! Fehler bei der Suche nach '{query}': {e}")
    
    print(f"   -> {count} Bilder geladen.")
    return count

def start_download():
    # 1. Alle Vogelarten aus der JSON-Datei laden
    if not os.path.exists(LABELS_DATEI):
        print(f"FEHLER: Datei '{LABELS_DATEI}' nicht gefunden!")
        return
    
    arten_liste = load_labels(LABELS_DATEI)
    gesamt_arten = len(arten_liste)
    
    print("=" * 60)
    print(f"  Bilder-Download für {gesamt_arten} Arten aus '{LABELS_DATEI}'")
    print("=" * 60)
    print(f"  Zielordner:      {DOWNLOAD_ORDNER}")
    print(f"  Bilder pro Suche: {ANZAHL_PRO_SUCHE}")
    print(f"  Min. Bildgröße:  {pic_min_size}x{pic_min_size} px")
    print("=" * 60)
    
    gesamt_downloaded = 0
    ergebnis_pro_art = {}  # Zusammenfassung am Ende

    # 2. Jede Art nacheinander abarbeiten
    for art_index, vogelart in enumerate(arten_liste, start=1):
        print(f"\n{'─' * 60}")
        print(f"  [{art_index}/{gesamt_arten}] {vogelart}")
        print(f"{'─' * 60}")
        
        # Zielordner für diese Art vorbereiten
        final_target_dir = os.path.join(DOWNLOAD_ORDNER, vogelart)
        if not os.path.exists(final_target_dir):
            os.makedirs(final_target_dir)

        # Deutsche Suchbegriffe generieren
        # Vielseitige Varianten für unterschiedliche Posen/Hintergründe
        search_queries = [
            f"{vogelart}",               # Einfach nur der Name
            f"{vogelart} Vogel",         # Standard
            f"{vogelart} im Flug",       # Action-Bilder (wichtig!)
            f"{vogelart} Futterplatz",   # Kontext: Futterhaus
            f"{vogelart} Garten",        # Kontext: Natur/Hintergrund
            f"{vogelart} Fotografie",    # Oft qualitativ hochwertige Bilder
            f"{vogelart} Weibchen"       # Falls Geschlechtsunterschiede existieren
        ]

        print(f"   Zielordner: {final_target_dir}")
        
        art_downloaded = 0

        # Alle Suchbegriffe für diese Art abarbeiten
        for query in search_queries:
            try:
                count = download_images(query, final_target_dir, ANZAHL_PRO_SUCHE)
                art_downloaded += count
                
                # Kurze Pause für DuckDuckGo
                time.sleep(2) 
                
            except Exception as e:
                print(f"   Kritischer Fehler bei '{query}': {e}")

        ergebnis_pro_art[vogelart] = art_downloaded
        gesamt_downloaded += art_downloaded
        print(f"\n   >>> {vogelart}: {art_downloaded} Bilder geladen.")

    # 3. Gesamtzusammenfassung ausgeben
    print("\n" + "=" * 60)
    print("  GESAMTÜBERSICHT")
    print("=" * 60)
    for art, anzahl in ergebnis_pro_art.items():
        status = "✓" if anzahl > 0 else "✗"
        print(f"   {status} {art:.<35} {anzahl:>4} Bilder")
    print("─" * 60)
    print(f"   GESAMT: {gesamt_downloaded} Bilder für {gesamt_arten} Arten")
    print("=" * 60)

if __name__ == "__main__":
    start_download()