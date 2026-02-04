import os
import time
import requests
from duckduckgo_search import DDGS

# --- KONFIGURATION ---
GESUCHTER_VOGEL = "Tannenmeise"     # Nur noch der deutsche Name nötig
ANZAHL_PRO_SUCHE = 30             # Bilder pro Suchbegriff
DOWNLOAD_ORDNER = "neue_bilder_downloads"

def clean_filename(query, index):
    """Erstellt saubere Dateinamen: Stieglitz_Vogel_DDG_01.jpg"""
    # Leerzeichen durch Unterstriche ersetzen
    clean_query = query.replace(" ", "_")
    # Umlaute ersetzen (optional, aber sicherer für manche Windows-Systeme)
    clean_query = clean_query.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    
    return f"{clean_query}_DDG_{index}.jpg"

def download_images(query, folder, max_images):
    """Lädt Bilder via DuckDuckGo herunter"""
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
                        
                        print(f"      + Gespeichert: {filename}")
                        count += 1
                        
                except Exception:
                    continue

    except Exception as e:
        print(f"   ! Fehler bei der Suche nach '{query}': {e}")
    
    print(f"   -> {count} Bilder geladen.")
    return count

def start_download():
    # 1. Zielordner vorbereiten
    final_target_dir = os.path.join(DOWNLOAD_ORDNER, GESUCHTER_VOGEL)
    if not os.path.exists(final_target_dir):
        os.makedirs(final_target_dir)

    # 2. Deutsche Suchbegriffe generieren
    # Da wir nur deutsch suchen, habe ich die Varianten erhöht,
    # um trotzdem vielseitige Bilder (Hintergründe/Posen) zu bekommen.
    search_queries = [
        f"{GESUCHTER_VOGEL}",               # Einfach nur der Name
        f"{GESUCHTER_VOGEL} Vogel",         # Standard
        f"{GESUCHTER_VOGEL} im Flug",       # Action-Bilder (wichtig!)
        f"{GESUCHTER_VOGEL} Futterplatz",   # Kontext: Futterhaus
        f"{GESUCHTER_VOGEL} Garten",        # Kontext: Natur/Hintergrund
        f"{GESUCHTER_VOGEL} Fotografie",    # Oft qualitativ hochwertige Bilder
        f"{GESUCHTER_VOGEL} Weibchen"       # Falls Geschlechtsunterschiede existieren
    ]

    print(f"\n--- Starte Deutsche Suche für '{GESUCHTER_VOGEL}' ---")
    print(f"Zielordner: {final_target_dir}\n")

    total_downloaded = 0

    # 3. Alle Suchbegriffe abarbeiten
    for query in search_queries:
        try:
            count = download_images(query, final_target_dir, ANZAHL_PRO_SUCHE)
            total_downloaded += count
            
            # Kurze Pause für DuckDuckGo
            time.sleep(2) 
            
        except Exception as e:
            print(f"Kritischer Fehler bei '{query}': {e}")

    print("\n" + "="*50)
    print(f"FERTIG! Insgesamt {total_downloaded} Bilder im Ordner:")
    print(f"{final_target_dir}")
    print("="*50)

if __name__ == "__main__":
    start_download()