import os
import time
import requests
import re
from io import BytesIO
from PIL import Image
from duckduckgo_search import DDGS

# --- KONFIGURATION ---
GESUCHTER_VOGEL = "Schwanzmeise"     # Nur noch der deutsche Name nötig
ANZAHL_PRO_SUCHE = 30             # Bilder pro Suchbegriff
DOWNLOAD_ORDNER = "neue_bilder_downloads"
pic_min_size = 500                # Minimale Kantenlänge (Breite UND Höhe) in Pixeln

def clean_filename(query, index, engine="DDG"):
    """Erstellt saubere Dateinamen: Stieglitz_Vogel_DDG_01.jpg"""
    # Leerzeichen durch Unterstriche ersetzen
    clean_query = query.replace(" ", "_")
    # Umlaute ersetzen (optional, aber sicherer für manche Windows-Systeme)
    clean_query = clean_query.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    
    return f"{clean_query}_{engine}_{index}.jpg"

def get_bing_images(query, max_images):
    """Holt Bild-URLs von Bing Images"""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    url = f"https://www.bing.com/images/search?q={query}"
    try:
        res = requests.get(url, headers=headers, timeout=10)
        links = re.findall(r'murl&quot;:&quot;(.*?)&quot;', res.text)
        return links[:max_images]
    except Exception as e:
        print(f"   ! Fehler bei Bing-Suche: {e}")
        return []

def download_images(query, folder, max_images, min_size=pic_min_size, engine="DDG"):
    """Lädt Bilder herunter (nur wenn beide Kanten >= min_size)"""
    print(f"   > Suche nach: '{query}' via {engine}...")
    
    count = 0
    try:
        image_urls = []
        if engine == "DDG":
            with DDGS() as ddgs:
                results = ddgs.images(query, region="de-de", safesearch="off", max_results=max_images)
                image_urls = [r.get('image') for r in results if r.get('image')]
        elif engine == "BING":
            image_urls = get_bing_images(query, max_images)

        for image_url in image_urls:
            if not image_url: continue
            
            try:
                response = requests.get(image_url, timeout=5)
                if response.status_code == 200:
                    try:
                        img = Image.open(BytesIO(response.content))
                        width, height = img.size
                    except Exception:
                        print(f"      - Übersprungen (ungültiges Bild)")
                        continue
                    
                    if width < min_size or height < min_size:
                        print(f"      - Übersprungen: {width}x{height} (min. {min_size}x{min_size} erforderlich)")
                        continue
                    
                    filename = clean_filename(query, count, engine)
                    filepath = os.path.join(folder, filename)
                    
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    
                    print(f"      + Gespeichert: {filename} ({width}x{height})")
                    count += 1
                    
            except Exception:
                continue

    except Exception as e:
        print(f"   ! Fehler bei der Suche nach '{query}' via {engine}: {e}")
    
    print(f"   -> {count} Bilder geladen via {engine}.")
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
    engines = ["DDG", "BING"]
    for query in search_queries:
        for engine in engines:
            try:
                count = download_images(query, final_target_dir, ANZAHL_PRO_SUCHE, engine=engine)
                total_downloaded += count
                
                # Kurze Pause für Suchmaschinen
                time.sleep(2) 
                
            except Exception as e:
                print(f"Kritischer Fehler bei '{query}' ({engine}): {e}")

    print("\n" + "="*50)
    print(f"FERTIG! Insgesamt {total_downloaded} Bilder im Ordner:")
    print(f"{final_target_dir}")
    print("="*50)

if __name__ == "__main__":
    start_download()