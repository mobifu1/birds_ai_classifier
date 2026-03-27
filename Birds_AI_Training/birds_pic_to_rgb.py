import os
from PIL import Image

# --- KONFIGURATION ---
CURRENT_DIR = os.getcwd()
DATASET_NAME = "birds_training_dataset"
DATASET_PFAD = os.path.join(CURRENT_DIR, DATASET_NAME)

def convert_images_to_rgb():
    """
    Durchsucht den Ordner 'birds_training_dataset' nach Bildern und wandelt
    alle Bilder, die nicht im Standard-RGB-Format sind (wie z.B. Bilder mit
    Farbpaletten oder Transparenz), in lupenreines RGB ohne Transparenz um.
    """
    if not os.path.exists(DATASET_PFAD):
        print(f"FEHLER: Der Ordner '{DATASET_NAME}' wurde nicht gefunden.")
        return

    count_total = 0
    count_converted = 0

    print(f"Überprüfe Bilder in '{DATASET_PFAD}'...")

    for subdir, dirs, files in os.walk(DATASET_PFAD):
        for file in files:
            # Wir betrachten die gängigen Bildformate
            if file.lower().endswith(('.png', '.gif', '.jpg', '.jpeg', '.bmp', '.webp')):
                file_path = os.path.join(subdir, file)
                count_total += 1
                
                try:
                    with Image.open(file_path) as img:
                        original_mode = img.mode
                        
                        # Wir überprüfen, ob das Bild noch nicht reines RGB ist
                        if original_mode != 'RGB':
                            # Wenn Transparenz (Alpha-Kanal) vorhanden ist: Weißen Hintergrund anlegen
                            # (Schützt davor, dass transparente Ränder einfach schwarz werden)
                            if original_mode in ('RGBA', 'LA') or (original_mode == 'P' and 'transparency' in img.info):
                                alpha_image = img.convert('RGBA')
                                background = Image.new('RGB', alpha_image.size, (255, 255, 255))
                                # Das transparente Bild auf den weißen Hintergrund kleben
                                background.paste(alpha_image, mask=alpha_image.split()[3])
                                final_img = background
                            else:
                                # Normales Konvertieren für reine Graustufen- oder Farbpaletten-Bilder
                                final_img = img.convert('RGB')
                                
                            # Wir speichern das Bild im reinen RGB-Format wieder ab
                            final_img.save(file_path)
                            count_converted += 1
                            print(f"[Konvertiert] {file} (Format war: {original_mode})")
                            
                except Exception as e:
                    print(f"[Fehler] Konnte Bild '{file_path}' nicht verarbeiten: {e}")

    print("\n" + "="*50)
    print("ZUSAMMENFASSUNG:")
    print(f"Überprüfte Bilder insgesamt: {count_total}")
    print(f"Davon in RGB umgewandelt:    {count_converted}")
    print("="*50)
    print("Fertig! Die lästige Warnung beim Training sollte nun verschwinden.")

if __name__ == '__main__':
    print("Starte Bild-Korrektur...")
    convert_images_to_rgb()
