import os
import json
import sys
import shutil
import numpy as np
import random  # <--- NEU: Für den Zufallsmix
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# --- KONFIGURATION ---
CURRENT_DIR = os.getcwd()
DATASET_NAME = "birds_training_dataset"
DATASET_CLEAN_NAME = "birds_training_dataset_masked_source"
DATASET_PFAD = os.path.join(CURRENT_DIR, DATASET_NAME)
DATASET_CLEAN_PFAD = os.path.join(CURRENT_DIR, DATASET_CLEAN_NAME)

MODEL_DATEI = "my_birds_modell.keras"
LABELS_DATEI = "model_labels.json"
IMG_SIZE = (224, 224)
BATCH_SIZE = 8 # Falls Speicher reicht, gerne auf 16 erhöhen
EPOCHS = 40  

# --- MASK PARAMETER (bezogen auf 224x224 Zielgröße) ---
MASK_TOP = 14    
MASK_BOTTOM = 10 

# --- FUNKTIONEN ---

def erstelle_trainings_bericht(history):
    """Erstellt eine Text-Diagnose basierend auf den Trainingsdaten."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    final_acc = acc[-1] * 100
    final_val_acc = val_acc[-1] * 100
    gap = final_acc - final_val_acc
    best_val_loss = min(val_loss)

    print("\n" + "="*50)
    print("          TRAININGS-ANALYSE BERICHT          ")
    print("="*50)
    print(f"Genauigkeit Training:                 {final_acc:.2f}%")
    print(f"Genauigkeit Validierung:              {final_val_acc:.2f}%")
    print(f"Abweichung (Gap):                     {gap:.2f}%")
    print("-" * 50)
    
    # Diagnose Logik
    if gap > 15:
        print("🔴 OVERFITTING ERKANNT! (Gap zu groß)")
    elif final_acc < 60:
        print("🟠 UNDERFITTING (Lernt noch nicht richtig)")
    elif val_loss[-1] > best_val_loss + 0.2:
        print("⚠️ WARNUNG: Validation Loss steigt an (Overfitting beginnt)")
    elif final_val_acc > 75 and gap < 10:
        print("🟢 GUTES ERGEBNIS! Robustes Modell.")
    else:
        print("⚪ ERGEBNIS OKAY.")
    print("="*50 + "\n")

def apply_mask_to_array(img_array):
    """Schwärzt die Bereiche im Array (Zeitstempel entfernen)."""
    img_array[:MASK_TOP, :, :] = 0
    h = img_array.shape[0]
    img_array[h-MASK_BOTTOM:, :, :] = 0
    return img_array

def prepare_masked_dataset():
    """
    Erstellt eine maskierte Kopie des Datensatzes UND mischt die Dateinamen.
    Das Mischen ist wichtig, damit 'validation_split' zufällige Bilder wählt 
    und nicht nur die letzten Aufnahmen eines Tages.
    """
    if os.path.exists(DATASET_CLEAN_PFAD):
        print(f"Lösche alten Datensatz {DATASET_CLEAN_NAME} für Neu-Mischung...")
        try:
            shutil.rmtree(DATASET_CLEAN_PFAD)
        except OSError as e:
            print(f"Warnung: {e}")
            
    os.makedirs(DATASET_CLEAN_PFAD, exist_ok=True)
    print(f"Erstelle gemischten & maskierten Datensatz...")
    
    count = 0
    for subdir, dirs, files in os.walk(DATASET_PFAD):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                rel_path = os.path.relpath(subdir, DATASET_PFAD)
                target_subdir = os.path.join(DATASET_CLEAN_PFAD, rel_path)
                os.makedirs(target_subdir, exist_ok=True)
                
                try:
                    # 1. Bild laden und maskieren
                    img = load_img(os.path.join(subdir, file), target_size=IMG_SIZE)
                    img_array = img_to_array(img)
                    img_array = apply_mask_to_array(img_array)
                    
                    # 2. NEU: Zufälligen Präfix generieren (z.B. "84921_")
                    # Das zwingt den Generator später, die Bilder zufällig zu sortieren
                    random_prefix = str(random.randint(10000, 99999))
                    new_filename = f"{random_prefix}_{file}"
                    
                    # 3. Speichern mit neuem Namen
                    save_path = os.path.join(target_subdir, new_filename)
                    array_to_img(img_array).save(save_path)
                    count += 1
                except Exception as e:
                    print(f"Fehler bei Bild {file}: {e}")
    
    print(f"Fertig. {count} Bilder vorbereitet und zufällig benannt.")

def check_data_before_start():
    if not os.path.exists(DATASET_PFAD):
        print(f"FEHLER: Der Ordner '{DATASET_NAME}' existiert nicht!")
        sys.exit(1)

def train():
    check_data_before_start()
    
    # 1. Maskierung & Mischen durchführen
    # Dies wird jetzt bei jedem Start neu gemacht, damit wir immer einen frischen Mix haben
    prepare_masked_dataset()

    # 2. Generator Setup
    datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        rotation_range=30,      
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2 # Nimmt jetzt zufällige 20% dank der Umbenennung
    )

    print("Lade Bilder in Generator...")
    train_generator = datagen.flow_from_directory(
        DATASET_CLEAN_PFAD,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        DATASET_CLEAN_PFAD,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    if train_generator.samples == 0:
        print("ABBRUCH: Keine Bilder geladen.")
        return

    # Labels speichern
    class_indices = train_generator.class_indices
    labels = {v: k for k, v in class_indices.items()}
    with open(LABELS_DATEI, 'w') as f:
        json.dump(labels, f)
    print(f"Klassen gefunden: {len(labels)}")

    # 3. Modell Aufbau
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    base_model.trainable = True
    fine_tune_at = len(base_model.layers) - 30
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x) 
    predictions = Dense(len(labels), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=1e-5), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    # 4. Callbacks
    callbacks_list = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_DATEI, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=1e-7, verbose=1)
    ]

    print("Starte Training mit Random Split...")
    
    history = model.fit(
        train_generator, 
        epochs=EPOCHS, 
        validation_data=validation_generator,
        callbacks=callbacks_list
    )

    print("Training abgeschlossen.")
    erstelle_trainings_bericht(history)
    print(f"Das beste Modell liegt unter: {MODEL_DATEI}")

if __name__ == "__main__":
    train()