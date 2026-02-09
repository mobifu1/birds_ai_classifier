import os
import json
import sys
import shutil
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
# Wechsel von MobileNetV2 zu InceptionV3
from tensorflow.keras.applications import InceptionV3 
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# --- KONFIGURATION ---
CURRENT_DIR = os.getcwd()
DATASET_NAME = "birds_training_dataset"
DATASET_TEMP_NAME = "birds_training_dataset_shuffled_temp"
DATASET_PFAD = os.path.join(CURRENT_DIR, DATASET_NAME)
DATASET_TEMP_PFAD = os.path.join(CURRENT_DIR, DATASET_TEMP_NAME)

MODEL_DATEI = "my_birds_modell_400.keras"
LABELS_DATEI = "model_labels.json"

# --- ÄNDERUNG: Jetzt auf 400x400 gesetzt ---
IMG_SIZE = (400, 400) 
# -----------------------------------------------------------------

BATCH_SIZE = 8 
EPOCHS = 40  

# --- FUNKTIONEN (unverändert) ---

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

def prepare_shuffled_dataset():
    """Erstellt eine Kopie des Datensatzes und mischt die Dateinamen."""
    if os.path.exists(DATASET_TEMP_PFAD):
        print(f"Lösche alten Temp-Ordner {DATASET_TEMP_NAME}...")
        try:
            shutil.rmtree(DATASET_TEMP_PFAD)
        except OSError as e:
            print(f"Warnung: {e}")
            
    os.makedirs(DATASET_TEMP_PFAD, exist_ok=True)
    print(f"Erstelle gemischten Datensatz auf {IMG_SIZE}...")
    
    count = 0
    for subdir, dirs, files in os.walk(DATASET_PFAD):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                rel_path = os.path.relpath(subdir, DATASET_PFAD)
                target_subdir = os.path.join(DATASET_TEMP_PFAD, rel_path)
                os.makedirs(target_subdir, exist_ok=True)
                
                try:
                    img = load_img(os.path.join(subdir, file), target_size=IMG_SIZE)
                    img_array = img_to_array(img)
                    random_prefix = str(random.randint(10000, 99999))
                    new_filename = f"{random_prefix}_{file}"
                    save_path = os.path.join(target_subdir, new_filename)
                    array_to_img(img_array).save(save_path)
                    count += 1
                except Exception as e:
                    print(f"Fehler bei Bild {file}: {e}")
    print(f"Fertig. {count} Bilder vorbereitet.")

def check_data_before_start():
    if not os.path.exists(DATASET_PFAD):
        print(f"FEHLER: Der Ordner '{DATASET_NAME}' existiert nicht!")
        sys.exit(1)

def train():
    check_data_before_start()
    prepare_shuffled_dataset()

    # Anpassung: Preprocessing für InceptionV3 nutzen
    datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.inception_v3.preprocess_input,
        rotation_range=30,      
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2 
    )

    print("Lade Bilder in Generator...")
    train_generator = datagen.flow_from_directory(
        DATASET_TEMP_PFAD,  
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        DATASET_TEMP_PFAD,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    if train_generator.samples == 0:
        print("ABBRUCH: Keine Bilder geladen.")
        return

    class_indices = train_generator.class_indices
    labels = {v: k for k, v in class_indices.items()}
    with open(LABELS_DATEI, 'w') as f:
        json.dump(labels, f)
    print(f"Klassen gefunden: {len(labels)}")

    # --- MODELL-WECHSEL ZU INCEPTION V3 ---
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    
    base_model.trainable = True
    # Fine-Tuning der letzten Schichten
    fine_tune_at = len(base_model.layers) - 40 
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

    callbacks_list = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_DATEI, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=1e-7, verbose=1)
    ]

    print("Starte Training mit InceptionV3 (400x400)...")
    
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