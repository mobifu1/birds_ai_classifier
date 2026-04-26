import os
import json
import sys
import shutil
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img # type: ignore
from sklearn.utils.class_weight import compute_class_weight

# --- MATPLOTLIB FÜR GRAPHEN ---
import matplotlib
matplotlib.use('Agg') # Verhindert GUI-Fehler im reinen CMD-Fenster
import matplotlib.pyplot as plt

# --- MODELL: InceptionV3 (wie gewünscht) ---
from tensorflow.keras.applications import InceptionV3 # type: ignore
from tensorflow.keras.applications.inception_v3 import preprocess_input # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# --- KONFIGURATION ---
CURRENT_DIR = os.getcwd()
DATASET_NAME = "birds_training_dataset"
DATASET_TEMP_NAME = "birds_training_dataset_shuffled_temp"
DATASET_PFAD = os.path.join(CURRENT_DIR, DATASET_NAME)
DATASET_TEMP_PFAD = os.path.join(CURRENT_DIR, DATASET_TEMP_NAME)

MODEL_DATEI = "my_birds_modell_299x299.keras"
LABELS_DATEI = "model_labels.json"

# --- AUFLÖSUNG (Höhe, Breite) ---
IMG_HEIGHT = 299
IMG_WIDTH = 299
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH) 
# -----------------------------------------------------------------

BATCH_SIZE = 16
EPOCHS_PHASE1 = 20   # Phase 1: Nur Classifier-Head trainieren
EPOCHS_PHASE2 = 20   # Phase 2: Fine-Tuning der oberen InceptionV3-Layer
FINE_TUNE_FROM = 'mixed7'  # Ab diesem Layer wird aufgetaut
FINE_TUNE_LR = 1e-5        # Niedrigere Lernrate für Fine-Tuning

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
    print("         TRAININGS-ANALYSE BERICHT         ")
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

def speichere_trainings_graph(history, phase1_epochs=None):
    """Speichert den Trainingsverlauf als PNG-Bild ab."""
    print("Erstelle Trainings-Graph...")
    plt.figure(figsize=(12, 5))

    # 1. Graph: Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    if phase1_epochs is not None:
        plt.axvline(x=phase1_epochs - 0.5, color='red', linestyle='--', alpha=0.7, label='Phase 1 → 2')
    plt.title('Modell Genauigkeit')
    plt.xlabel('Epoche')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # 2. Graph: Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    if phase1_epochs is not None:
        plt.axvline(x=phase1_epochs - 0.5, color='red', linestyle='--', alpha=0.7, label='Phase 1 → 2')
    plt.title('Modell Fehlerwert (Loss)')
    plt.xlabel('Epoche')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Speichern und schließen
    plt.tight_layout()
    plt.savefig('trainings_verlauf.png')
    plt.close()
    print("📈 Trainings-Graph wurde als 'trainings_verlauf.png' gespeichert.")

def prepare_shuffled_dataset():
    """Erstellt eine Kopie des Datensatzes und passt die Größe an."""
    if os.path.exists(DATASET_TEMP_PFAD):
        print(f"Lösche alten Temp-Ordner {DATASET_TEMP_NAME}...")
        try:
            shutil.rmtree(DATASET_TEMP_PFAD)
        except OSError as e:
            print(f"Warnung: {e}")
            
    os.makedirs(DATASET_TEMP_PFAD, exist_ok=True)
    print(f"Erstelle gemischten Datensatz auf {IMG_WIDTH}x{IMG_HEIGHT}...")
    
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

class PerClassMetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_generator):
        super().__init__()
        self.val_gen = validation_generator
        self.class_indices = validation_generator.class_indices
        self.labels = {v: k for k, v in self.class_indices.items()}
        self.log_file = "results_ai_training.txt"

    def on_epoch_end(self, epoch, logs=None):
        print(f"\nBerechne klassenspezifische Metriken für Epoche {epoch + 1}...")
        self.val_gen.reset()
        
        y_true = self.val_gen.classes
        predictions = self.model.predict(self.val_gen, steps=len(self.val_gen), verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        
        num_classes = len(self.labels)
        class_correct = [0] * num_classes
        class_total = [0] * num_classes
        
        for t, p in zip(y_true, y_pred):
            class_total[t] += 1
            if t == p:
                class_correct[t] += 1
                
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"--- Epoche {epoch + 1} ---\n")
            for i in range(num_classes):
                total = class_total[i]
                acc = (class_correct[i] / total * 100) if total > 0 else 0.0
                class_name = self.labels[i]
                f.write(f"Klasse '{class_name}': {acc:.2f}% Genauigkeit ({class_correct[i]}/{total})\n")
            f.write("\n")
        print(f"Klassenspezifische Ergebnisse an '{self.log_file}' angehängt.\n")

def check_data_before_start():
    if not os.path.exists(DATASET_PFAD):
        print(f"FEHLER: Der Ordner '{DATASET_NAME}' existiert nicht!")
        sys.exit(1)

def train():
    check_data_before_start()
    prepare_shuffled_dataset()

    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,      
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        brightness_range=[0.5, 1.5], # <-- HIER ANGEPASST FÜR HARTE SCHATTEN
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
        subset='validation',
        shuffle=False
    )

    if train_generator.samples == 0:
        print("ABBRUCH: Keine Bilder geladen.")
        return

    class_indices = train_generator.class_indices
    labels = {v: k for k, v in class_indices.items()}
    with open(LABELS_DATEI, 'w') as f:
        json.dump(labels, f)
    print(f"Klassen gefunden: {len(labels)}")

    # Berechne die Gewichte (seltene Klassen werden stärker gewichtet)
    classes = train_generator.classes
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(classes),
        y=classes
    )
    class_weight_dict = dict(enumerate(class_weights))

    # --- Standard-Callbacks (werden für beide Phasen verwendet) ---
    def erstelle_callbacks():
        return [
            EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
            ModelCheckpoint(MODEL_DATEI, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=1e-7, verbose=1),
            PerClassMetricsCallback(validation_generator)
        ]

    # --- MODELL LADEN ODER NEU BAUEN ---
    if os.path.exists(MODEL_DATEI):
        print(f"\n✅ Gespeichertes Modell '{MODEL_DATEI}' gefunden!")
        print("Lade bestehende Gewichte für Fine-Tuning (Phase 2)...")
        model = load_model(MODEL_DATEI)

        # Fine-Tuning: Obere Layer auftauen
        set_trainable = False
        trainable_count = 0
        frozen_count = 0
        for layer in model.layers:
            if layer.name == FINE_TUNE_FROM:
                set_trainable = True
            if set_trainable:
                layer.trainable = True
                trainable_count += 1
            else:
                layer.trainable = False
                frozen_count += 1

        print(f"🔓 {trainable_count} Layer aufgetaut, {frozen_count} Layer eingefroren (ab '{FINE_TUNE_FROM}')")

        model.compile(
            optimizer=Adam(learning_rate=FINE_TUNE_LR),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        print("\n🔧 Starte Phase 2: Fine-Tuning...")
        history = model.fit(
            train_generator,
            epochs=EPOCHS_PHASE2,
            validation_data=validation_generator,
            callbacks=erstelle_callbacks(),
            class_weight=class_weight_dict
        )

        print("Training abgeschlossen.")
        erstelle_trainings_bericht(history)
        speichere_trainings_graph(history)
        print(f"Das beste Modell liegt unter: {MODEL_DATEI}")

    else:
        print(f"\n⚠️ Kein gespeichertes Modell gefunden.")
        print(f"Erstelle NEUES InceptionV3 Modell mit Input Shape ({IMG_HEIGHT}, {IMG_WIDTH}, 3)...")

        base_model = InceptionV3(
            weights='imagenet',
            include_top=False,
            input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
        )

        # Phase 1: Basis-Modell komplett einfrieren
        base_model.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        predictions = Dense(len(labels), activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # ============================================================
        # PHASE 1: Nur Classifier-Head trainieren (Base eingefroren)
        # ============================================================
        print("\n🧊 Starte Phase 1: Nur Classifier-Head trainieren (Base eingefroren)...")
        history_phase1 = model.fit(
            train_generator,
            epochs=EPOCHS_PHASE1,
            validation_data=validation_generator,
            callbacks=erstelle_callbacks(),
            class_weight=class_weight_dict
        )

        phase1_val_acc = history_phase1.history['val_accuracy'][-1] * 100
        phase1_epochs_actual = len(history_phase1.history['accuracy'])
        print(f"\n📊 Phase 1 abgeschlossen! Validierungs-Genauigkeit: {phase1_val_acc:.2f}%")

        # ============================================================
        # PHASE 2: Fine-Tuning der oberen InceptionV3-Layer
        # ============================================================
        print(f"\n🔓 Starte Phase 2: Fine-Tuning (Layer ab '{FINE_TUNE_FROM}' werden aufgetaut)...")

        set_trainable = False
        trainable_count = 0
        frozen_count = 0
        for layer in base_model.layers:
            if layer.name == FINE_TUNE_FROM:
                set_trainable = True
            if set_trainable:
                layer.trainable = True
                trainable_count += 1
            else:
                layer.trainable = False
                frozen_count += 1

        print(f"   {trainable_count} Layer aufgetaut, {frozen_count} Layer eingefroren")

        # Neu kompilieren mit niedrigerer Lernrate!
        model.compile(
            optimizer=Adam(learning_rate=FINE_TUNE_LR),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        history_phase2 = model.fit(
            train_generator,
            epochs=EPOCHS_PHASE2,
            validation_data=validation_generator,
            callbacks=erstelle_callbacks(),
            class_weight=class_weight_dict
        )

        phase2_val_acc = history_phase2.history['val_accuracy'][-1] * 100
        print(f"\n📊 Phase 2 abgeschlossen! Validierungs-Genauigkeit: {phase2_val_acc:.2f}%")

        # Historien kombinieren für Bericht und Graph
        combined = {}
        for key in history_phase1.history.keys():
            combined[key] = history_phase1.history[key] + history_phase2.history[key]

        class CombinedHistory:
            def __init__(self, h):
                self.history = h

        history = CombinedHistory(combined)

        print("\nTraining abgeschlossen.")
        erstelle_trainings_bericht(history)
        speichere_trainings_graph(history, phase1_epochs=phase1_epochs_actual)
        print(f"Das beste Modell liegt unter: {MODEL_DATEI}")

if __name__ == "__main__":
    train()