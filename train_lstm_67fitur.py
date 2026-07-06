import os
import glob
import json
import numpy as np
import tensorflow as tf
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import layers, models, callbacks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUG_DIR = os.path.join(BASE_DIR, "dataset_landmarks_aug_20")
RAW_LANDMARK_DIR = os.path.join(BASE_DIR, "dataset_landmarks_20")
DATA_DIR   = AUG_DIR if os.path.exists(AUG_DIR) else RAW_LANDMARK_DIR
MODELS_DIR = os.path.join(BASE_DIR, "models_20")
SEQ_LEN    = 20
NUM_FEATS  = 67

EPOCHS     = 100
BATCH_SIZE = 16
VAL_SPLIT  = 0.2
LEARNING_RATE = 0.001

def audit_vectors(X, sample_idx=0):
    sample = X[sample_idx]
    vectors = sample[:, -4:]
    avg_mag = np.mean(np.abs(vectors))

    print(f"\n[AUDIT DATA] Cek Sampel ke-{sample_idx}...")
    print(f"   Shape: {sample.shape} (Wajib {SEQ_LEN}, {NUM_FEATS})")
    print(f"   Kekuatan Vektor Rata-rata: {avg_mag:.5f}")

    if avg_mag < 0.0001:
        print("\n[BAHAYA] ⚠️  FITUR VEKTOR KOSONG (0.0)!")
        print("Model tidak akan tahu arah gerak (Kiri/Kanan).")
        return False
    return True


def load_dataset():
    if not os.path.exists(DATA_DIR):
        print(f"[ERROR] Folder {DATA_DIR} tidak ditemukan!")
        return None, None, None

    all_folders = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])

    X_data = []
    y_data = []
    final_classes = []

    print(f"Scanning Dataset: {DATA_DIR}")

    for folder_name in all_folders:
        class_path = os.path.join(DATA_DIR, folder_name)
        files = glob.glob(os.path.join(class_path, "*.npz"))
        valid_samples_count = 0


        for fpath in files:
            try:
                with np.load(fpath, allow_pickle=True) as f:
                    data = f['data']
                    if data.shape == (SEQ_LEN, NUM_FEATS):
                        X_data.append(data)
                        y_data.append(len(final_classes))
                        valid_samples_count += 1
            except:
                pass

        if valid_samples_count > 0:
            final_classes.append(folder_name)
            print(f"   [OK] Kelas '{folder_name}': {valid_samples_count} sampel")
        else:
            print(f"   [SKIP] Kelas '{folder_name}' KOSONG/RUSAK (Diabaikan)")

    if len(final_classes) == 0:
        print("[FATAL] Tidak ada data valid sama sekali!")
        return None, None, None

    X = np.array(X_data, dtype=np.float32)
    y = np.array(y_data, dtype=np.int32)

    print("-" * 30)
    print(f"Total Data: {len(X)} sampel")
    print(f"Kelas Valid: {final_classes}")

    return X, y, final_classes



def build_model(num_classes):
    model = models.Sequential([
        layers.Input(shape=(SEQ_LEN, NUM_FEATS)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Prediksi AI')
    plt.ylabel('Label Asli')
    plt.title('Confusion Matrix (Evaluasi Arah & Gerak)')

    save_path = os.path.join(MODELS_DIR, "confusion_matrix.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Confusion Matrix tersimpan di: {save_path}")

def main():

    X, y, classes = load_dataset()
    if X is None or len(X) == 0: return
    if not audit_vectors(X): input("ENTER untuk paksa lanjut...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=VAL_SPLIT, stratify=y, random_state=42
    )


    model = build_model(len(classes))
    os.makedirs(MODELS_DIR, exist_ok=True)

    callbacks_list = [
        callbacks.ModelCheckpoint(os.path.join(MODELS_DIR, "gesture_lstm_20frame.keras"), save_best_only=True, monitor='val_loss'),
        callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    ]

    print("\n>>> MULAI TRAINING...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks_list,
        verbose=1
    )


    with open(os.path.join(MODELS_DIR, "classes.json"), "w") as f:
        json.dump(classes, f)




    print("\n>>> EVALUASI MODEL...")


    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)


    plot_confusion_matrix(y_test, y_pred, classes)


    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    err_rate = 1.0 - acc

    print("\n=== METRIK EVALUASI KESELURUHAN ===")
    print(f"Akurasi (Accuracy)         : {acc*100:.2f}%")
    print(f"Presisi (Precision)        : {prec*100:.2f}%")
    print(f"Recall (Sensitivitas)      : {rec*100:.2f}%")
    print(f"F1-Score                   : {f1*100:.2f}%")
    print(f"Error Rate                 : {err_rate*100:.2f}%")

    print("\n=== CLASSIFICATION REPORT (Per Kelas) ===")
    print(classification_report(y_test, y_pred, target_names=classes))


    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1); plt.plot(history.history['accuracy']); plt.plot(history.history['val_accuracy']); plt.title('Akurasi')
    plt.subplot(1, 2, 2); plt.plot(history.history['loss']); plt.plot(history.history['val_loss']); plt.title('Loss')
    plt.savefig(os.path.join(MODELS_DIR, "training_plot.png"))

    print("\n[SELESAI] Semua hasil evaluasi ada di folder:", MODELS_DIR)

if __name__ == "__main__":
    main()
