import os
import glob
import json
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score




BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR        = os.path.join(BASE_DIR, "models_20")
KERAS_MODEL_PATH  = os.path.join(MODELS_DIR, "gesture_lstm_20frame.keras")
TFLITE_MODEL_PATH = os.path.join(MODELS_DIR, "gesture_lstm_20frame.tflite")
AUG_DATA_DIR      = os.path.join(BASE_DIR, "dataset_landmarks_aug_20")
RAW_DATA_DIR      = os.path.join(BASE_DIR, "dataset_landmarks_20")
DATA_DIR          = AUG_DATA_DIR if os.path.exists(AUG_DATA_DIR) else RAW_DATA_DIR
CLASSES_PATH      = os.path.join(MODELS_DIR, "classes.json")

SEQ_LEN           = 20
NUM_FEATS         = 67
SEED              = 42




def load_dataset_for_test(data_dir):
    X = []
    y = []


    if not os.path.exists(data_dir):
        print(f"[ERROR] {data_dir} tidak ditemukan.")
        return None, None, None

    all_folders = sorted([d for d in os.listdir(data_dir)
                          if os.path.isdir(os.path.join(data_dir, d))])
    classes = []

    print(f"[INFO] Memuat dataset untuk validasi TFLite...")
    print(f"[INFO] Mencari data dengan shape ({SEQ_LEN}, {NUM_FEATS})...")

    for folder_name in all_folders:
        cls_folder = os.path.join(data_dir, folder_name)
        files = glob.glob(os.path.join(cls_folder, "*.npz"))

        if len(files) == 0: continue

        current_label = len(classes)
        classes.append(folder_name)

        valid_count = 0
        for fpath in files:
            try:
                with np.load(fpath, allow_pickle=True) as data:
                    features = data['data']

                    if features.shape == (SEQ_LEN, NUM_FEATS):
                        X.append(features)
                        y.append(current_label)
                        valid_count += 1
            except Exception:
                pass


    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    return X, y, classes




def run_tflite_inference(tflite_path, X_test):
    print(f"[INFO] Menjalankan tes pada {len(X_test)} sampel menggunakan TFLite...")


    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]['index']
    output_index = output_details[0]['index']

    y_pred = []


    for i in range(len(X_test)):

        input_data = np.expand_dims(X_test[i], axis=0).astype(np.float32)

        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_index)[0]
        predicted_class = np.argmax(output_data)
        y_pred.append(predicted_class)

        if (i+1) % 500 == 0:
            print(f"   Processed {i+1}...")

    return np.array(y_pred)




def plot_cm(y_true, y_pred, classes, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title("TFLite Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label (TFLite)')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()




def main():

    X, y, classes = load_dataset_for_test(DATA_DIR)
    if X is None or len(X) == 0:
        print("[ERROR] Tidak ada data valid (Cek 67 fitur).")
        return


    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    print(f"[INFO] Jumlah Data Test: {len(X_test)} sampel")


    if not os.path.exists(KERAS_MODEL_PATH):
        print(f"[ERROR] File {KERAS_MODEL_PATH} tidak ditemukan.")
        return

    print("\n=== 1. KONVERSI KE TFLITE ===")
    model = tf.keras.models.load_model(KERAS_MODEL_PATH)


    converter = tf.lite.TFLiteConverter.from_keras_model(model)


    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]

    tflite_model = converter.convert()


    with open(TFLITE_MODEL_PATH, "wb") as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(TFLITE_MODEL_PATH) / 1024
    print(f"[SUCCESS] TFLite saved: {TFLITE_MODEL_PATH} ({size_kb:.2f} KB)")


    print("\n=== 2. TEST TFLITE PADA DATA TEST ===")

    y_pred_tflite = run_tflite_inference(TFLITE_MODEL_PATH, X_test)


    acc = accuracy_score(y_test, y_pred_tflite)
    print(f"\n[RESULT] TFLite Accuracy: {acc*100:.2f}%")


    print("\n--- Classification Report (TFLite) ---")
    report = classification_report(y_test, y_pred_tflite, target_names=classes)
    print(report)


    with open(os.path.join(MODELS_DIR, "tflite_report.txt"), "w") as f:
        f.write(f"TFLite Accuracy: {acc*100:.2f}%\n\n")
        f.write(report)


    plot_cm(y_test, y_pred_tflite, classes, os.path.join(MODELS_DIR, "tflite_confusion_matrix.png"))

    print("\n[DONE] Selesai. File TFLite siap digunakan untuk Realtime.")

if __name__ == "__main__":
    main()
