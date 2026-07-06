import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import json
import os
import time
from collections import deque

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH        = os.path.join(BASE_DIR, "models_20", "gesture_lstm_20frame.keras")
CLASSES_PATH      = os.path.join(BASE_DIR, "models_20", "classes.json")

SEQ_LEN           = 20
NUM_FEATS         = 67
CONF_THRESH       = 0.9
SMOOTHING_LEN     = 5
WRIST_IDX      = 0
MIDDLE_MCP_IDX = 9

def calculate_features(lm_curr, lm_prev, handedness):
    wrist_curr = lm_curr[WRIST_IDX].copy()
    mid_curr   = lm_curr[MIDDLE_MCP_IDX].copy()
    palm_size  = np.linalg.norm(mid_curr - wrist_curr) + 1e-6


    rel_pos = lm_curr - wrist_curr
    rel_pos = rel_pos / palm_size
    if handedness == 'Left': rel_pos[:, 0] *= -1
    flat_pos = rel_pos.reshape(-1)


    if lm_prev is None:
        vector_feats = np.zeros(4, dtype=np.float32)
    else:
        wrist_prev = lm_prev[WRIST_IDX].copy()
        mid_prev   = lm_prev[MIDDLE_MCP_IDX].copy()

        d_wrist = (wrist_curr[:2] - wrist_prev[:2]) / palm_size
        d_mid   = (mid_curr[:2]   - mid_prev[:2])   / palm_size

        if handedness == 'Left':
            d_wrist[0] *= -1
            d_mid[0]   *= -1


        vector_feats = np.concatenate([d_wrist, d_mid]) * 20.0

    return np.concatenate([flat_pos, vector_feats])

def draw_info(img, text, x, y, color=(0, 255, 0), scale=1.0):
    cv2.putText(img, text, (x+2, y+2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), 2)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)

def draw_fps(img, fps):
    text = f"FPS: {fps:.1f}"
    (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    draw_info(img, text, img.shape[1] - text_w - 12, 30, (0, 255, 255), 0.7)

def main():
    try:
        with open(CLASSES_PATH, "r") as f: classes = json.load(f)
    except:
        print("[ERROR] classes.json tidak ditemukan")
        return

    print("[INFO] Load model...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("[INFO] Model siap")
    except Exception as e:
        print(f"[ERROR] Gagal load model: {e}")
        return

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, model_complexity=0)

    cap = cv2.VideoCapture(2)
    sequence_buffer = deque(maxlen=SEQ_LEN)
    prediction_history = deque(maxlen=SMOOTHING_LEN)
    lm_prev = None
    prev_time = time.perf_counter()

    print("[INFO] Keras aktif | ESC keluar")

    while True:
        ok, frame = cap.read()
        if not ok: break

        frame = cv2.resize(frame, (640, 480))
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            lm_obj = results.multi_hand_landmarks[0]
            hand_lbl = results.multi_handedness[0].classification[0].label
            mp.solutions.drawing_utils.draw_landmarks(frame, lm_obj, mp_hands.HAND_CONNECTIONS)
            lm_curr = np.array([[p.x, p.y, p.z] for p in lm_obj.landmark], dtype=np.float32)

            feats = calculate_features(lm_curr, lm_prev, hand_lbl)
            sequence_buffer.append(feats)
            lm_prev = lm_curr

            if len(sequence_buffer) == SEQ_LEN:
                input_tensor = np.expand_dims(list(sequence_buffer), axis=0)
                prediction = model.predict(input_tensor, verbose=0)[0]

                prediction_history.append(prediction)
                avg_pred = np.mean(prediction_history, axis=0)
                idx_max = np.argmax(avg_pred)
                final_conf = avg_pred[idx_max]
                label_pred = classes[idx_max]
                final_label = label_pred if final_conf >= CONF_THRESH else "Unclear"
                col = (0, 255, 0) if final_label != "Unclear" else (0, 0, 255)

                draw_info(frame, f"Pred: {final_label}", 10, 40, col, 1.2)

        else:
            if len(sequence_buffer) > 0:
                sequence_buffer.clear()
                prediction_history.clear()
                lm_prev = None
            draw_info(frame, "No hand", 10, 40, (200, 200, 200), 1.0)

        now = time.perf_counter()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        draw_fps(frame, fps)

        cv2.imshow("Gesture Keras", frame)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
