import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import json
from collections import deque
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, "models_20", "gesture_lstm_20frame.keras")
CLASSES_PATH  = os.path.join(BASE_DIR, "models_20", "classes.json")

SEQ_LEN       = 20
CONF_THRESH   = 0.85
WRIST_IDX      = 0
MIDDLE_MCP_IDX = 9


DEBUG_MOV_THRESH = 2.0

def calculate_features(lm_curr, lm_prev, handedness):

    raw_wrist_curr = lm_curr[WRIST_IDX].copy()
    raw_mid_curr   = lm_curr[MIDDLE_MCP_IDX].copy()

    palm_size = np.linalg.norm(raw_mid_curr - raw_wrist_curr) + 1e-6


    if lm_prev is None:
        vector_feats = np.zeros(4, dtype=np.float32)
        debug_vec = (0, 0)
    else:
        raw_wrist_prev = lm_prev[WRIST_IDX].copy()
        raw_mid_prev   = lm_prev[MIDDLE_MCP_IDX].copy()





        d_wrist = (raw_wrist_curr[:2] - raw_wrist_prev[:2]) / palm_size
        d_mid   = (raw_mid_curr[:2]   - raw_mid_prev[:2])   / palm_size


        if handedness == 'Left':
            d_wrist[0] *= -1
            d_mid[0]   *= -1


        debug_vec = (d_wrist[0] * palm_size, d_wrist[1] * palm_size)


        vector_feats = np.concatenate([d_wrist, d_mid]) * 20.0


    rel_pos = lm_curr - raw_wrist_curr
    rel_pos = rel_pos / palm_size
    if handedness == 'Left': rel_pos[:, 0] *= -1
    flat_pos = rel_pos.reshape(-1)

    return np.concatenate([flat_pos, vector_feats]), debug_vec

def get_direction_text(dx, dy):
    mag = np.sqrt(dx**2 + dy**2)
    if mag < DEBUG_MOV_THRESH:
        return "Diam", (200, 200, 200)


    if abs(dx) > abs(dy):

        if dx > 0: return "Kanan", (0, 255, 0)
        else:      return "Kiri", (0, 0, 255)
    else:

        if dy > 0: return "Turun", (0, 255, 255)
        else:      return "Naik", (255, 0, 255)

def draw_fps(img, fps):
    text = f"FPS: {fps:.1f}"
    (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    x = img.shape[1] - text_w - 12
    cv2.putText(img, text, (x + 2, 30 + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, text, (x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

def main():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(CLASSES_PATH, "r") as f: classes = json.load(f)
        print(f"[INFO] Model siap: {classes}")
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, model_complexity=0)
    cap = cv2.VideoCapture(1)

    sequence_buffer = deque(maxlen=SEQ_LEN)
    lm_prev = None
    prev_time = time.perf_counter()

    print("[INFO] Debug arah | ESC keluar")

    while True:
        ok, frame = cap.read()
        if not ok: break


        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        final_label = "Standby"
        conf = 0.0
        debug_dx, debug_dy = 0, 0
        direction_txt = "-"
        dir_color = (100, 100, 100)

        if results.multi_hand_landmarks:
            lm_obj = results.multi_hand_landmarks[0]
            hand_lbl = results.multi_handedness[0].classification[0].label
            mp.solutions.drawing_utils.draw_landmarks(frame, lm_obj, mp_hands.HAND_CONNECTIONS)

            h, w, c = frame.shape
            lm_curr = np.array([[p.x * w, p.y * h, p.z] for p in lm_obj.landmark], dtype=np.float32)

            feats, (debug_dx, debug_dy) = calculate_features(lm_curr, lm_prev, hand_lbl)

            direction_txt, dir_color = get_direction_text(debug_dx, debug_dy)

            sequence_buffer.append(feats)
            lm_prev = lm_curr

            if len(sequence_buffer) == SEQ_LEN:
                input_tensor = np.expand_dims(list(sequence_buffer), axis=0)
                preds = model.predict(input_tensor, verbose=0)[0]
                idx = np.argmax(preds)
                conf = preds[idx]

                label = classes[idx]

                if conf > CONF_THRESH:
                    final_label = label
                else:
                    final_label = "Unclear"

        else:
            if len(sequence_buffer) > 0:
                sequence_buffer.clear()
                lm_prev = None
            final_label = "No hand"

        cv2.rectangle(frame, (0, 0), (280, 160), (0, 0, 0), -1)
        cv2.putText(frame, f"AI: {final_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Conf: {conf*100:.1f}%", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.line(frame, (10, 75), (270, 75), (100, 100, 100), 1)
        cv2.putText(frame, f"DIR: {direction_txt}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, dir_color, 2)
        cv2.putText(frame, f"dX: {debug_dx:.2f}", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"dY: {debug_dy:.2f}", (120, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        now = time.perf_counter()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        draw_fps(frame, fps)

        cv2.imshow("Debug Keras", frame)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
