import os
import time

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import json
import urllib.error
import urllib.parse
import urllib.request
from collections import deque

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TFLITE_MODEL_PATH = os.path.join(BASE_DIR, "models_20", "gesture_lstm_20frame.tflite")
CLASSES_PATH      = os.path.join(BASE_DIR, "models_20", "classes.json")

SEQ_LEN           = 20
NUM_FEATS         = 67
CONF_THRESH       = 0.90
SMOOTHING_LEN     = 5
WRIST_IDX      = 0
MIDDLE_MCP_IDX = 9
BLYNK_TEMPLATE_ID = "TMPL6QqBUQLFN"
BLYNK_TEMPLATE_NAME = "Remote AC kosan"
# BLYNK_AUTH_TOKEN = os.getenv("BLYNK_AUTH_TOKEN", "YOUR_TOKEN_HERE")
BLYNK_AUTH_TOKEN = os.getenv("BLYNK_AUTH_TOKEN", "")
BLYNK_BASE_URL = os.getenv("BLYNK_BASE_URL", "https://blynk.cloud/external/api")
BLYNK_TIMEOUT_SEC = 2.0
COMMAND_COOLDOWN_SEC = 1.0
MOMENTARY_PRESS_SEC = 0.12
VP_POWER = "V0"
VP_TEMP_UP = "V1"
VP_TEMP_DOWN = "V2"
VP_MODE_TOGGLE = "V3"
VP_LCD = "V4"
VP_TEMPDSP = "V5"
VP_MERK_AC = "V6"
VP_BTN_NEXT = "V7"
VP_BTN_PREV = "V8"

GESTURE_TO_COMMAND = {
    "ThumbUp": ("power", 1, "Power ON"),
    "ThumbDown": ("power", 0, "Power OFF"),
    "Temp_up": ("pulse", VP_TEMP_UP, "Temp +1"),
    "Temp_down": ("pulse", VP_TEMP_DOWN, "Temp -1"),
    "Mode": ("pulse", VP_MODE_TOGGLE, "Mode Toggle"),
}

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

class BlynkACController:
    def __init__(self, auth_token):
        self.auth_token = auth_token.strip()
        self.last_status = "Idle"
        self.last_command = "-"
        self.last_error = ""
        self.last_command_time = 0.0
        self.last_sent_label = None
        self.armed = True

    def _write_pin(self, pin, value):
        params = urllib.parse.urlencode({"token": self.auth_token, pin: value})
        url = f"{BLYNK_BASE_URL}/update?{params}"
        with urllib.request.urlopen(url, timeout=BLYNK_TIMEOUT_SEC) as response:
            response.read()

    def _pulse_pin(self, pin):
        self._write_pin(pin, 1)
        time.sleep(MOMENTARY_PRESS_SEC)
        self._write_pin(pin, 0)

    def update_state(self, label, now):
        command = GESTURE_TO_COMMAND.get(label)
        if command is None:
            self.armed = True
            if label in {"Negative", "Unclear", "No hand"}:
                self.last_sent_label = None
            return False

        if not self.auth_token:
            self.last_status = "Token kosong"
            self.last_error = "BLYNK_AUTH_TOKEN belum diisi"
            return False

        if not self.armed and label == self.last_sent_label:
            return False

        if now - self.last_command_time < COMMAND_COOLDOWN_SEC:
            return False

        action, target, command_text = command

        try:
            if action == "power":
                self._write_pin(VP_POWER, target)
            else:
                self._pulse_pin(target)

            self.last_status = "Terkirim"
            self.last_command = command_text
            self.last_error = ""
            self.last_command_time = now
            self.last_sent_label = label
            self.armed = False
            return True
        except urllib.error.URLError as exc:
            self.last_status = "Blynk error"
            self.last_error = str(exc.reason)
        except Exception as exc:
            self.last_status = "Blynk error"
            self.last_error = str(exc)

        self.last_command_time = now
        self.last_sent_label = label
        self.armed = False
        return False

def draw_blynk_status(img, controller):
    status_color = (0, 255, 255) if controller.last_status == "Terkirim" else (200, 200, 200)
    if "error" in controller.last_status.lower() or "kosong" in controller.last_status.lower():
        status_color = (0, 0, 255)

    draw_info(img, f"AC: {controller.last_command}", 10, 78, (255, 255, 0), 0.75)
    draw_info(img, f"Blynk: {controller.last_status}", 10, 106, status_color, 0.7)

    if controller.last_error:
        short_error = controller.last_error[:42]
        draw_info(img, short_error, 10, 132, (0, 0, 255), 0.55)

def main():
    with open(CLASSES_PATH, "r") as f: classes = json.load(f)

    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    input_idx = interpreter.get_input_details()[0]['index']
    output_idx = interpreter.get_output_details()[0]['index']

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, model_complexity=0)

    cap = cv2.VideoCapture(0)
    sequence_buffer = deque(maxlen=SEQ_LEN)
    prediction_history = deque(maxlen=SMOOTHING_LEN)
    lm_prev = None
    prev_time = time.perf_counter()
    blynk = BlynkACController(BLYNK_AUTH_TOKEN)

    print(f"[INFO] TFLite aktif | Blynk: {BLYNK_TEMPLATE_NAME} | ESC keluar")

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
                input_tensor = np.expand_dims(list(sequence_buffer), axis=0).astype(np.float32)
                interpreter.set_tensor(input_idx, input_tensor)
                interpreter.invoke()
                prediction = interpreter.get_tensor(output_idx)[0]

                prediction_history.append(prediction)
                avg_pred = np.mean(prediction_history, axis=0)
                idx_max = np.argmax(avg_pred)
                final_conf = avg_pred[idx_max]
                label_pred = classes[idx_max]
                final_label = label_pred if final_conf >= CONF_THRESH else "Unclear"
                col = (0, 255, 0) if final_label != "Unclear" else (0, 0, 255)
                blynk.update_state(final_label, time.monotonic())

                draw_info(frame, f"Pred: {final_label}", 10, 40, col, 1.2)

        else:
            if len(sequence_buffer) > 0:
                sequence_buffer.clear()
                prediction_history.clear()
                lm_prev = None
            blynk.update_state("No hand", time.monotonic())
            draw_info(frame, "No hand", 10, 40, (200, 200, 200), 1.0)

        draw_blynk_status(frame, blynk)

        now = time.perf_counter()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        draw_fps(frame, fps)

        cv2.imshow("Gesture TFLite", frame)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
