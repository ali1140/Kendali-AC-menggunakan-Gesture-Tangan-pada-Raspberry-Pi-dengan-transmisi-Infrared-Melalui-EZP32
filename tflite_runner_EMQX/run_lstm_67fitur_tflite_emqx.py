import json
import os
import ssl
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None

RUNNER_DIR = os.path.dirname(os.path.abspath(__file__))
EKSPERIMEN_DIR = os.path.dirname(RUNNER_DIR)
DEVICES_CONFIG_PATH = os.getenv("DEVICES_CONFIG_PATH", os.path.join(RUNNER_DIR, "devices.json"))
TFLITE_MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(EKSPERIMEN_DIR, "models_20", "gesture_lstm_20frame.tflite"),
)
CLASSES_PATH = os.getenv(
    "CLASSES_PATH",
    os.path.join(EKSPERIMEN_DIR, "models_20", "classes.json"),
)

SEQ_LEN = 20
NUM_FEATS = 67
CONF_THRESH = 0.90
SMOOTHING_LEN = 5
WRIST_IDX = 0
MIDDLE_MCP_IDX = 9
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))

MQTT_SERVER = os.getenv("MQTT_SERVER", "c18fc82c.ala.eu-central-1.emqxsl.com")
MQTT_PORT = int(os.getenv("MQTT_PORT", "8883"))
MQTT_USER = os.getenv("MQTT_USER", "")
MQTT_PASS = os.getenv("MQTT_PASS", "")
DEVICE_TOPIC_CODE = os.getenv("DEVICE_TOPIC_CODE", "ESP32-AC01")
ACTIVE_DEVICE_INDEX = int(os.getenv("ACTIVE_DEVICE_INDEX", "0"))
MQTT_TLS_INSECURE = os.getenv("MQTT_TLS_INSECURE", "0").lower() in {"1", "true", "yes"}
MQTT_QOS = int(os.getenv("MQTT_QOS", "1"))

COMMAND_COOLDOWN_SEC = float(os.getenv("COMMAND_COOLDOWN_SEC", "1.0"))
CONFIG_RELOAD_SEC = float(os.getenv("CONFIG_RELOAD_SEC", "2.0"))
DEVICE_SELECT_HOLD_SEC = float(os.getenv("DEVICE_SELECT_HOLD_SEC", "0.8"))
DEVICE_SELECT_COOLDOWN_SEC = float(os.getenv("DEVICE_SELECT_COOLDOWN_SEC", "1.5"))
NO_HAND_RESET_SEC = float(os.getenv("NO_HAND_RESET_SEC", "3.0"))
FINGER_TIP_IDS = [4, 8, 12, 16, 20]

GESTURE_TO_COMMAND = {
    "ThumbUp": ("POWER", 1, "Power ON"),
    "ThumbDown": ("POWER", 0, "Power OFF"),
    "Temp_up": ("TEMP_UP", 1, "Temp +1"),
    "Temp_down": ("TEMP_DOWN", 1, "Temp -1"),
    "Mode": ("MODE_TOGGLE", 1, "Mode Toggle"),
}

DEVICE_SELECT_PATTERNS = {
    1: [0, 1, 0, 0, 0],
    2: [0, 1, 1, 0, 0],
    3: [0, 1, 1, 1, 0],
    4: [0, 1, 1, 1, 1],
    5: [1, 1, 1, 1, 1],
}


def load_devices():
    if os.path.exists(DEVICES_CONFIG_PATH):
        try:
            with open(DEVICES_CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            devices = data.get("devices", data if isinstance(data, list) else [])
            parsed = []
            for item in devices:
                topic_code = str(item.get("topicCode", "")).strip()
                if not topic_code:
                    continue
                parsed.append({
                    "name": str(item.get("name", topic_code)).strip() or topic_code,
                    "topicCode": topic_code,
                    "defaultVendor": str(item.get("defaultVendor", "Midea")).strip() or "Midea",
                })
            return parsed
        except Exception as exc:
            print(f"[WARN] Gagal baca devices config: {exc}")

    return []


def get_devices_mtime():
    try:
        return os.path.getmtime(DEVICES_CONFIG_PATH)
    except OSError:
        return 0.0


def calculate_features(lm_curr, lm_prev, handedness):
    wrist_curr = lm_curr[WRIST_IDX].copy()
    mid_curr = lm_curr[MIDDLE_MCP_IDX].copy()
    palm_size = np.linalg.norm(mid_curr - wrist_curr) + 1e-6

    rel_pos = lm_curr - wrist_curr
    rel_pos = rel_pos / palm_size
    if handedness == "Left":
        rel_pos[:, 0] *= -1
    flat_pos = rel_pos.reshape(-1)

    if lm_prev is None:
        vector_feats = np.zeros(4, dtype=np.float32)
    else:
        wrist_prev = lm_prev[WRIST_IDX].copy()
        mid_prev = lm_prev[MIDDLE_MCP_IDX].copy()

        d_wrist = (wrist_curr[:2] - wrist_prev[:2]) / palm_size
        d_mid = (mid_curr[:2] - mid_prev[:2]) / palm_size

        if handedness == "Left":
            d_wrist[0] *= -1
            d_mid[0] *= -1

        vector_feats = np.concatenate([d_wrist, d_mid]) * 20.0

    return np.concatenate([flat_pos, vector_feats])


def detect_open_fingers(hand_landmarks, hand_label):
    landmarks = hand_landmarks.landmark
    fingers = []

    if hand_label == "Right":
        fingers.append(1 if landmarks[FINGER_TIP_IDS[0]].x < landmarks[FINGER_TIP_IDS[0] - 1].x else 0)
    else:
        fingers.append(1 if landmarks[FINGER_TIP_IDS[0]].x > landmarks[FINGER_TIP_IDS[0] - 1].x else 0)

    for idx in range(1, 5):
        tip_id = FINGER_TIP_IDS[idx]
        fingers.append(1 if landmarks[tip_id].y < landmarks[tip_id - 2].y else 0)

    return fingers, fingers.count(1)


def draw_info(img, text, x, y, color=(0, 255, 0), scale=1.0):
    cv2.putText(img, text, (x + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def draw_fps(img, fps):
    text = f"FPS: {fps:.1f}"
    (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    draw_info(img, text, img.shape[1] - text_w - 12, 30, (0, 255, 255), 0.7)


class EMQXACController:
    def __init__(self, devices, active_index=0):
        self.devices = devices
        self.active_index = max(0, min(active_index, len(devices) - 1)) if devices else 0
        self.last_status = "Pilih device" if devices else "Tidak ada device"
        self.last_command = "-"
        self.last_error = ""
        self.last_esp_status = "-"
        self.last_command_time = 0.0
        self.last_sent_label = None
        self.armed = True
        self.connected = False
        self.client = None
        self.device_selected = False

        if mqtt is None:
            self.last_status = "MQTT library kosong"
            self.last_error = "Install paho-mqtt"
            return

        self._connect_async()

    @property
    def active_device(self):
        if not self.devices:
            return None
        return self.devices[self.active_index]

    @property
    def active_name(self):
        device = self.active_device
        return device["name"] if device else "-"

    @property
    def active_topic_code(self):
        device = self.active_device
        return device["topicCode"] if device else ""

    @property
    def active_default_vendor(self):
        device = self.active_device
        return device.get("defaultVendor", "-") if device else "-"

    @property
    def control_topic(self):
        return f"{self.active_topic_code}/control" if self.active_topic_code else ""

    @property
    def status_topic(self):
        return f"{self.active_topic_code}/status" if self.active_topic_code else ""

    def _create_client(self):
        client_id = f"Gesture-TFLite-EMQX-{os.getpid()}-{int(time.time())}"
        if hasattr(mqtt, "CallbackAPIVersion"):
            return mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id=client_id)
        return mqtt.Client(client_id=client_id)

    def _connect_async(self):
        try:
            self.client = self._create_client()
            self.client.username_pw_set(MQTT_USER, MQTT_PASS)
            self.client.tls_set(cert_reqs=ssl.CERT_REQUIRED)
            self.client.tls_insecure_set(MQTT_TLS_INSECURE)
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.on_message = self._on_message
            self.client.connect_async(MQTT_SERVER, MQTT_PORT, keepalive=60)
            self.client.loop_start()
            self.last_status = "Menghubungkan"
        except Exception as exc:
            self.connected = False
            self.last_status = "MQTT error"
            self.last_error = str(exc)

    def _reason_code(self, rc):
        try:
            return int(rc)
        except Exception:
            return 0 if str(rc).lower() == "success" else -1

    def _on_connect(self, client, userdata, flags, rc):
        if self._reason_code(rc) == 0:
            self.connected = True
            self.last_status = "Terhubung" if self.devices else "Tidak ada device"
            self.last_error = ""
            if self.status_topic:
                client.subscribe(self.status_topic, qos=MQTT_QOS)
        else:
            self.connected = False
            self.last_status = "MQTT gagal"
            self.last_error = f"Connect rc={rc}"

    def _on_disconnect(self, client, userdata, rc):
        self.connected = False
        self.last_status = "Terputus"
        if self._reason_code(rc) != 0:
            self.last_error = f"Disconnect rc={rc}"

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except Exception:
            self.last_esp_status = msg.payload.decode("utf-8", errors="ignore")[:64]
            return

        power = "ON" if int(payload.get("power", 0)) == 1 else "OFF"
        mode = payload.get("mode", "-")
        temp = payload.get("temp", "-")
        vendor = payload.get("vendor", "-")
        self.last_esp_status = f"{power} | {mode} | {temp}C | {vendor}"

    def _publish_command(self, cmd, val):
        if not self.devices:
            self.last_status = "Tidak ada device"
            return False

        if not self.device_selected:
            self.last_status = "Pilih device dulu"
            return False

        if self.client is None:
            self.last_status = "MQTT library kosong"
            return False

        if not self.connected:
            self.last_status = "Belum terhubung"
            return False

        payload = json.dumps({"cmd": cmd, "val": val})
        result = self.client.publish(self.control_topic, payload, qos=MQTT_QOS, retain=False)
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            self.last_status = "Publish gagal"
            self.last_error = f"Publish rc={result.rc}"
            return False

        return True

    def switch_device(self, index, mark_selected=True):
        if index < 0 or index >= len(self.devices):
            self.last_status = "Device tidak ada"
            return False

        old_status_topic = self.status_topic
        self.active_index = index
        self.device_selected = mark_selected
        self.last_command = "-"
        self.last_esp_status = "-"
        self.last_sent_label = None
        self.armed = True

        if self.client is not None and self.connected:
            try:
                if old_status_topic:
                    self.client.unsubscribe(old_status_topic)
                if self.status_topic:
                    self.client.subscribe(self.status_topic, qos=MQTT_QOS)
            except Exception as exc:
                self.last_error = str(exc)

        self.last_status = f"Device: {self.active_name}" if self.device_selected else "Pilih device"
        print(f"[INFO] Active device: {self.active_name} ({self.active_topic_code})")
        print(f"[INFO] Default vendor: {self.active_default_vendor}")
        print(f"[INFO] Topic control: {self.control_topic}")
        print(f"[INFO] Topic status : {self.status_topic}")
        return True

    def clear_device_selection(self):
        self.device_selected = False
        self.last_command = "-"
        self.last_esp_status = "-"
        self.last_sent_label = None
        self.armed = True
        self.last_status = "Pilih device" if self.devices else "Tidak ada device"

    def update_devices(self, devices):
        if not devices:
            old_status_topic = self.status_topic
            self.devices = []
            self.active_index = 0
            self.clear_device_selection()
            if self.client is not None and self.connected and old_status_topic:
                try:
                    self.client.unsubscribe(old_status_topic)
                except Exception as exc:
                    self.last_error = str(exc)
            print("[INFO] Device config kosong. Gesture command dinonaktifkan.")
            return True

        current_topic = self.active_topic_code
        was_selected = self.device_selected
        self.devices = devices
        new_index = next((idx for idx, item in enumerate(devices) if item["topicCode"] == current_topic), 0)
        return self.switch_device(new_index, mark_selected=was_selected)

    def update_state(self, label, now):
        if not self.devices:
            self.armed = True
            self.last_sent_label = None
            self.last_status = "Tidak ada device"
            return False

        command = GESTURE_TO_COMMAND.get(label)
        if command is None:
            self.armed = True
            if label in {"Negative", "Unclear", "No hand"}:
                self.last_sent_label = None
            return False

        if not self.armed and label == self.last_sent_label:
            return False

        if now - self.last_command_time < COMMAND_COOLDOWN_SEC:
            return False

        cmd, value, command_text = command
        if self._publish_command(cmd, value):
            self.last_status = "Terkirim"
            self.last_command = command_text
            self.last_error = ""
            self.last_command_time = now
            self.last_sent_label = label
            self.armed = False
            return True

        self.last_command_time = now
        self.last_sent_label = label
        self.armed = False
        return False

    def close(self):
        if self.client is not None:
            self.client.loop_stop()
            self.client.disconnect()


class FingerDeviceSelector:
    def __init__(self, hold_sec, cooldown_sec):
        self.hold_sec = hold_sec
        self.cooldown_sec = cooldown_sec
        self.candidate_index = None
        self.candidate_since = 0.0
        self.last_switch_time = 0.0
        self.last_finger_count = 0
        self.status_text = "Device select: -"
        self.is_selecting = False
        self.just_selected = False

    def _target_from_fingers(self, fingers):
        count = fingers.count(1)
        pattern = DEVICE_SELECT_PATTERNS.get(count)
        if pattern is None or fingers != pattern:
            return None
        return count - 1

    def reset(self):
        self.candidate_index = None
        self.candidate_since = 0.0
        self.last_finger_count = 0
        self.status_text = "Device select: -"
        self.is_selecting = False
        self.just_selected = False

    def update(self, fingers, controller, now):
        self.just_selected = False
        self.last_finger_count = fingers.count(1)
        target_index = self._target_from_fingers(fingers)

        if target_index is None or target_index >= len(controller.devices):
            self.candidate_index = None
            self.candidate_since = 0.0
            self.status_text = f"Fingers: {self.last_finger_count} | Device select: -"
            self.is_selecting = False
            return False

        self.is_selecting = True
        if self.candidate_index != target_index:
            self.candidate_index = target_index
            self.candidate_since = now

        held_for = now - self.candidate_since
        device_no = target_index + 1

        if target_index == controller.active_index and controller.device_selected:
            self.status_text = f"Fingers: {self.last_finger_count} | Device {device_no} aktif"
            return True

        if now - self.last_switch_time < self.cooldown_sec:
            self.status_text = f"Fingers: {self.last_finger_count} | Tunggu..."
            return True

        if held_for >= self.hold_sec:
            controller.switch_device(target_index)
            self.last_switch_time = now
            self.candidate_since = now
            self.just_selected = True
            self.status_text = f"Fingers: {self.last_finger_count} | Pilih device {device_no}"
            return True

        self.status_text = f"Fingers: {self.last_finger_count} | Tahan device {device_no}"
        return True


def draw_mqtt_status(img, controller):
    status_color = (0, 255, 255) if controller.last_status in {"Terkirim", "Terhubung"} else (200, 200, 200)
    if "error" in controller.last_status.lower() or "gagal" in controller.last_status.lower() or "kosong" in controller.last_status.lower():
        status_color = (0, 0, 255)

    draw_info(img, f"AC: {controller.last_command}", 10, 78, (255, 255, 0), 0.75)
    draw_info(img, f"EMQX: {controller.last_status}", 10, 106, status_color, 0.7)
    device_name = controller.active_name if controller.device_selected else "-"
    topic_code = controller.active_topic_code if controller.device_selected else "-"
    default_vendor = controller.active_default_vendor if controller.device_selected else "-"
    esp_status = controller.last_esp_status if controller.device_selected else "-"
    draw_info(img, f"Device: {device_name}", 10, 132, (255, 220, 120), 0.55)
    draw_info(img, f"Topic: {topic_code}", 10, 156, (180, 220, 255), 0.55)
    draw_info(img, f"Default: {default_vendor}", 10, 180, (255, 180, 180), 0.55)
    draw_info(img, f"ESP: {esp_status}", 10, 204, (200, 255, 200), 0.55)

    if controller.last_error:
        draw_info(img, controller.last_error[:42], 10, 228, (0, 0, 255), 0.55)


def main():
    with open(CLASSES_PATH, "r") as f:
        classes = json.load(f)
    devices = load_devices()
    devices_mtime = get_devices_mtime()
    last_config_check = time.monotonic()

    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    input_idx = interpreter.get_input_details()[0]["index"]
    output_idx = interpreter.get_output_details()[0]["index"]

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, model_complexity=0)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Kamera index {CAMERA_INDEX} tidak bisa dibuka. Cek /dev/video* dan HOST_VIDEO_DEVICE.")
        return

    sequence_buffer = deque(maxlen=SEQ_LEN)
    prediction_history = deque(maxlen=SMOOTHING_LEN)
    lm_prev = None
    prev_time = time.perf_counter()
    emqx = EMQXACController(devices, ACTIVE_DEVICE_INDEX)
    finger_selector = FingerDeviceSelector(DEVICE_SELECT_HOLD_SEC, DEVICE_SELECT_COOLDOWN_SEC)
    interaction_mode = "select"
    last_hand_seen = time.monotonic()
    mode_status = "Mode: PILIH DEVICE"

    print(f"[INFO] TFLite aktif | EMQX: {MQTT_SERVER}:{MQTT_PORT} | ESC keluar")
    print("[INFO] Tombol angka 1-9 untuk ganti active device")
    print("[INFO] Pose jari untuk pilih device: 1=telunjuk, 2=telunjuk+tengah, 3=telunjuk+tengah+manis, 4=empat jari, 5=lima jari")
    print(f"[INFO] Device terpilih aktif sampai tangan tidak terdeteksi {NO_HAND_RESET_SEC:.1f} detik")
    for idx, device in enumerate(devices, start=1):
        marker = "*" if emqx.device_selected and idx - 1 == emqx.active_index else " "
        print(f"[INFO] {marker} {idx}. {device['name']} ({device['topicCode']}) | {device.get('defaultVendor', '-')}")
    print(f"[INFO] Topic control: {emqx.control_topic}")
    print(f"[INFO] Topic status : {emqx.status_topic}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.resize(frame, (640, 480))
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                lm_obj = results.multi_hand_landmarks[0]
                hand_lbl = results.multi_handedness[0].classification[0].label
                mp.solutions.drawing_utils.draw_landmarks(frame, lm_obj, mp_hands.HAND_CONNECTIONS)
                lm_curr = np.array([[p.x, p.y, p.z] for p in lm_obj.landmark], dtype=np.float32)
                fingers, _ = detect_open_fingers(lm_obj, hand_lbl)
                now_mono = time.monotonic()
                last_hand_seen = now_mono

                if interaction_mode == "select":
                    finger_selector.update(fingers, emqx, now_mono)
                    if finger_selector.just_selected:
                        interaction_mode = "gesture"
                        sequence_buffer.clear()
                        prediction_history.clear()
                        lm_prev = None
                        mode_status = "Mode: GESTURE COMMAND"
                    else:
                        mode_status = "Mode: PILIH DEVICE"
                else:
                    finger_selector.status_text = "Device aktif | Menunggu gesture"
                    mode_status = "Mode: GESTURE COMMAND"

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
                    if interaction_mode == "gesture":
                        emqx.update_state(final_label, time.monotonic())

                    draw_info(frame, f"Pred: {final_label}", 10, 40, col, 1.2)
            else:
                if sequence_buffer:
                    sequence_buffer.clear()
                    prediction_history.clear()
                    lm_prev = None
                now_mono = time.monotonic()
                no_hand_for = now_mono - last_hand_seen
                if interaction_mode == "gesture" and no_hand_for < NO_HAND_RESET_SEC:
                    remaining = max(0.0, NO_HAND_RESET_SEC - no_hand_for)
                    finger_selector.status_text = f"No hand: reset {remaining:.1f}s"
                    mode_status = "Mode: GESTURE COMMAND"
                elif interaction_mode == "gesture":
                    interaction_mode = "select"
                    finger_selector.reset()
                    emqx.clear_device_selection()
                    emqx.update_state("No hand", now_mono)
                    mode_status = "Mode: PILIH DEVICE"
                else:
                    finger_selector.reset()
                    emqx.update_state("No hand", now_mono)
                    mode_status = "Mode: PILIH DEVICE"
                draw_info(frame, "No hand", 10, 40, (200, 200, 200), 1.0)

            draw_mqtt_status(frame, emqx)
            draw_info(frame, finger_selector.status_text, 10, 252, (255, 255, 255), 0.55)
            draw_info(frame, mode_status, 10, 276, (120, 255, 255), 0.55)

            now_monotonic = time.monotonic()
            if now_monotonic - last_config_check >= CONFIG_RELOAD_SEC:
                last_config_check = now_monotonic
                current_mtime = get_devices_mtime()
                if current_mtime != devices_mtime:
                    devices_mtime = current_mtime
                    emqx.update_devices(load_devices())

            now = time.perf_counter()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            draw_fps(frame, fps)

            cv2.imshow("Gesture TFLite EMQX", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if ord("1") <= key <= ord("9"):
                if emqx.switch_device(key - ord("1")):
                    interaction_mode = "gesture"
                    last_hand_seen = time.monotonic()
                    finger_selector.reset()
                    finger_selector.status_text = "Device aktif | Menunggu gesture"
                    sequence_buffer.clear()
                    prediction_history.clear()
                    lm_prev = None
    finally:
        emqx.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
