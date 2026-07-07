import atexit
import json
import os
import queue
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, jsonify, render_template

try:
    import tensorflow as tf

    Interpreter = tf.lite.Interpreter
    INFERENCE_BACKEND = "tensorflow"
except ImportError:
    from tflite_runtime.interpreter import Interpreter
    INFERENCE_BACKEND = "tflite-runtime"


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(BASE_DIR.parent / "models_20" / "gesture_lstm_20frame.tflite")))
CLASSES_PATH = Path(os.getenv("CLASSES_PATH", str(BASE_DIR.parent / "models_20" / "classes.json")))

APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))
APP_DEBUG = os.getenv("APP_DEBUG", "0").lower() in {"1", "true", "yes", "on"}
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
DEFAULT_CAMERA_BACKEND = "auto" if os.name == "nt" else "v4l2"
CAMERA_BACKEND = os.getenv("CAMERA_BACKEND", DEFAULT_CAMERA_BACKEND).lower()
CAMERA_FOURCC = os.getenv("CAMERA_FOURCC", "MJPG")
CAPTURE_FPS = int(os.getenv("CAPTURE_FPS", "30"))
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "640"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "480"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "80"))
FRAME_POLL_MS = int(os.getenv("FRAME_POLL_MS", "40"))
STATUS_POLL_MS = int(os.getenv("STATUS_POLL_MS", "3000"))
TFLITE_THREADS = int(os.getenv("TFLITE_THREADS", str(max(1, min(4, os.cpu_count() or 1)))))
OPENCV_THREADS = int(os.getenv("OPENCV_THREADS", "1"))
TF_INTRA_THREADS = int(os.getenv("TF_INTRA_THREADS", "0"))
TF_INTER_THREADS = int(os.getenv("TF_INTER_THREADS", "0"))

SEQ_LEN = 20
NUM_FEATS = 67
CONF_THRESH = float(os.getenv("CONF_THRESH", "0.90"))
SMOOTHING_LEN = 5
WRIST_IDX = 0
MIDDLE_MCP_IDX = 9

BLYNK_ENABLED = os.getenv("BLYNK_ENABLED", "0").lower() in {"1", "true", "yes", "on"}
BLYNK_TEMPLATE_ID = "TMPL6QqBUQLFN"
BLYNK_TEMPLATE_NAME = "Remote AC kosan"
BLYNK_AUTH_TOKEN = os.getenv("BLYNK_AUTH_TOKEN", "-M8TAiPEyWX_im5nyzZUeoiV2ksid1CF")
BLYNK_BASE_URL = os.getenv("BLYNK_BASE_URL", "https://blynk.cloud/external/api")
BLYNK_TIMEOUT_SEC = float(os.getenv("BLYNK_TIMEOUT_SEC", "2.0"))
BLYNK_STATUS_SYNC_SEC = float(os.getenv("BLYNK_STATUS_SYNC_SEC", "15.0"))
COMMAND_COOLDOWN_SEC = float(os.getenv("COMMAND_COOLDOWN_SEC", "1.0"))
MOMENTARY_PRESS_SEC = float(os.getenv("MOMENTARY_PRESS_SEC", "0.12"))

VP_POWER = "V0"
VP_TEMP_UP = "V1"
VP_TEMP_DOWN = "V2"
VP_MODE_TOGGLE = "V3"
VP_LCD = "V4"
VP_TEMPDSP = "V5"
VP_MERK_AC = "V6"
VP_BTN_NEXT = "V7"
VP_BTN_PREV = "V8"

ACTION_MAP = {
    "power_on": ("power", 1, "Power ON"),
    "power_off": ("power", 0, "Power OFF"),
    "temp_up": ("pulse", VP_TEMP_UP, "Temp +1"),
    "temp_down": ("pulse", VP_TEMP_DOWN, "Temp -1"),
    "mode_toggle": ("pulse", VP_MODE_TOGGLE, "Mode Toggle"),
    "vendor_next": ("pulse", VP_BTN_NEXT, "Merk Next"),
    "vendor_prev": ("pulse", VP_BTN_PREV, "Merk Prev"),
}

GESTURE_TO_ACTION = {
    "ThumbUp": "power_on",
    "ThumbDown": "power_off",
    "Temp_up": "temp_up",
    "Temp_down": "temp_down",
    "Mode": "mode_toggle",
}

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = APP_DEBUG

try:
    cv2.setNumThreads(OPENCV_THREADS)
except Exception:
    pass

if INFERENCE_BACKEND == "tensorflow":
    try:
        if TF_INTRA_THREADS > 0:
            tf.config.threading.set_intra_op_parallelism_threads(TF_INTRA_THREADS)
        if TF_INTER_THREADS > 0:
            tf.config.threading.set_inter_op_parallelism_threads(TF_INTER_THREADS)
    except Exception:
        pass


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


def draw_info(img, text, x, y, color=(0, 255, 0), scale=1.0, thickness=2):
    cv2.putText(img, text, (x + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


class BlynkACController:
    def __init__(self, auth_token, enabled):
        self.auth_token = auth_token.strip()
        self.enabled = enabled
        self.lock = threading.Lock()
        self.last_status = "Idle" if enabled else "Off"
        self.last_command = "-"
        self.last_error = ""
        self.last_sent_label = None
        self.last_command_time = 0.0
        self.armed = True
        self.vendor_name = "-"
        self.temperature = "-"
        self.sync_stop_event = threading.Event()
        self.sync_now_event = threading.Event()
        self.sync_thread = None
        self.command_stop_event = threading.Event()
        self.command_queue = queue.Queue(maxsize=16)
        self.command_thread = None

    def _write_pin(self, pin, value):
        params = urllib.parse.urlencode({"token": self.auth_token, pin: value})
        url = f"{BLYNK_BASE_URL}/update?{params}"
        with urllib.request.urlopen(url, timeout=BLYNK_TIMEOUT_SEC) as response:
            response.read()

    def _read_pin(self, pin):
        params = urllib.parse.urlencode({"token": self.auth_token, pin: pin})
        url = f"{BLYNK_BASE_URL}/get?{params}"
        with urllib.request.urlopen(url, timeout=BLYNK_TIMEOUT_SEC) as response:
            return response.read().decode("utf-8").strip()

    def _pulse_pin(self, pin):
        self._write_pin(pin, 1)
        time.sleep(MOMENTARY_PRESS_SEC)
        self._write_pin(pin, 0)

    def _send_action_now(self, action_key):
        action = ACTION_MAP.get(action_key)
        if action is None:
            return False, "Aksi tidak dikenal"

        if not self.enabled:
            self.last_status = "Off"
            self.last_error = ""
            return False, "Blynk nonaktif"

        if not self.auth_token:
            self.last_status = "Token kosong"
            self.last_error = "Isi BLYNK_AUTH_TOKEN"
            return False, self.last_error

        action_type, target, command_text = action

        with self.lock:
            try:
                if action_type == "power":
                    self._write_pin(VP_POWER, target)
                else:
                    self._pulse_pin(target)

                self.last_status = "Terkirim"
                self.last_command = command_text
                self.last_error = ""
                self.last_command_time = time.monotonic()
                self.request_sync()
                return True, command_text
            except urllib.error.URLError as exc:
                self.last_status = "Blynk error"
                self.last_error = str(exc.reason)
            except Exception as exc:
                self.last_status = "Blynk error"
                self.last_error = str(exc)

        return False, self.last_error

    def queue_action(self, action_key):
        action = ACTION_MAP.get(action_key)
        if action is None:
            return False, "Aksi tidak dikenal"

        if not self.enabled:
            self.last_status = "Off"
            self.last_error = ""
            return False, "Blynk nonaktif"

        if not self.auth_token:
            self.last_status = "Token kosong"
            self.last_error = "BLYNK_AUTH_TOKEN belum diisi"
            return False, self.last_error

        command_text = action[2]

        try:
            self.command_queue.put_nowait(action_key)
            self.last_status = "Queued"
            self.last_command = command_text
            self.last_error = ""
            return True, command_text
        except queue.Full:
            self.last_status = "Queue penuh"
            self.last_error = "Perintah terlalu cepat"
            return False, self.last_error

    def update_state(self, label, now):
        if not self.enabled:
            self.armed = True
            if label in {"Negative", "Unclear", "No hand"}:
                self.last_sent_label = None
            return False

        action_key = GESTURE_TO_ACTION.get(label)
        if action_key is None:
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

        queued, _ = self.queue_action(action_key)
        if queued:
            self.last_command_time = now
            self.last_sent_label = label
            self.armed = False
        return queued

    def start(self):
        if not self.enabled:
            return
        if not self.auth_token:
            return
        if not (self.sync_thread and self.sync_thread.is_alive()):
            self.sync_stop_event.clear()
            self.sync_now_event.clear()
            self.sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
            self.sync_thread.start()
        if not (self.command_thread and self.command_thread.is_alive()):
            self.command_stop_event.clear()
            self.command_thread = threading.Thread(target=self._command_loop, daemon=True)
            self.command_thread.start()

    def stop(self):
        if not self.enabled:
            return
        self.sync_stop_event.set()
        self.sync_now_event.set()
        if self.sync_thread and self.sync_thread.is_alive():
            self.sync_thread.join(timeout=1.0)
        self.command_stop_event.set()
        if self.command_thread and self.command_thread.is_alive():
            self.command_thread.join(timeout=1.0)

    def request_sync(self):
        if not self.enabled:
            return
        self.sync_now_event.set()

    def _sync_loop(self):
        while not self.sync_stop_event.is_set():
            self.sync_status()
            self.sync_now_event.wait(BLYNK_STATUS_SYNC_SEC)
            self.sync_now_event.clear()

    def _command_loop(self):
        while not self.command_stop_event.is_set():
            try:
                action_key = self.command_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            self._send_action_now(action_key)

    def sync_status(self):
        if not self.enabled:
            return
        if not self.auth_token:
            return

        try:
            with self.lock:
                vendor = self._read_pin(VP_MERK_AC).strip('"')
                temp = self._read_pin(VP_TEMPDSP).strip('"')
            if vendor:
                self.vendor_name = vendor
            if temp:
                self.temperature = temp
        except Exception:
            pass

    def status_dict(self):
        return {
            "enabled": self.enabled,
            "last_status": self.last_status,
            "last_command": self.last_command,
            "last_error": self.last_error,
            "vendor_name": self.vendor_name,
            "temperature": self.temperature,
        }


class GestureWebApp:
    def __init__(self):
        self.lock = threading.Lock()
        self.raw_frame_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.capture_thread = None
        self.inference_thread = None
        self.cap = None
        self.latest_raw_frame = None
        self.latest_frame_id = 0
        self.latest_jpeg = self._make_placeholder("Menyiapkan kamera...")
        self.status = {
            "label": "Standby",
            "confidence": 0.0,
            "fps": 0.0,
            "buffer": 0,
            "model_enabled": True,
            "camera_index": CAMERA_INDEX,
            "backend": INFERENCE_BACKEND,
            "blynk_enabled": BLYNK_ENABLED,
            "blynk": {
                "last_status": "Idle",
                "last_command": "-",
                "last_error": "",
                "vendor_name": "-",
                "temperature": "-",
            },
            "error": "",
        }
        self.sequence_buffer = deque(maxlen=SEQ_LEN)
        self.prediction_history = deque(maxlen=SMOOTHING_LEN)
        self.lm_prev = None
        self.model_enabled = True
        self.blynk = BlynkACController(BLYNK_AUTH_TOKEN, BLYNK_ENABLED)
        self.blynk.start()
        self.startup_error = ""

        try:
            with open(CLASSES_PATH, "r", encoding="utf-8") as f:
                self.classes = json.load(f)
            try:
                self.interpreter = Interpreter(model_path=str(MODEL_PATH), num_threads=TFLITE_THREADS)
            except TypeError:
                self.interpreter = Interpreter(model_path=str(MODEL_PATH))
            self.interpreter.allocate_tensors()
            self.input_idx = self.interpreter.get_input_details()[0]["index"]
            self.output_idx = self.interpreter.get_output_details()[0]["index"]
            self.hands = mp.solutions.hands.Hands(max_num_hands=1, model_complexity=0)
            self.drawer = mp.solutions.drawing_utils
            self.hand_connections = mp.solutions.hands.HAND_CONNECTIONS
        except Exception as exc:
            self.startup_error = str(exc)
            self.classes = []
            self.interpreter = None
            self.input_idx = None
            self.output_idx = None
            self.hands = None
            self.drawer = None
            self.hand_connections = None

    def start(self):
        if not (self.capture_thread and self.capture_thread.is_alive()):
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
        if not (self.inference_thread and self.inference_thread.is_alive()):
            self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
            self.inference_thread.start()

    def stop(self):
        self.stop_event.set()
        if self.capture_thread:
            self.capture_thread.join(timeout=2)
        if self.inference_thread:
            self.inference_thread.join(timeout=2)
        self.blynk.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.hands is not None:
            self.hands.close()

    def _open_camera(self):
        backend = cv2.CAP_V4L2 if CAMERA_BACKEND == "v4l2" and hasattr(cv2, "CAP_V4L2") else None
        cap = cv2.VideoCapture(CAMERA_INDEX, backend) if backend is not None else cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened() and backend is not None:
            cap.release()
            cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if len(CAMERA_FOURCC) == 4:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CAMERA_FOURCC))
        cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        return cap if cap.isOpened() else None

    def _reset_sequence(self):
        self.sequence_buffer.clear()
        self.prediction_history.clear()
        self.lm_prev = None

    def set_model_enabled(self, enabled):
        enabled = bool(enabled)
        with self.lock:
            self.model_enabled = enabled
            self.status["model_enabled"] = enabled
            if not enabled:
                self.status["label"] = "Model off"
                self.status["confidence"] = 0.0
                self.status["buffer"] = 0
        self._reset_sequence()
        self.blynk.update_state("No hand", time.monotonic())

    def toggle_model(self):
        with self.lock:
            next_state = not self.model_enabled
        self.set_model_enabled(next_state)
        return next_state

    def _set_raw_frame(self, frame):
        with self.raw_frame_lock:
            self.latest_raw_frame = frame
            self.latest_frame_id += 1

    def _get_raw_frame_copy(self):
        with self.raw_frame_lock:
            if self.latest_raw_frame is None:
                return None, self.latest_frame_id
            return self.latest_raw_frame.copy(), self.latest_frame_id

    def _set_frame(self, frame):
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if ok:
            with self.lock:
                self.latest_jpeg = encoded.tobytes()

    def _make_placeholder(self, message, error_text=""):
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        frame[:] = (20, 24, 30)
        draw_info(frame, "Gesture AC Web", 22, 52, (0, 255, 255), 1.0)
        draw_info(frame, message, 22, 98, (255, 255, 255), 0.8, 1)
        if error_text:
            draw_info(frame, error_text[:54], 22, 138, (0, 0, 255), 0.65, 1)
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        return encoded.tobytes() if ok else b""

    def _update_status(self, **kwargs):
        with self.lock:
            self.status.update(kwargs)
            self.status["blynk"] = self.blynk.status_dict()

    def _draw_overlay(self, frame, label, confidence, fps):
        color = (0, 255, 0) if label not in {"Unclear", "No hand"} else (0, 0, 255)
        draw_info(frame, f"Pred: {label}", 10, 36, color, 0.95)
        draw_info(frame, f"Conf: {confidence * 100:.1f}%", 10, 66, (220, 220, 220), 0.68, 1)
        model_text = "Model: ON" if self.model_enabled else "Model: OFF"
        model_color = (0, 255, 255) if self.model_enabled else (180, 180, 180)
        draw_info(frame, model_text, 10, 96, model_color, 0.68, 1)
        if self.blynk.enabled:
            draw_info(frame, f"AC: {self.blynk.last_command}", 10, 126, (255, 255, 0), 0.68, 1)
            draw_info(frame, f"Merk: {self.blynk.vendor_name}", 10, 156, (160, 230, 255), 0.68, 1)
            blynk_color = (0, 255, 255) if self.blynk.last_status == "Terkirim" else (220, 220, 220)
            if "error" in self.blynk.last_status.lower() or "kosong" in self.blynk.last_status.lower():
                blynk_color = (0, 0, 255)
            draw_info(frame, f"Blynk: {self.blynk.last_status}", 10, 186, blynk_color, 0.68, 1)
            if self.blynk.last_error:
                draw_info(frame, self.blynk.last_error[:42], 10, 214, (0, 0, 255), 0.55, 1)
        fps_text = f"FPS: {fps:.1f}"
        (text_w, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        draw_info(frame, fps_text, frame.shape[1] - text_w - 12, 30, (0, 255, 255), 0.7)

    def get_frame(self):
        with self.lock:
            return self.latest_jpeg

    def _capture_loop(self):
        if self.startup_error:
                self._update_status(error=self.startup_error)
                self.latest_jpeg = self._make_placeholder("Gagal start model", self.startup_error)
                return

        while not self.stop_event.is_set():
            if self.cap is None or not self.cap.isOpened():
                self.cap = self._open_camera()
                if self.cap is None:
                    self._update_status(error="Kamera belum siap")
                    self.latest_jpeg = self._make_placeholder("Menunggu kamera...", "Periksa /dev/video0")
                    self.stop_event.wait(2.0)
                    continue

            ok, frame = self.cap.read()
            if not ok:
                self._update_status(error="Gagal membaca kamera")
                self.latest_jpeg = self._make_placeholder("Gagal baca frame", "Mencoba ulang kamera")
                self.cap.release()
                self.cap = None
                self.stop_event.wait(0.5)
                continue

            if frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT:
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            frame = cv2.flip(frame, 1)
            self._set_raw_frame(frame)

        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _inference_loop(self):
        prev_infer_time = time.perf_counter()
        last_frame_id = -1

        if self.startup_error:
            return

        while not self.stop_event.is_set():
            frame, frame_id = self._get_raw_frame_copy()
            if frame is None:
                self.stop_event.wait(0.01)
                continue

            if frame_id == last_frame_id:
                self.stop_event.wait(0.003)
                continue

            last_frame_id = frame_id

            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb)

                label = "Standby"
                confidence = 0.0
                with self.lock:
                    model_enabled = self.model_enabled

                if results.multi_hand_landmarks:
                    lm_obj = results.multi_hand_landmarks[0]
                    hand_lbl = "Right"
                    if results.multi_handedness and results.multi_handedness[0].classification:
                        hand_lbl = results.multi_handedness[0].classification[0].label
                    self.drawer.draw_landmarks(frame, lm_obj, self.hand_connections)
                    if model_enabled:
                        lm_curr = np.array([[p.x, p.y, p.z] for p in lm_obj.landmark], dtype=np.float32)

                        feats = calculate_features(lm_curr, self.lm_prev, hand_lbl)
                        self.sequence_buffer.append(feats)
                        self.lm_prev = lm_curr

                        if len(self.sequence_buffer) == SEQ_LEN:
                            input_tensor = np.expand_dims(list(self.sequence_buffer), axis=0).astype(np.float32)
                            self.interpreter.set_tensor(self.input_idx, input_tensor)
                            self.interpreter.invoke()
                            prediction = self.interpreter.get_tensor(self.output_idx)[0]

                            self.prediction_history.append(prediction)
                            avg_pred = np.mean(self.prediction_history, axis=0)
                            idx_max = int(np.argmax(avg_pred))
                            final_conf = float(avg_pred[idx_max])
                            label_pred = self.classes[idx_max]
                            label = label_pred if final_conf >= CONF_THRESH else "Unclear"
                            confidence = final_conf
                            self.blynk.update_state(label, time.monotonic())
                    else:
                        self._reset_sequence()
                        self.blynk.update_state("No hand", time.monotonic())
                        label = "Model off"
                else:
                    self._reset_sequence()
                    self.blynk.update_state("No hand", time.monotonic())
                    label = "No hand"

                now = time.perf_counter()
                fps = 1.0 / max(now - prev_infer_time, 1e-6)
                prev_infer_time = now

                self._draw_overlay(frame, label, confidence, fps)
                self._set_frame(frame)

                self._update_status(
                    label=label,
                    confidence=confidence,
                    fps=fps,
                    buffer=len(self.sequence_buffer),
                    error="",
                )
            except Exception as exc:
                self._reset_sequence()
                self.blynk.last_status = "Runtime error"
                self.blynk.last_error = str(exc)
                self._update_status(
                    label="Recovered",
                    confidence=0.0,
                    fps=0.0,
                    buffer=0,
                    error=str(exc),
                )
                self.stop_event.wait(0.2)

    def get_status(self):
        with self.lock:
            status = dict(self.status)
            status["classes"] = self.classes
            status["model_path"] = str(MODEL_PATH.name)
            status["template_name"] = BLYNK_TEMPLATE_NAME
            status["actions"] = (
                {
                    key: ACTION_MAP[key][2]
                    for key in ["power_on", "power_off", "temp_up", "temp_down", "mode_toggle", "vendor_prev", "vendor_next"]
                }
                if self.blynk.enabled
                else {}
            )
            return status

    def trigger_manual(self, action_key):
        return self.blynk.queue_action(action_key)


engine = None
engine_lock = threading.Lock()


def get_engine():
    global engine
    with engine_lock:
        if engine is None:
            engine = GestureWebApp()
            engine.start()
        return engine


def shutdown_engine():
    global engine
    with engine_lock:
        if engine is not None:
            engine.stop()
            engine = None


atexit.register(shutdown_engine)


@app.get("/")
def index():
    current_engine = get_engine()
    return render_template(
        "index.html",
        blynk_template_name=BLYNK_TEMPLATE_NAME,
        blynk_template_id=BLYNK_TEMPLATE_ID,
        blynk_enabled=BLYNK_ENABLED,
        actions=current_engine.get_status()["actions"],
        frame_poll_ms=FRAME_POLL_MS,
        status_poll_ms=STATUS_POLL_MS,
    )


@app.get("/api/frame")
def api_frame():
    frame = get_engine().get_frame()
    response = Response(frame, mimetype="image/jpeg")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.get("/api/status")
def api_status():
    return jsonify(get_engine().get_status())


@app.post("/api/manual/<action_key>")
def api_manual(action_key):
    current_engine = get_engine()
    sent, message = current_engine.trigger_manual(action_key)
    return jsonify({"ok": sent, "message": message, "status": current_engine.get_status()})


@app.route("/api/model/toggle", methods=["POST", "GET"])
def api_model_toggle():
    try:
        current_engine = get_engine()
        enabled = current_engine.toggle_model()
        return jsonify(
            {
                "ok": True,
                "model_enabled": enabled,
                "message": "Model aktif" if enabled else "Model dimatikan",
                "status": current_engine.get_status(),
            }
        )
    except Exception as exc:
        return jsonify({"ok": False, "message": str(exc)}), 500


@app.get("/health")
def health():
    return jsonify({"ok": True, "backend": INFERENCE_BACKEND})


if __name__ == "__main__":
    if (not APP_DEBUG) or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        get_engine()
    app.run(
        host=APP_HOST,
        port=APP_PORT,
        debug=APP_DEBUG,
        threaded=True,
        use_reloader=APP_DEBUG,
    )
