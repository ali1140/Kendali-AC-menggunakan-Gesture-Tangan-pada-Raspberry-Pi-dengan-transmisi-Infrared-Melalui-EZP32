#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/home/pi/Desktop/Ali/eksperimen/tflite_runner_EMQX}"
DISPLAY="${DISPLAY:-:0}"
XAUTHORITY="${XAUTHORITY:-/home/pi/.Xauthority}"
CONFIG_WEB_PORT="${CONFIG_WEB_PORT:-80}"
STA_IFACE="${STA_IFACE:-wlan0}"
HOST_VIDEO_DEVICE="${HOST_VIDEO_DEVICE:-/dev/video0}"

cd "${APP_DIR}"

export DISPLAY
export XAUTHORITY
export CONFIG_WEB_PORT
export STA_IFACE
export HOST_VIDEO_DEVICE

echo "[INFO] Menyiapkan akses display untuk window deteksi..."
if command -v xhost >/dev/null 2>&1; then
  xhost +SI:localuser:pi >/dev/null 2>&1 || true
  xhost +local:root >/dev/null 2>&1 || true
else
  echo "[WARN] xhost tidak ditemukan. Jika window deteksi gagal, install x11-xserver-utils."
fi

echo "[INFO] Menunggu Docker siap..."
for _ in $(seq 1 30); do
  if docker info >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

if ! docker info >/dev/null 2>&1; then
  echo "[ERROR] Docker belum siap."
  exit 1
fi

echo "[INFO] Menjalankan Gesture AC control stack..."
if [[ ! -e "${HOST_VIDEO_DEVICE}" ]]; then
  echo "[WARN] Kamera ${HOST_VIDEO_DEVICE} belum ditemukan. Container tetap dijalankan dan akan restart otomatis."
fi

docker compose up -d

echo "[OK] Service aktif:"
docker compose ps
