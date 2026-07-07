#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/home/pi/Desktop/Ali/eksperimen/tflite_runner_EMQX}"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Jalankan dengan sudo:"
  echo "sudo ./install_all_autostart.sh"
  exit 1
fi

cd "${APP_DIR}"

chmod +x ./*.sh

echo "[INFO] Build Docker image..."
docker compose build

echo "[INFO] Install WiFi bootstrap service..."
"${APP_DIR}/install_network_bootstrap_service.sh"

echo "[INFO] Install Gesture AC control service..."
"${APP_DIR}/install_ac_autostart_service.sh"

echo "[INFO] Menjalankan service sekarang..."
systemctl restart gesture-ac-network-bootstrap.service
systemctl restart gesture-ac-control.service

echo "[OK] Autostart selesai."
echo "Cek status:"
echo "systemctl status gesture-ac-network-bootstrap.service"
echo "systemctl status gesture-ac-control.service"
