#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/home/pi/Desktop/Ali/eksperimen/tflite_runner_EMQX}"
STA_IFACE="${STA_IFACE:-wlan0}"
SERVICE_PATH="/etc/systemd/system/gesture-ac-network-bootstrap.service"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Jalankan dengan sudo:"
  echo "sudo ./install_network_bootstrap_service.sh"
  exit 1
fi

chmod +x "${APP_DIR}/network_bootstrap.sh" \
  "${APP_DIR}/setup_wifi_portal_ap.sh" \
  "${APP_DIR}/disable_wifi_portal_ap.sh"

cat >"${SERVICE_PATH}" <<EOF_SERVICE
[Unit]
Description=Gesture AC WiFi setup portal bootstrap
After=NetworkManager.service docker.service
Wants=NetworkManager.service docker.service

[Service]
Type=simple
WorkingDirectory=${APP_DIR}
Environment=APP_DIR=${APP_DIR}
Environment=STA_IFACE=${STA_IFACE}
Environment=CHECK_INTERVAL_SEC=20
Environment=RECONNECT_WAIT_SEC=12
ExecStart=${APP_DIR}/network_bootstrap.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF_SERVICE

systemctl daemon-reload
systemctl enable gesture-ac-network-bootstrap.service

echo "[OK] Service terpasang."
echo "Tes sekarang:"
echo "sudo systemctl start gesture-ac-network-bootstrap.service"
echo "sudo systemctl status gesture-ac-network-bootstrap.service"
