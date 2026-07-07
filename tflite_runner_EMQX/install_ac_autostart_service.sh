#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/home/pi/Desktop/Ali/eksperimen/tflite_runner_EMQX}"
SERVICE_PATH="/etc/systemd/system/gesture-ac-control.service"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Jalankan dengan sudo:"
  echo "sudo ./install_ac_autostart_service.sh"
  exit 1
fi

chmod +x "${APP_DIR}/start_ac_control.sh" "${APP_DIR}/stop_ac_control.sh"

cat >"${SERVICE_PATH}" <<EOF_SERVICE
[Unit]
Description=Gesture AC TFLite EMQX control stack
After=docker.service NetworkManager.service
Wants=docker.service NetworkManager.service

[Service]
Type=oneshot
WorkingDirectory=${APP_DIR}
Environment=APP_DIR=${APP_DIR}
Environment=DISPLAY=:0
Environment=XAUTHORITY=/home/pi/.Xauthority
Environment=CONFIG_WEB_PORT=80
Environment=STA_IFACE=wlan0
Environment=HOST_VIDEO_DEVICE=/dev/video0
ExecStart=${APP_DIR}/start_ac_control.sh
ExecStop=${APP_DIR}/stop_ac_control.sh
RemainAfterExit=yes
TimeoutStartSec=180

[Install]
WantedBy=multi-user.target
EOF_SERVICE

systemctl daemon-reload
systemctl enable gesture-ac-control.service

echo "[OK] Service gesture-ac-control terpasang."
echo "Build image sekali jika belum:"
echo "cd ${APP_DIR} && sudo docker compose build"
echo
echo "Tes sekarang:"
echo "sudo systemctl start gesture-ac-control.service"
echo "sudo systemctl status gesture-ac-control.service"
