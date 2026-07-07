#!/usr/bin/env bash
set -euo pipefail

AP_IFACE="${AP_IFACE:-wlan0}"
AP_SSID="${AP_SSID:-GestureAC-Setup}"
AP_PASS="${AP_PASS:-12345678}"
AP_IP="${AP_IP:-192.168.4.1}"
AP_CON_NAME="${AP_CON_NAME:-gesture-ac-wifi-portal}"
NM_DNSMASQ_DIR="/etc/NetworkManager/dnsmasq-shared.d"
NM_DNSMASQ_CONF="${NM_DNSMASQ_DIR}/gesture-ac-captive.conf"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Jalankan dengan sudo:"
  echo "sudo ./setup_wifi_portal_ap.sh"
  exit 1
fi

if [[ "${#AP_PASS}" -lt 8 ]]; then
  echo "AP_PASS minimal 8 karakter."
  exit 1
fi

if ! command -v nmcli >/dev/null 2>&1; then
  echo "NetworkManager/nmcli tidak ditemukan. Install dulu:"
  echo "sudo apt update && sudo apt install -y network-manager"
  exit 1
fi

if ! nmcli device status | awk '{print $1}' | grep -qx "${AP_IFACE}"; then
  echo "[ERROR] AP_IFACE ${AP_IFACE} tidak ditemukan."
  nmcli device status
  exit 1
fi

echo "[INFO] Membuat portal WiFi sementara di ${AP_IFACE}"
echo "[INFO] Saat user berhasil connect ke WiFi target, AP ini akan terputus."

mkdir -p "${NM_DNSMASQ_DIR}"
cat >"${NM_DNSMASQ_CONF}" <<EOF_DNSMASQ
address=/#/${AP_IP}
dhcp-option=114,http://${AP_IP}/
EOF_DNSMASQ

systemctl reload NetworkManager >/dev/null 2>&1 || systemctl restart NetworkManager

nmcli connection delete "${AP_CON_NAME}" >/dev/null 2>&1 || true

nmcli connection add \
  type wifi \
  ifname "${AP_IFACE}" \
  con-name "${AP_CON_NAME}" \
  autoconnect no \
  ssid "${AP_SSID}"

nmcli connection modify "${AP_CON_NAME}" \
  802-11-wireless.mode ap \
  802-11-wireless.band bg \
  802-11-wireless.channel 6 \
  ipv4.method shared \
  ipv4.addresses "${AP_IP}/24" \
  ipv6.method ignore \
  wifi-sec.key-mgmt wpa-psk \
  wifi-sec.psk "${AP_PASS}"

nmcli connection up "${AP_CON_NAME}"

echo "[OK] Portal WiFi aktif."
echo "[OK] SSID : ${AP_SSID}"
echo "[OK] PASS : ${AP_PASS}"
echo "[OK] Web  : http://${AP_IP}"
echo "[OK] Captive DNS wildcard aktif: ${NM_DNSMASQ_CONF}"
echo
echo "Jalankan web config jika belum aktif:"
echo "cd /home/pi/Desktop/Ali/eksperimen/tflite_runner_EMQX"
echo "STA_IFACE=${AP_IFACE} sudo -E docker compose up -d --build emqx-config-web"
