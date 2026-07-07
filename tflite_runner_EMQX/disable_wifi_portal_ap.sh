#!/usr/bin/env bash
set -euo pipefail

AP_CON_NAME="${AP_CON_NAME:-gesture-ac-wifi-portal}"
NM_DNSMASQ_CONF="/etc/NetworkManager/dnsmasq-shared.d/gesture-ac-captive.conf"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Jalankan dengan sudo:"
  echo "sudo ./disable_wifi_portal_ap.sh"
  exit 1
fi

nmcli connection down "${AP_CON_NAME}" >/dev/null 2>&1 || true
nmcli connection delete "${AP_CON_NAME}" >/dev/null 2>&1 || true
rm -f "${NM_DNSMASQ_CONF}"
systemctl reload NetworkManager >/dev/null 2>&1 || systemctl restart NetworkManager

echo "[OK] Portal WiFi sementara dimatikan."
