#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/home/pi/Desktop/Ali/eksperimen/tflite_runner_EMQX}"
STA_IFACE="${STA_IFACE:-wlan0}"
CHECK_HOST="${CHECK_HOST:-1.1.1.1}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-20}"
AP_CON_NAME="${AP_CON_NAME:-gesture-ac-wifi-portal}"
RECONNECT_WAIT_SEC="${RECONNECT_WAIT_SEC:-12}"

cd "${APP_DIR}"

ensure_web_config() {
  echo "[INFO] Memastikan web config aktif..."
  STA_IFACE="${STA_IFACE}" docker compose up -d emqx-config-web
}

internet_ok() {
  ping -I "${STA_IFACE}" -c 1 -W 3 "${CHECK_HOST}" >/dev/null 2>&1
}

wifi_connected() {
  local state
  state="$(nmcli -t -f DEVICE,TYPE,STATE dev status | awk -F: -v iface="${STA_IFACE}" '$1 == iface && $2 == "wifi" {print $3; exit}')"
  [[ "${state}" == "connected" ]]
}

active_wifi_name() {
  nmcli -t -f DEVICE,TYPE,STATE,CONNECTION dev status | awk -F: -v iface="${STA_IFACE}" '$1 == iface && $2 == "wifi" {print $4; exit}'
}

ap_is_active() {
  nmcli -t -f NAME connection show --active | grep -qx "${AP_CON_NAME}"
}

try_saved_wifi() {
  echo "[INFO] Mencoba reconnect ke WiFi tersimpan..."
  nmcli radio wifi on >/dev/null 2>&1 || true
  nmcli device disconnect "${STA_IFACE}" >/dev/null 2>&1 || true
  sleep 2
  nmcli device connect "${STA_IFACE}" >/dev/null 2>&1 || true

  for _ in $(seq 1 "${RECONNECT_WAIT_SEC}"); do
    if wifi_connected; then
      echo "[OK] Terhubung ke WiFi tersimpan: $(active_wifi_name)"
      return 0
    fi
    sleep 1
  done

  echo "[WARN] Tidak berhasil reconnect ke WiFi tersimpan."
  return 1
}

ensure_ap_setup() {
  if ap_is_active; then
    echo "[INFO] Portal AP sudah aktif."
    return
  fi

  echo "[WARN] Internet/WiFi belum aktif lewat ${STA_IFACE}. Menyalakan portal WiFi setup..."
  AP_IFACE="${STA_IFACE}" AP_CON_NAME="${AP_CON_NAME}" ./setup_wifi_portal_ap.sh
}

ensure_web_config

while true; do
  if wifi_connected; then
    if internet_ok; then
      echo "[OK] WiFi aktif ($(active_wifi_name)) dan internet bisa ping. Monitoring ulang ${CHECK_INTERVAL_SEC}s."
    else
      echo "[WARN] WiFi aktif ($(active_wifi_name)), tetapi ping internet gagal. Koneksi tidak diganggu."
    fi
  else
    ensure_web_config
    if try_saved_wifi; then
      echo "[OK] Reconnect WiFi berhasil. AP setup tidak dinyalakan."
    else
      ensure_ap_setup
    fi
  fi

  sleep "${CHECK_INTERVAL_SEC}"
done
