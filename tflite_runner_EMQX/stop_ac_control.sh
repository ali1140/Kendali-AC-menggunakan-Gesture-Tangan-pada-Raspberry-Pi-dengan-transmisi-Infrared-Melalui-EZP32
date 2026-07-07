#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/home/pi/Desktop/Ali/eksperimen/tflite_runner_EMQX}"

cd "${APP_DIR}"
docker compose stop
docker compose ps
