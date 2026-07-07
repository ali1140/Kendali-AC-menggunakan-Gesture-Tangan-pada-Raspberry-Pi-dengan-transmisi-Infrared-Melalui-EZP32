# Gesture AC Web for Raspberry Pi

Web ini menampilkan live inference gesture dari kamera Raspberry Pi, lalu mengirim command AC ke Blynk.

## Isi folder

- `app.py`: server Flask + pipeline inference + integrasi Blynk.
- `templates/index.html`: halaman web lokal.
- `static/style.css`: tampilan dashboard.
- `Dockerfile`: image untuk Raspberry Pi.
- `docker-compose.yml`: cara jalan paling cepat.

## Fitur

- Stream video MJPEG lokal.
- Status prediksi, confidence, FPS, dan command AC.
- Tombol manual untuk tes Blynk tanpa gesture.
- Mapping gesture:
  - `ThumbUp` -> Power ON
  - `ThumbDown` -> Power OFF
  - `Temp_up` -> suhu naik
  - `Temp_down` -> suhu turun

## Jalankan di Raspberry Pi

1. Salin folder ini ke Raspberry Pi.
2. Pastikan model tersedia di `../models_45`.
3. Jalankan:

```bash
docker compose up --build
```

4. Buka dari perangkat satu jaringan:

```text
http://IP_RASPBERRY_PI:8000
```

## Catatan penting

- Setup ini diasumsikan untuk Raspberry Pi OS 64-bit.
- Compose saat ini memetakan `/dev/video0`.
- Jika kamera Anda beda index, ubah `devices` atau `CAMERA_INDEX`.
- Token Blynk bisa diganti lewat environment `BLYNK_AUTH_TOKEN`.
- Stack ini dipin ke `mediapipe==0.10.5` dan `tensorflow-aarch64==2.14.0` agar model LSTM TFLite dengan `SELECT_TF_OPS` bisa berjalan di ARM64 Raspberry Pi.
