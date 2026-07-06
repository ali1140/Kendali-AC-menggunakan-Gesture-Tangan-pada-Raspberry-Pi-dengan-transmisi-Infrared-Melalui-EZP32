# TFLite Runner

Folder ini hanya pembungkus Docker untuk menjalankan file original:

`/home/pi/Desktop/Ali/eksperimen/run_lstm_67fitur_tflite.py`

Kode Python original tidak diubah oleh runner ini.

## Struktur di Raspberry Pi

```text
/home/pi/Desktop/Ali/eksperimen/
  run_lstm_67fitur_tflite.py
  models_45/
  tflite_runner/
```

## Cara Menjalankan Window OpenCV di Desktop Raspberry Pi

1. Pastikan Raspberry Pi sedang login ke desktop.
2. Cek device kamera host:

```bash
ls /dev/video*
```

Jika kamera Anda bukan `/dev/video0`, set env `HOST_VIDEO_DEVICE` saat menjalankan compose.

3. Jalankan:

```bash
export DISPLAY=:0
xhost +SI:localuser:pi
cd /home/pi/Desktop/Ali/eksperimen/tflite_runner
sudo docker compose up -d --build
```

3. Lihat log:

```bash
sudo docker compose logs -f
```

Window OpenCV akan muncul di desktop Raspberry Pi, sehingga bisa ikut terlihat dari VNC jika Anda sedang remote desktop.

Contoh jika kamera host ada di `/dev/video2`:

```bash
export DISPLAY=:0
xhost +SI:localuser:pi
cd /home/pi/Desktop/Ali/eksperimen/tflite_runner
HOST_VIDEO_DEVICE=/dev/video2 sudo -E docker compose up -d --build
```

## Stop

```bash
cd /home/pi/Desktop/Ali/eksperimen/tflite_runner
sudo docker compose down
```
