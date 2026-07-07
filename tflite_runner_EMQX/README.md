# TFLite Runner EMQX

Folder ini berisi program kontrol AC yang dipakai saat ini:

- `run_lstm_67fitur_tflite_emqx.py`: deteksi gesture TFLite dan kirim command AC lewat EMQX.
- `config_server.py`: web lokal untuk tambah device, ubah topic, ubah vendor default, kontrol AC manual, dan setting WiFi.
- `Remote_AC_EMQX_DefaultVendor.ino`: firmware ESP32 penerima command EMQX dan pengirim IR.
- `docker-compose.yml`: menjalankan runner gesture dan web config.
- `network_bootstrap.sh`: menyalakan portal WiFi setup jika Raspberry Pi tidak punya internet.

## Jalankan Manual

```bash
cd /home/pi/Desktop/Ali/eksperimen/tflite_runner_EMQX
sudo docker compose build
sudo docker compose up -d
```

Lihat log:

```bash
sudo docker compose logs -f gesture-tflite-emqx
sudo docker compose logs -f emqx-config-web
```

Stop:

```bash
sudo docker compose stop
```

## Autostart Saat Raspberry Pi Menyala

Jalankan sekali di Raspberry Pi:

```bash
cd /home/pi/Desktop/Ali/eksperimen/tflite_runner_EMQX
chmod +x ./*.sh
sudo ./install_all_autostart.sh
```

Script ini akan:

- Build image Docker.
- Install service `gesture-ac-network-bootstrap.service`.
- Install service `gesture-ac-control.service`.
- Menjalankan keduanya sekarang.
- Mengaktifkan keduanya otomatis saat boot berikutnya.

Cek status:

```bash
systemctl status gesture-ac-network-bootstrap.service
systemctl status gesture-ac-control.service
```

Lihat log boot:

```bash
journalctl -u gesture-ac-network-bootstrap.service -n 80 --no-pager
journalctl -u gesture-ac-control.service -n 80 --no-pager
```

## Alur Boot

Saat Raspberry Pi menyala:

```text
1. gesture-ac-network-bootstrap.service berjalan terus sebagai monitor jaringan.
2. Web config dinyalakan.
3. Jika internet lewat wlan0 aktif, AP setup tidak dinyalakan.
4. Jika internet tidak aktif, AP GestureAC-Setup dinyalakan.
5. gesture-ac-control.service menjalankan Docker Compose.
6. Runner gesture dan web config berjalan otomatis.
```

Runner deteksi gesture tetap dijalankan walaupun internet belum tersedia. Saat offline, status EMQX akan menunggu/reconnect, tetapi kamera, MediaPipe, dan TFLite tetap berjalan. Command AC baru bisa terkirim saat koneksi EMQX tersedia.

Sebelum menjalankan deteksi, `start_ac_control.sh` otomatis menjalankan:

```bash
export DISPLAY=:0
xhost +SI:localuser:pi
```

Ini diperlukan agar window OpenCV dari container bisa tampil di desktop Raspberry Pi.

Jika WiFi/internet hilang setelah boot, `network_bootstrap.sh` akan mengecek ulang berkala. Default interval cek adalah 20 detik.

Logika monitor:

```text
1. Jika wlan0 masih tersambung ke WiFi, koneksi tidak diganggu walaupun ping internet gagal.
2. Jika wlan0 tidak tersambung ke WiFi, Raspberry Pi mencoba reconnect ke jaringan tersimpan.
3. Jika reconnect gagal, AP GestureAC-Setup dinyalakan.
```

Ini mencegah kasus WiFi sudah tersambung tetapi diputus hanya karena internet lambat atau ping gagal sesaat.

Portal setup saat offline:

```text
SSID : GestureAC-Setup
PASS : 12345678
Web  : http://192.168.4.1
```

Jika HP/laptop terhubung ke AP tetapi halaman tidak otomatis terbuka, buka manual:

```text
http://192.168.4.1
```

Script `setup_wifi_portal_ap.sh` juga memasang DNS wildcard captive portal di:

```text
/etc/NetworkManager/dnsmasq-shared.d/gesture-ac-captive.conf
```

Tujuannya agar domain pengecekan captive portal Android/iOS diarahkan ke `192.168.4.1`.

Setelah user memasukkan SSID/password WiFi target dari web, AP setup akan putus dan Raspberry Pi berpindah ke WiFi target.

## Topic EMQX

Pola topic:

```text
<topicCode>/control
<topicCode>/status
```

Contoh:

```text
ESP32-AC01/control
ESP32-AC01/status
```

Mapping gesture:

```text
ThumbUp    -> POWER ON
ThumbDown  -> POWER OFF
Temp_up    -> TEMP_UP
Temp_down  -> TEMP_DOWN
Mode       -> MODE_TOGGLE
```

## Multi Device

Daftar device disimpan di `devices.json`.

Web config dan runner gesture memakai file yang sama. Jika `devices.json` berubah dari web, runner akan reload otomatis.

Contoh:

```json
{
  "devices": [
    {
      "name": "AC 1",
      "topicCode": "ESP32-AC01",
      "defaultVendor": "Midea"
    },
    {
      "name": "AC 2",
      "topicCode": "ESP32-AC02",
      "defaultVendor": "LG"
    }
  ]
}
```

## Web Config

Saat Raspberry Pi sudah terhubung ke WiFi, buka dari HP/laptop satu jaringan:

```text
http://IP_RASPI
http://gesture-ac.local
```

Jika `.local` tidak bisa di HP, gunakan IP Raspberry Pi:

```bash
hostname -I
```

Fitur web:

- Indikator internet.
- Input SSID/password WiFi.
- Tambah/edit/hapus device.
- Ubah vendor default.
- Publish discovery panel ke EMQX.

Web config sengaja tidak menyediakan kontrol AC penuh. Kontrol AC dilakukan lewat gesture dan aplikasi; web hanya untuk setup WiFi dan default merk.

Discovery panel dikirim retained ke:

```text
gesture-ac/panel/discovery
```

Aplikasi Flutter dapat mengambil URL panel dari topic ini tanpa scan IP manual.

## File Autostart

File yang dipakai:

- `start_ac_control.sh`: menjalankan stack Docker.
- `stop_ac_control.sh`: menghentikan stack Docker.
- `install_ac_autostart_service.sh`: install service kontrol AC.
- `install_all_autostart.sh`: install semua service yang diperlukan.
- `install_network_bootstrap_service.sh`: install service WiFi bootstrap.
- `network_bootstrap.sh`: cek internet dan nyalakan AP setup jika offline.
- `setup_wifi_portal_ap.sh`: membuat/menyalakan AP `GestureAC-Setup`.
- `disable_wifi_portal_ap.sh`: mematikan AP setup manual.

## Catatan

Raspberry Pi 5 umumnya hanya punya satu interface WiFi `wlan0`. Karena itu AP setup dan koneksi WiFi internet tidak berjalan bersamaan pada interface yang sama. AP setup dipakai untuk kondisi awal/offline; setelah user submit WiFi, `wlan0` pindah ke WiFi target.
