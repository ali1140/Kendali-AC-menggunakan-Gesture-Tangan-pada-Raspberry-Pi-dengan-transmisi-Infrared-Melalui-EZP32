import json
import os
import shutil
import socket
import ssl
import subprocess
import time
import atexit
import threading
from urllib.parse import quote

from flask import jsonify, redirect, render_template_string, request, url_for
from flask import Flask

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None

try:
    from zeroconf import ServiceInfo, Zeroconf
except ImportError:
    ServiceInfo = None
    Zeroconf = None

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICES_CONFIG_PATH = os.getenv("DEVICES_CONFIG_PATH", os.path.join(APP_DIR, "devices.json"))

MQTT_SERVER = os.getenv("MQTT_SERVER", "c18fc82c.ala.eu-central-1.emqxsl.com")
MQTT_PORT = int(os.getenv("MQTT_PORT", "8883"))
MQTT_USER = os.getenv("MQTT_USER", "")
MQTT_PASS = os.getenv("MQTT_PASS", "")
MQTT_TLS_INSECURE = os.getenv("MQTT_TLS_INSECURE", "0").lower() in {"1", "true", "yes"}
MQTT_QOS = int(os.getenv("MQTT_QOS", "1"))
DISCOVERY_TOPIC = os.getenv("DISCOVERY_TOPIC", "gesture-ac/panel/discovery")
DISCOVERY_PUBLISH_SEC = float(os.getenv("DISCOVERY_PUBLISH_SEC", "30"))

VENDORS = ["Midea", "LG", "Sharp", "Daikin", "Panasonic"]
LOCAL_NAME = os.getenv("LOCAL_NAME", "gesture-ac")
ZEROCONF_NAME = os.getenv("ZEROCONF_NAME", "Gesture AC")
ZEROCONF_ENABLED = os.getenv("ZEROCONF_ENABLED", "1").lower() in {"1", "true", "yes"}
STA_IFACE = os.getenv("STA_IFACE", "wlan0")

app = Flask(__name__)
_zeroconf = None
_zeroconf_info = None
_discovery_thread_started = False


def local_ip_address():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("1.1.1.1", 80))
            return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def config_web_port():
    return int(os.getenv("CONFIG_WEB_PORT", "8000"))


def panel_url():
    ip_addr = local_ip_address()
    port = config_web_port()
    if port == 80:
        return f"http://{ip_addr}"
    return f"http://{ip_addr}:{port}"


def panel_discovery_payload():
    url = panel_url()
    ip_addr = url.replace("http://", "").split(":")[0].split("/")[0]
    return {
        "app": "gesture-ac-config",
        "name": ZEROCONF_NAME,
        "localName": LOCAL_NAME,
        "url": url,
        "ip": ip_addr,
        "port": config_web_port(),
        "version": 1,
        "ts": int(time.time()),
    }


def publish_panel_discovery():
    if mqtt is None:
        return False

    try:
        client = mqtt_client()
        client.loop_start()
        result = client.publish(
            DISCOVERY_TOPIC,
            json.dumps(panel_discovery_payload()),
            qos=MQTT_QOS,
            retain=True,
        )
        result.wait_for_publish(timeout=5)
        client.loop_stop()
        client.disconnect()
        print(f"[DISCOVERY] Published panel URL to {DISCOVERY_TOPIC}: {panel_url()}")
        return True
    except Exception as exc:
        print(f"[DISCOVERY] Gagal publish: {exc}")
        return False


def discovery_publish_loop():
    while True:
        publish_panel_discovery()
        time.sleep(DISCOVERY_PUBLISH_SEC)


def start_discovery_publisher():
    global _discovery_thread_started
    if _discovery_thread_started:
        return

    _discovery_thread_started = True
    thread = threading.Thread(target=discovery_publish_loop, daemon=True)
    thread.start()


def start_zeroconf_service():
    global _zeroconf, _zeroconf_info

    if not ZEROCONF_ENABLED or Zeroconf is None or ServiceInfo is None:
        return

    try:
        port = int(os.getenv("CONFIG_WEB_PORT", "8000"))
        if os.name != "nt" and port == 8000:
            port = 80

        ip_addr = local_ip_address()
        service_type = "_http._tcp.local."
        service_name = f"{ZEROCONF_NAME}.{service_type}"
        server_name = f"{LOCAL_NAME}.local."
        properties = {
            "path": "/",
            "app": "gesture-ac-config",
        }

        _zeroconf_info = ServiceInfo(
            service_type,
            service_name,
            addresses=[socket.inet_aton(ip_addr)],
            port=port,
            properties=properties,
            server=server_name,
        )
        _zeroconf = Zeroconf()
        _zeroconf.register_service(_zeroconf_info)
        print(f"[ZEROCONF] Advertised {service_name} at http://{LOCAL_NAME}.local:{port} ({ip_addr})")
    except Exception as exc:
        print(f"[ZEROCONF] Gagal start: {exc}")


def stop_zeroconf_service():
    global _zeroconf, _zeroconf_info

    if _zeroconf is None:
        return

    try:
        if _zeroconf_info is not None:
            _zeroconf.unregister_service(_zeroconf_info)
        _zeroconf.close()
    except Exception as exc:
        print(f"[ZEROCONF] Gagal stop: {exc}")


start_zeroconf_service()
atexit.register(stop_zeroconf_service)


def run_command(args, timeout=12):
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)


def nmcli_available():
    return shutil.which("nmcli") is not None


def internet_online():
    try:
        with socket.create_connection(("1.1.1.1", 53), timeout=2):
            return True
    except OSError:
        return False


def wifi_status():
    status = {
        "online": internet_online(),
        "nmcli": nmcli_available(),
        "iface": STA_IFACE,
        "connection": "-",
        "state": "unknown",
        "ip": request.host.split(":")[0] if request else "-",
        "networks": [],
        "error": "",
    }

    if not status["nmcli"]:
        status["error"] = "nmcli tidak tersedia. WiFi setup hanya aktif di Raspberry Pi/NetworkManager."
        return status

    try:
        result = run_command(["nmcli", "-t", "-f", "DEVICE,TYPE,STATE,CONNECTION", "dev"], timeout=5)
        for line in result.stdout.splitlines():
            parts = line.split(":")
            if len(parts) >= 4 and parts[0] == STA_IFACE:
                status["state"] = parts[2] or "-"
                status["connection"] = parts[3] or "-"
                break
    except Exception as exc:
        status["error"] = str(exc)

    status["networks"] = scan_wifi_networks()
    return status


def scan_wifi_networks():
    if not nmcli_available():
        return []

    try:
        run_command(["nmcli", "dev", "wifi", "rescan", "ifname", STA_IFACE], timeout=8)
        result = run_command(["nmcli", "-t", "-f", "SSID,SIGNAL,SECURITY", "dev", "wifi", "list", "ifname", STA_IFACE], timeout=8)
    except Exception:
        return []

    networks = []
    seen = set()
    for line in result.stdout.splitlines():
        parts = line.split(":")
        ssid = parts[0].strip() if parts else ""
        if not ssid or ssid in seen:
            continue
        seen.add(ssid)
        networks.append({
            "ssid": ssid,
            "signal": parts[1].strip() if len(parts) > 1 else "-",
            "security": parts[2].strip() if len(parts) > 2 else "-",
        })
    return networks[:12]


def read_devices():
    if not os.path.exists(DEVICES_CONFIG_PATH):
        return []

    with open(DEVICES_CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    devices = data.get("devices", data if isinstance(data, list) else [])
    cleaned = []
    for item in devices:
        topic_code = str(item.get("topicCode", "")).strip()
        if not topic_code:
            continue
        cleaned.append({
            "name": str(item.get("name", topic_code)).strip() or topic_code,
            "topicCode": topic_code,
            "defaultVendor": str(item.get("defaultVendor", "Midea")).strip() or "Midea",
        })
    return cleaned


def write_devices(devices):
    with open(DEVICES_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump({"devices": devices}, f, indent=2)


def mqtt_client():
    if mqtt is None:
        raise RuntimeError("paho-mqtt belum terinstall")

    client_id = f"raspi-config-web-{int(time.time())}"
    if hasattr(mqtt, "CallbackAPIVersion"):
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id=client_id)
    else:
        client = mqtt.Client(client_id=client_id)

    client.username_pw_set(MQTT_USER, MQTT_PASS)
    client.tls_set(cert_reqs=ssl.CERT_REQUIRED)
    client.tls_insecure_set(MQTT_TLS_INSECURE)
    client.connect(MQTT_SERVER, MQTT_PORT, keepalive=30)
    return client


def publish_command(device, payload):
    client = mqtt_client()
    topic = f"{device['topicCode']}/control"
    client.loop_start()
    result = client.publish(topic, json.dumps(payload), qos=MQTT_QOS, retain=False)
    result.wait_for_publish()
    client.loop_stop()
    client.disconnect()


def set_vendor_payload(device):
    vendor = device.get("defaultVendor", "Midea")
    return {
        "cmd": "SET_VENDOR",
        "vendor": vendor,
        "vendorIndex": VENDORS.index(vendor) if vendor in VENDORS else 0,
    }


PAGE = """
<!doctype html>
<html lang="id">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Gesture AC Setup</title>
  <style>
    :root { --bg:#f6f5f1; --card:#ffffff; --ink:#202428; --muted:#6c747c; --line:#ded8cc; --accent:#1f7a68; --danger:#b84a3a; }
    * { box-sizing:border-box; }
    body { margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; background:var(--bg); color:var(--ink); }
    main { width:min(820px,100%); margin:0 auto; padding:22px; }
    header { margin:8px 0 20px; }
    h1 { margin:0 0 6px; font-size:32px; letter-spacing:-0.04em; font-weight:850; }
    h2 { margin:0 0 12px; font-size:20px; }
    p { color:var(--muted); line-height:1.5; }
    a { color:var(--accent); font-weight:700; }
    .card { background:var(--card); border:1px solid var(--line); border-radius:18px; padding:18px; margin:14px 0; box-shadow:0 10px 28px rgba(20,24,28,.05); }
    .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:14px; }
    .status { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; }
    .pill { border:1px solid var(--line); border-radius:14px; padding:12px; background:#fbfaf7; }
    .pill span { display:block; color:var(--muted); font-size:12px; margin-bottom:4px; }
    .ok { color:#16744f; font-weight:800; }
    .bad { color:var(--danger); font-weight:800; }
    label { display:block; margin:12px 0 6px; color:var(--muted); font-size:13px; font-weight:750; }
    input, select { width:100%; border:1px solid var(--line); border-radius:12px; padding:11px 12px; font-size:15px; background:white; color:var(--ink); }
    button { border:0; border-radius:12px; padding:10px 13px; background:var(--accent); color:white; font-weight:800; cursor:pointer; }
    button.secondary { background:#45525d; }
    button.danger { background:var(--danger); }
    .actions { display:flex; flex-wrap:wrap; gap:8px; margin-top:14px; }
    .notice { margin:12px 0; padding:11px 13px; border-radius:14px; background:#fff7d7; border:1px solid #ead38d; color:#6f5508; }
    .topic { font-family:ui-monospace,SFMono-Regular,Consolas,monospace; color:#315e54; overflow-wrap:anywhere; font-size:13px; }
    .muted { color:var(--muted); font-size:14px; }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Gesture AC Setup</h1>
      <p>Halaman ini hanya untuk mengganti WiFi Raspberry Pi dan mengatur default merk AC tiap device.</p>
      <p class="muted">Akses cepat: <a href="http://{{ local_name }}.local">{{ local_name }}.local</a> | <a href="{{ url_for('access_help') }}">QR</a></p>
    </header>

    {% if message %}
      <div class="notice">{{ message }}</div>
    {% endif %}

    <section class="card">
      <h2>Status Jaringan</h2>
      <div class="status">
        <div class="pill"><span>Internet</span><b class="{{ 'ok' if wifi.online else 'bad' }}">{{ 'Online' if wifi.online else 'Offline' }}</b></div>
        <div class="pill"><span>Interface</span><b>{{ wifi.iface }}</b></div>
        <div class="pill"><span>WiFi</span><b>{{ wifi.connection }}</b></div>
        <div class="pill"><span>Alamat</span><b>{{ wifi.ip }}</b></div>
      </div>
      {% if wifi.error %}
        <div class="notice">{{ wifi.error }}</div>
      {% endif %}
    </section>

    <section class="card">
      <h2>Ganti WiFi</h2>
      <p class="muted">Jika halaman ini dibuka dari <b>GestureAC-Setup</b>, koneksi akan terputus setelah Raspberry Pi berhasil pindah ke WiFi target.</p>
      <form method="post" action="{{ url_for('connect_wifi') }}">
        <label>Pilih SSID</label>
        <select name="ssid_select" onchange="document.getElementById('ssid-input').value=this.value">
          <option value="">-- pilih jaringan --</option>
          {% for net in wifi.networks %}
            <option value="{{ net.ssid }}">{{ net.ssid }} | {{ net.signal }}% | {{ net.security }}</option>
          {% endfor %}
        </select>
        <label>SSID</label>
        <input id="ssid-input" name="ssid" placeholder="Nama WiFi" required>
        <label>Password</label>
        <input name="password" type="password" placeholder="Password WiFi">
        <div class="actions">
          <button type="submit">Connect WiFi</button>
          <a href="{{ url_for('index') }}"><button type="button" class="secondary">Refresh</button></a>
        </div>
      </form>
    </section>

    <section class="grid">
      {% for device in devices %}
      <article class="card">
        <h2>{{ loop.index }}. {{ device.name }}</h2>
        <p class="topic">{{ device.topicCode }}</p>
        <form method="post" action="{{ url_for('save_device') }}">
          <input type="hidden" name="index" value="{{ loop.index0 }}">
          <label>Nama Device</label>
          <input name="name" value="{{ device.name }}" required>
          <label>Topic Code</label>
          <input name="topicCode" value="{{ device.topicCode }}" required>
          <label>Default Merk AC</label>
          <select name="defaultVendor">
            {% for vendor in vendors %}
              <option value="{{ vendor }}" {% if vendor == device.defaultVendor %}selected{% endif %}>{{ vendor }}</option>
            {% endfor %}
          </select>
          <div class="actions">
            <button type="submit">Simpan</button>
            <button class="secondary" formaction="{{ url_for('apply_vendor', index=loop.index0) }}">Simpan + Kirim Merk</button>
            <button class="danger" formaction="{{ url_for('delete_device', index=loop.index0) }}">Hapus</button>
          </div>
        </form>
      </article>
      {% endfor %}

      <form class="card" method="post" action="{{ url_for('save_device') }}">
        <h2>Tambah Device</h2>
        <label>Nama Device</label>
        <input name="name" placeholder="AC Kamar" required>
        <label>Topic Code</label>
        <input name="topicCode" placeholder="ESP32-AC03" required>
        <label>Default Merk AC</label>
        <select name="defaultVendor">
          {% for vendor in vendors %}
            <option value="{{ vendor }}">{{ vendor }}</option>
          {% endfor %}
        </select>
        <div class="actions">
          <button type="submit">Tambah</button>
        </div>
      </form>
    </section>
  </main>
</body>
</html>
"""

def render_index(message=None):
    return render_template_string(
        PAGE,
        devices=read_devices(),
        vendors=VENDORS,
        wifi=wifi_status(),
        message=message or request.args.get("message", ""),
        local_name=LOCAL_NAME,
    )


def redirect_with_message(message):
    return redirect(url_for("index", message=message))


@app.get("/")
def index():
    return render_index()


@app.get("/api/discovery")
def api_discovery():
    payload = panel_discovery_payload()
    payload["url"] = request.host_url.rstrip("/")
    payload["ip"] = request.host.split(":")[0]
    return jsonify(payload)


@app.get("/generate_204")
@app.get("/gen_204")
@app.get("/hotspot-detect.html")
@app.get("/ncsi.txt")
@app.get("/connecttest.txt")
@app.get("/canonical.html")
@app.get("/success.txt")
def captive_check():
    return render_index()


@app.get("/akses")
@app.get("/qr")
def access_help():
    host_url = request.host_url.rstrip("/")
    mdns_url = f"http://{LOCAL_NAME}.local"
    host_url_qr = quote(host_url, safe="")
    mdns_url_qr = quote(mdns_url, safe="")
    return f"""
<!doctype html>
<html lang="id">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Akses Gesture AC</title>
  <style>
    body {{ margin:0; font-family:system-ui,sans-serif; background:#f4f1e8; color:#14212a; }}
    main {{ width:min(780px,100%); margin:auto; padding:24px; }}
    .card {{ background:white; border-radius:24px; padding:20px; margin:16px 0; box-shadow:0 16px 44px rgba(0,0,0,.08); }}
    img {{ width:220px; height:220px; image-rendering:pixelated; }}
    a {{ color:#0f8f7f; font-weight:800; overflow-wrap:anywhere; }}
  </style>
</head>
<body>
  <main>
    <h1>Akses Gesture AC</h1>
    <div class="card">
      <h2>Alamat IP yang sedang dipakai</h2>
      <p><a href="{host_url}">{host_url}</a></p>
      <img alt="QR IP" src="https://api.qrserver.com/v1/create-qr-code/?size=220x220&data={host_url_qr}">
    </div>
    <div class="card">
      <h2>Nama lokal</h2>
      <p><a href="{mdns_url}">{mdns_url}</a></p>
      <img alt="QR Local" src="https://api.qrserver.com/v1/create-qr-code/?size=220x220&data={mdns_url_qr}">
      <p>Jika nama lokal tidak bisa di HP, gunakan QR/alamat IP di atas.</p>
    </div>
  </main>
</body>
</html>
"""


@app.get("/<path:_path>")
def captive_fallback(_path):
    return render_index()


@app.post("/devices")
def save_device():
    devices = read_devices()
    name = request.form.get("name", "").strip()
    topic_code = request.form.get("topicCode", "").strip()
    default_vendor = request.form.get("defaultVendor", "Midea").strip()
    index_value = request.form.get("index", "").strip()

    if not name or not topic_code:
        return redirect(url_for("index"))

    item = {
        "name": name,
        "topicCode": topic_code,
        "defaultVendor": default_vendor if default_vendor in VENDORS else "Midea",
    }

    if index_value.isdigit() and int(index_value) < len(devices):
        devices[int(index_value)] = item
    else:
        devices.append(item)

    write_devices(devices)
    return redirect_with_message("Device disimpan.")


@app.post("/wifi/connect")
def connect_wifi():
    ssid = (request.form.get("ssid") or request.form.get("ssid_select") or "").strip()
    password = request.form.get("password", "")

    if not ssid:
        return redirect_with_message("SSID kosong.")

    if not nmcli_available():
        return redirect_with_message("nmcli tidak tersedia. Jalankan web config di Raspberry Pi dengan NetworkManager.")

    cmd = ["nmcli", "dev", "wifi", "connect", ssid, "ifname", STA_IFACE]
    if password:
        cmd.extend(["password", password])

    try:
        result = run_command(cmd, timeout=35)
        if result.returncode == 0:
            return redirect_with_message(f"Berhasil connect ke WiFi {ssid}.")
        error_text = (result.stderr or result.stdout or "Gagal connect").strip()
        return redirect_with_message(error_text[:180])
    except Exception as exc:
        return redirect_with_message(f"Gagal connect WiFi: {exc}")


@app.post("/devices/<int:index>/delete")
def delete_device(index):
    devices = read_devices()
    if 0 <= index < len(devices):
        devices.pop(index)
        write_devices(devices)
    return redirect_with_message("Device dihapus.")


@app.post("/devices/<int:index>/apply-vendor")
def apply_vendor(index):
    devices = read_devices()
    if 0 <= index < len(devices):
        try:
            publish_command(devices[index], set_vendor_payload(devices[index]))
            return redirect_with_message(f"Vendor default dikirim ke {devices[index]['name']}.")
        except Exception as exc:
            print(f"[WARN] Gagal kirim SET_VENDOR: {exc}")
            return redirect_with_message(f"Gagal kirim vendor: {exc}")
    return redirect_with_message("Device tidak ditemukan.")


if __name__ == "__main__":
    start_discovery_publisher()
    app.run(host="0.0.0.0", port=config_web_port())
