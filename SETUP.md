# Raspberry Pi 4 — Poultry Disease Detection Setup Guide
# OS: Raspberry Pi OS Bookworm 64-bit (Debian 12)

## ─── STEP 1: Run this ONCE on your Windows/Mac machine ──────────────────────
## Export .pt models → .onnx (much faster on Pi's CPU)

# On Windows (PowerShell):
cd C:\Users\OVIYA\OneDrive\Desktop\Poultry_disease_Prediction
python convert_to_onnx.py --imgsz 416

# This creates:
#   models/general.onnx
#   models/fecal.onnx
#   models/appearance.onnx
#   models/behavior.onnx


## ─── STEP 2: Copy project to Raspberry Pi ───────────────────────────────────
## From Windows PowerShell (replace PI_IP with your Pi's IP):

rsync -av --exclude __pycache__ --exclude "*.pyc" \
  "C:/Users/OVIYA/OneDrive/Desktop/Poultry_disease_Prediction/" \
  pi@PI_IP:~/poultry/

# Or with SCP:
scp -r "C:\Users\OVIYA\OneDrive\Desktop\Poultry_disease_Prediction" pi@PI_IP:~/poultry


## ─── STEP 3: SSH into the Raspberry Pi ──────────────────────────────────────

ssh pi@PI_IP
cd ~/poultry


## ─── STEP 4: System dependencies (run once on Pi) ───────────────────────────

sudo apt update && sudo apt install -y \
    python3-venv \
    python3-pip \
    libopencv-dev \
    v4l-utils \
    libatlas-base-dev \
    libhdf5-dev


## ─── STEP 5: Verify USB camera is detected ──────────────────────────────────

v4l2-ctl --list-devices
# Should show: /dev/video0  (or video1 if built-in cam exists)

# Test camera capture:
v4l2-ctl --device=/dev/video0 --set-fmt-video=width=640,height=480,pixelformat=MJPG
v4l2-ctl --device=/dev/video0 --stream-mmap --stream-count=1


## ─── STEP 6: Create virtual environment ─────────────────────────────────────

python3 -m venv ~/poultry/venv --system-site-packages
source ~/poultry/venv/bin/activate


## ─── STEP 7: Install Python dependencies ────────────────────────────────────

pip install --upgrade pip wheel
pip install -r requirements.txt

# If onnxruntime fails to install from PyPI (rare on Bookworm):
# pip install onnxruntime --extra-index-url https://pkgs.dev.azure.com/onnxruntime/onnxruntime/_packaging/onnxruntime-linux-aarch64/pypi/simple/


## ─── STEP 8: Launch the application ─────────────────────────────────────────

source ~/poultry/venv/bin/activate
cd ~/poultry

# Mode B — Web dashboard (recommended, view from phone/laptop on same WiFi):
python app.py
# Open: http://<raspberry-pi-ip>:5000

# If camera is on /dev/video1:
CAMERA_SOURCE=1 python app.py

# Mode A — OpenCV desktop window (Pi with HDMI monitor or VNC):
DISPLAY_MODE=desktop python app.py

# Mode A via VNC (set DISPLAY env so OpenCV finds the VNC framebuffer):
DISPLAY=:0 DISPLAY_MODE=desktop python app.py


## ─── STEP 9: Auto-start on boot (optional) ───────────────────────────────────

# Create systemd service:
sudo tee /etc/systemd/system/poultry.service > /dev/null <<EOF
[Unit]
Description=Poultry Disease Detection
After=network.target

[Service]
ExecStart=/home/pi/poultry/venv/bin/python /home/pi/poultry/app.py
WorkingDirectory=/home/pi/poultry
Environment=CAMERA_SOURCE=0
Restart=always
RestartSec=5
User=pi

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable poultry
sudo systemctl start poultry

# Check status:
sudo systemctl status poultry
journalctl -u poultry -f


## ─── Troubleshooting ──────────────────────────────────────────────────────────

# Camera not found:
ls -la /dev/video*
sudo usermod -aG video pi   # add pi user to video group, then reboot

# Low FPS on Pi:
# Reduce INFER_INTERVAL_S in camera.py to 0.25 (4 inferences/sec)
# Or reduce imgsz in predictor.py to 320

# Memory error:
# Comment out `ultralytics` in requirements.txt after ONNX export
# Re-run: pip install -r requirements.txt

# VNC shows blank OpenCV window:
# Use Mode B (web) instead — no display required
