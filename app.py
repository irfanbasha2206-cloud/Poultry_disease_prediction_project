"""
app.py -- Flask web dashboard with mode-aware disease prediction.

Mode A (Desktop):  DISPLAY_MODE=desktop python app.py
Mode B (Web):      python app.py  ->  http://<pi-ip>:5000

Environment variables:
  CAMERA_SOURCE   int   USB camera index (default 0)
  DISPLAY_MODE    str   "web" (default) | "desktop"
  DEFAULT_MODE    str   head | fecal | triage | ensemble  (default: ensemble)
"""
import os
import threading
import traceback

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

from camera import PoultryVideoCamera
from predictor import PoultryPredictor, VALID_MODES

app = Flask(__name__)

CAMERA_SOURCE = int(os.environ.get("CAMERA_SOURCE", "0"))
DISPLAY_MODE  = os.environ.get("DISPLAY_MODE", "web").lower()
DEFAULT_MODE  = os.environ.get("DEFAULT_MODE", "ensemble")
if DEFAULT_MODE not in VALID_MODES:
    DEFAULT_MODE = "ensemble"

_init_lock = threading.Lock()
_camera:    PoultryVideoCamera | None = None
_predictor: PoultryPredictor   | None = None


def _get_predictor_unlocked() -> PoultryPredictor:
    global _predictor
    if _predictor is None:
        print("[app] loading models...")
        _predictor = PoultryPredictor()
        print("[app] models ready")
    return _predictor


def _get_camera() -> PoultryVideoCamera:
    global _camera
    with _init_lock:
        if _camera is None:
            _camera = PoultryVideoCamera(
                source=CAMERA_SOURCE,
                predictor=_get_predictor_unlocked(),
                mode=DEFAULT_MODE,
            )
    return _camera


def _mjpeg_stream(camera: PoultryVideoCamera):
    try:
        for frame in camera.frames():
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
    except GeneratorExit:
        return
    except Exception:
        traceback.print_exc()


@app.route("/")
def index():
    return render_template("index.html", valid_modes=VALID_MODES,
                           default_mode=DEFAULT_MODE)


@app.route("/video_feed")
def video_feed():
    try:
        camera = _get_camera()
    except Exception:
        traceback.print_exc()
        return "Camera failed to start.", 500
    return Response(
        _mjpeg_stream(camera),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/status")
def status():
    with _init_lock:
        cam = _camera
    if cam is None:
        return jsonify({"primary": None, "secondary": None,
                        "mode": DEFAULT_MODE, "inference_ms": 0})
    result = cam.latest_result()
    return jsonify(result or {"primary": None, "secondary": None,
                              "mode": DEFAULT_MODE, "inference_ms": 0})


@app.route("/set_mode", methods=["POST"])
def set_mode():
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "ensemble")
    cam  = _get_camera()
    applied = cam.set_mode(mode)
    return jsonify({"mode": applied, "ok": True})


@app.route("/get_mode")
def get_mode():
    with _init_lock:
        cam = _camera
    current = cam.get_mode() if cam else DEFAULT_MODE
    return jsonify({"mode": current})


@app.route("/set_threshold", methods=["POST"])
def set_threshold():
    data = request.get_json(silent=True) or {}
    thresh = float(data.get("threshold", 0.50))
    cam = _get_camera()
    cam.set_threshold(thresh)
    return jsonify({"threshold": thresh, "ok": True})


@app.route("/get_threshold")
def get_threshold():
    with _init_lock:
        cam = _camera
    thresh = cam.get_threshold() if cam else 0.50
    return jsonify({"threshold": thresh})


@app.route("/reconnect_camera", methods=["POST"])
def reconnect_camera():
    data = request.get_json(silent=True) or {}
    src = data.get("source")
    if src is not None:
        try:
            src = int(src)
        except ValueError:
            src = None
    cam = _get_camera()
    ok, idx = cam.reconnect(preferred_source=src)
    return jsonify({
        "ok": ok,
        "connected": cam.is_connected,
        "device_index": idx,
        "health": cam.health_status(),
    })


@app.route("/camera_health")
def camera_health():
    cam = _get_camera()
    return jsonify(cam.health_status())


@app.route("/toggle_camera", methods=["POST"])
def toggle_camera():
    data = request.get_json(silent=True) or {}
    desired = data.get("enabled")
    cam = _get_camera()
    state = cam.set_enabled(desired)
    return jsonify({"enabled": state, "ok": True})



@app.route("/predict_image", methods=["POST"])
def predict_image():
    if "image" not in request.files:
        return jsonify({"error": "no image uploaded"}), 400
    buf   = np.frombuffer(request.files["image"].read(), dtype=np.uint8)
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "could not decode image"}), 400

    # For uploaded images, use the mode selected in the UI.
    # Default to "head" so general.onnx (Fowl Pox, Favus, Coryza…) is always
    # tried first — most uploaded images show visible head/body lesions.
    mode = request.form.get("mode", "head")
    if mode not in VALID_MODES:
        mode = "head"

    with _init_lock:
        pred = _get_predictor_unlocked()

    try:
        result = pred.predict_uploaded(frame, mode=mode)
        d = result.to_dict()

        if not result.has_subject:
            d["error"] = "no_poultry_detected"
            d["user_message"] = "Kindly upload a chicken, poultry flock, or droppings image to predict disease."
            return jsonify(d)

        # Add bounding box for primary detection
        if result.primary is not None:
            d["primary"]["box"]        = list(result.primary.box)
            d["primary"]["image_size"] = [frame.shape[1], frame.shape[0]]

        return jsonify(d)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": str(exc),
            "primary": None,
            "has_subject": False,
            "all_detections": []
        }), 500


def _desktop_mode():
    print("[desktop] press Q to quit")
    camera = _get_camera()
    cv2.namedWindow("Poultry Disease Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Poultry Disease Detection", 800, 600)
    for jpeg_bytes in camera.frames():
        frame = cv2.imdecode(
            np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if frame is None:
            continue
        result = camera.latest_result()
        if result and result.get("primary"):
            p = result["primary"]
            label = p.get("label", "")
            conf  = p.get("confidence") or 0
            info  = f"{label}  {conf:.0%}"
            cv2.putText(frame, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 229, 160), 2)
        cv2.imshow("Poultry Disease Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if DISPLAY_MODE == "desktop":
        print("[app] Mode A -- OpenCV desktop window")
        try:
            with _init_lock:
                _get_predictor_unlocked()
        except Exception:
            traceback.print_exc()
            raise
        _desktop_mode()
    else:
        print("[app] Mode B -- Web dashboard  ->  http://0.0.0.0:5000")
        print("      Open http://<raspberry-pi-ip>:5000 on any device on the LAN.")
        try:
            with _init_lock:
                _get_predictor_unlocked()
        except Exception:
            traceback.print_exc()
            raise
        app.run(
            host="0.0.0.0",
            port=5000,
            debug=False,
            threaded=True,
            use_reloader=False,
        )
