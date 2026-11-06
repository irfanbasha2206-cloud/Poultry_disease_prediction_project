"""
app.py -- Flask web dashboard with mode-aware disease prediction.

Mode A (Desktop):  DISPLAY_MODE=desktop python app.py
Mode B (Web):      python app.py  ->  http://<pi-ip>:5000

Environment variables:
  CAMERA_SOURCE   int   USB camera index (default 0)
  DISPLAY_MODE    str   "web" (default) | "desktop"
  DEFAULT_MODE    str   head | fecal | triage | ensemble  (default: ensemble)
"""
from __future__ import annotations

import os
import threading
import traceback

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

from camera import PoultryVideoCamera
from predictor import PoultryPredictor, VALID_MODES
from gemini_agent import get_poultry_agent

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
    default_email = os.environ.get("DEFAULT_RECIPIENT", "")
    return render_template("index.html", valid_modes=VALID_MODES,
                           default_mode=DEFAULT_MODE,
                           default_email=default_email)


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

    agent = get_poultry_agent()
    
    # ── Agent Step 1: Intelligent Multimodal Subject Gatekeeping ────────
    eval_res = agent.evaluate_upload(frame)
    if not eval_res.get("is_poultry", False):
        return jsonify({
            "error": "no_poultry_detected",
            "has_subject": False,
            "primary": None,
            "all_detections": [],
            "agent_evaluated": True,
            "agent_category": eval_res.get("category", "NON_POULTRY"),
            "user_message": eval_res.get("reason", "Kindly upload a clear photo of a chicken, poultry flock, or droppings."),
            "override_reason": eval_res.get("reason", "Non-poultry subject rejected by AvianGuard AI Agent.")
        })

    # ── Agent Step 2: Intelligent Perspective Model Routing ────────────
    # Agent automatically routes bird photos to head/cranial models, and droppings to fecal models
    routed_mode = eval_res.get("routed_model") or request.form.get("mode", "head")
    if routed_mode not in VALID_MODES:
        routed_mode = "head"

    with _init_lock:
        pred = _get_predictor_unlocked()

    try:
        result = pred.predict_uploaded(frame, mode=routed_mode)
        d = result.to_dict()

        if not result.has_subject or result.primary is None:
            d["error"] = "no_poultry_detected"
            d["user_message"] = eval_res.get("reason") or "Kindly upload a clear chicken or droppings image to predict disease."
            return jsonify(d)

        # ── Stage-2: Gemini Multimodal Verification & Consensus ────────────
        try:
            from gemini_verifier import get_gemini_verifier, _crop_with_padding
            verifier = get_gemini_verifier()
            p_label = result.primary.label
            p_conf = result.primary.confidence
            roi_crop = None
            if result.primary.box:
                roi_crop = _crop_with_padding(frame, result.primary.box, pad_pct=0.15)
            else:
                roi_crop = frame

            ver_res = None
            if routed_mode == "head":
                ver_res = verifier.verify_cranial_appearance(frame, roi_crop, p_label, p_conf)
            elif routed_mode == "fecal":
                ver_res = None  # Pure model prediction for fecal mode without Gemini override
            elif routed_mode == "ensemble":
                summary = {
                    "primary_model": result.primary.source_model,
                    "primary_label": p_label,
                    "confidence": round(p_conf, 3),
                    "all_detections": [{"model": det.source_model, "label": det.label, "confidence": round(det.confidence, 3)} for det in result.all_detections]
                }
                extra_crops = [_crop_with_padding(frame, det.box, pad_pct=0.10) for det in result.all_detections[:3] if det.box]
                ver_res = verifier.verify_consensus_ensemble(frame, roi_crop, summary, extra_crops)
            elif routed_mode == "triage":
                ver_res = verifier.verify_behavior_temporal(frame, roi_crop, [frame], p_label, p_conf)

            if ver_res:
                d["gemini_verifier"] = ver_res
                if ver_res.get("is_healthy"):
                    d["primary"]["label"] = "Normal" if routed_mode == "triage" else "Healthy"
                    d["primary"]["confidence"] = ver_res.get("diagnostic_confidence", 95) / 100.0
                    d["primary"]["severity"] = "Normal"
                    d["primary"]["sev_key"] = "low"
                    d["primary"]["hint"] = ver_res.get("symptom_guide_summary", "")
                    d["primary"]["gemini_verified"] = True
                    d["primary"]["status_badge"] = "NORMAL" if routed_mode == "triage" else "HEALTHY"
                elif not ver_res.get("is_healthy"):
                    d["primary"]["label"] = ver_res.get("verified_diagnosis", d["primary"]["label"])
                    d["primary"]["confidence"] = ver_res.get("diagnostic_confidence", int(p_conf * 100)) / 100.0
                    d["primary"]["hint"] = ver_res.get("symptom_guide_summary", "")
                    d["primary"]["gemini_verified"] = True
                    d["primary"]["status_badge"] = "ABNORMAL" if (routed_mode == "triage" or d["primary"]["label"] in ("Abnormal", "Unhealthy")) else "VERIFIED ALERT"
                    if ver_res.get("differential_diagnosis"):
                        d["primary"]["differential_diagnosis"] = ver_res.get("differential_diagnosis")
        except Exception as ver_err:
            print(f"[app] Stage-2 verification upload error: {ver_err}")

        # ── Agent Step 3: Veterinary Clinical Consultation Synthesis ───────
        consultation = agent.generate_clinical_consultation(
            disease_label=d["primary"]["label"],
            confidence=d["primary"]["confidence"],
            category=eval_res.get("category", "CHICKEN_BIRD"),
            image_bgr=frame
        )
        d["agent_consultation"] = consultation
        d["agent_routing"] = f"AvianGuard Agent Verified ({eval_res.get('category')}) -> Routed to {routed_mode.upper()} Pipeline"

        # Add bounding box for primary detection
        if result.primary is not None:
            d["primary"]["box"]        = list(result.primary.box)
            d["primary"]["image_size"] = [frame.shape[1], frame.shape[0]]

        # ── Agent Step 4: Optional Automated Email Alert ───────────────────
        alert_email = request.form.get("alert_email", "").strip()
        auto_email  = request.form.get("send_email", "false").lower() == "true"
        if alert_email and auto_email and d["primary"]["label"] != "Healthy" and not d["primary"]["label"].startswith("Healthy"):
            threading.Thread(
                target=agent.send_email_alert,
                args=(alert_email, d["primary"]["label"], d["primary"]["confidence"], consultation, frame),
                daemon=True
            ).start()
            d["email_dispatched"] = True
            d["email_recipient"] = alert_email

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


@app.route("/send_email_alert", methods=["POST"])
def send_email_alert_endpoint():
    """Manual or on-demand email alert dispatcher."""
    data = request.get_json(silent=True) or request.form
    default_rec = os.environ.get("DEFAULT_RECIPIENT", "")
    recipient = data.get("email", "").strip() or default_rec
    disease = data.get("disease", "Suspected Avian Pathogen")
    confidence = float(data.get("confidence", 0.90))
    consultation = data.get("consultation", {})

    agent = get_poultry_agent()
    ok, msg = agent.send_email_alert(recipient, disease, confidence, consultation)
    return jsonify({"success": ok, "message": msg})


@app.route("/live_stream_alert", methods=["POST"])
def live_stream_alert_endpoint():
    """Automated email alert triggered during real-time camera surveillance."""
    data = request.get_json(silent=True) or {}
    default_rec = os.environ.get("DEFAULT_RECIPIENT", "")
    recipient = data.get("email", "").strip() or data.get("recipient", "").strip() or default_rec
    disease = data.get("disease", "Suspected Avian Pathogen")
    confidence = float(data.get("confidence", 0.90))

    agent = get_poultry_agent()
    consultation = agent.generate_clinical_consultation(disease, confidence, "CHICKEN_BIRD")

    raw_frame = None
    with _init_lock:
        if _camera is not None and getattr(_camera, "_raw", None) is not None:
            raw_frame = _camera._raw.copy()

    def _bg_send():
        try:
            agent.send_email_alert(recipient, disease, confidence, consultation, image_bgr=raw_frame)
            print(f"[app] Live camera email alert dispatched to {recipient} for {disease} ({confidence:.0%})")
        except Exception as e:
            print(f"[app] Live stream alert email error: {e}")

    threading.Thread(target=_bg_send, daemon=True).start()
    return jsonify({"success": True, "message": f"Live camera email alert dispatched to {recipient}"})



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
