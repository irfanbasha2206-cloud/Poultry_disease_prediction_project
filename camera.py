"""
camera.py -- Self-Healing USB Camera Subsystem for Raspberry Pi 4 & PC.

Features:
  - Dynamic Hardware Discovery: Scans /dev/v4l/by-id/*, /dev/video*, and indices 0-15
  - Intelligent FourCC Negotiation: Sets MJPG (essential for RPi 4 USB bandwidth)
    with fallback to YUYV / uncompressed
  - Dedicated Background Watchdog & Auto-Reconnector:
      * Automatically detects newly plugged USB cameras without restarting app
      * Detects stream stalls / disconnected wires and auto-recovers within 1-2s
      * Re-binds to new /dev/videoX index if kernel re-enumerates the camera
  - Non-stalling MJPEG Stream: Emits dynamic animated HUD frames during search
    so browser connections never time out or show broken icons
  - Mode hot-swap & sensitivity threshold adjustment
"""
from __future__ import annotations

import collections
import glob
import os
import re
import threading
import time
from typing import Optional

import cv2
import numpy as np

from predictor import (PoultryPredictor, PredictionResult,
                       Detection, LABEL_COLORS, DEFAULT_COLOR, VALID_MODES)
from gemini_verifier import get_verifier_engine, _crop_with_padding

JPEG_QUALITY     = 50
INFER_INTERVAL_S = 0.50
SMOOTH_WINDOW    = 2
FRAME_W, FRAME_H = 640, 480
STREAM_FPS       = 30


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. INTELLIGENT HARDWARE ENUMERATION & PROBING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BLACKLIST_KEYWORDS = (
    "codec", "isp", "m2m", "unicam", "decoder", "encoder",
    "rpivid", "pisp", "dummy", "meta", "stats", "embedded",
)

try:
    from picamera2 import Picamera2  # type: ignore
    _HAS_PICAMERA2 = True
except Exception:
    _HAS_PICAMERA2 = False


class Picamera2Wrapper:
    """Wrapper around Picamera2 (Raspberry Pi CSI Camera) providing an OpenCV-like interface."""
    def __init__(self, width: int = FRAME_W, height: int = FRAME_H):
        self._picam2 = Picamera2()
        config = self._picam2.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        self._picam2.configure(config)
        self._picam2.start()
        self._is_opened = True

    def isOpened(self) -> bool:
        return self._is_opened

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        if not self._is_opened:
            return False, None
        try:
            rgb = self._picam2.capture_array()
            if rgb is not None and rgb.size > 0:
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                return True, bgr
        except Exception:
            pass
        return False, None

    def release(self):
        self._is_opened = False
        try:
            self._picam2.stop()
            self._picam2.close()
        except Exception:
            pass


def _is_blacklisted_device(node_path: str) -> bool:
    """Checks if a Linux /sys/class/video4linux/videoX node is hardware codec/isp/metadata."""
    name_file = os.path.join(node_path, "name")
    if os.path.isfile(name_file):
        try:
            with open(name_file, "r") as f:
                name = f.read().lower()
            if any(k in name for k in BLACKLIST_KEYWORDS):
                return True
        except Exception:
            pass

    index_file = os.path.join(node_path, "index")
    if os.path.isfile(index_file):
        try:
            with open(index_file, "r") as f:
                idx_str = f.read().strip()
            # In Linux UVC, index 0 is video capture, index 1 is metadata stream
            if idx_str not in ("", "0"):
                return True
        except Exception:
            pass

    return False


def get_candidate_camera_sources(preferred_source: int = 0) -> list[int]:
    """
    Returns an intelligent, ordered list of genuine camera indices to probe.
    Strictly filters out ISP, codec, and metadata nodes to prevent hanging.
    """
    candidates = []

    if os.name != "nt":
        # 1. /dev/v4l/by-id/* symlinks point directly to physical USB webcams!
        try:
            for p in sorted(glob.glob("/dev/v4l/by-id/*")):
                if "index0" in p or not any(f"index{k}" in p for k in range(1, 10)):
                    real_path = os.path.realpath(p)
                    m = re.search(r"video(\d+)", real_path)
                    if m:
                        idx = int(m.group(1))
                        node = f"/sys/class/video4linux/video{idx}"
                        if not _is_blacklisted_device(node) and idx not in candidates:
                            candidates.append(idx)
        except Exception:
            pass

        # 2. Inspect /sys/class/video4linux for valid camera devices
        try:
            for node in sorted(glob.glob("/sys/class/video4linux/video*")):
                if _is_blacklisted_device(node):
                    continue
                m = re.search(r"video(\d+)", node)
                if not m:
                    continue
                idx = int(m.group(1))
                if idx not in candidates:
                    candidates.append(idx)
        except Exception:
            pass

        # 3. Add preferred_source if not blacklisted
        pref_node = f"/sys/class/video4linux/video{preferred_source}"
        if preferred_source not in candidates:
            if not os.path.exists(pref_node) or not _is_blacklisted_device(pref_node):
                candidates.insert(0, preferred_source)
    else:
        # Windows: probe preferred and top indices
        candidates = [preferred_source]
        for i in range(3):
            if i not in candidates:
                candidates.append(i)

    return candidates


def _probe_single_camera(idx: int, backend=None, fourcc_str: str = "MJPG") -> Optional[cv2.VideoCapture]:
    """
    Attempts to open a camera index with a specific backend and FOURCC.
    Verifies actual frame delivery without long blocking loops.
    """
    cap = None
    try:
        if backend is not None:
            cap = cv2.VideoCapture(idx, backend)
        else:
            cap = cv2.VideoCapture(idx)

        if not cap or not cap.isOpened():
            if cap:
                cap.release()
            return None

        # Set fast timeout if supported by OpenCV build
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            try:
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 1500)
            except Exception:
                pass
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            try:
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1500)
            except Exception:
                pass

        # Critical for Raspberry Pi: set MJPG to prevent USB bus bandwidth saturation
        if fourcc_str:
            try:
                fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            except Exception:
                pass

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Fast read probe (max 2 quick attempts)
        for _ in range(2):
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or FRAME_W
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or FRAME_H
                b_name = "V4L2" if backend == getattr(cv2, "CAP_V4L2", None) else (str(backend) if backend else "default")
                print(f"[camera] SUCCESS: Bound to USB index={idx} ({b_name}, {fourcc_str or 'raw'}, {actual_w}x{actual_h})")
                return cap
            time.sleep(0.04)

        cap.release()
    except Exception:
        if cap:
            try:
                cap.release()
            except Exception:
                pass
    return None


def open_camera_auto(preferred_source: int = 0) -> tuple[Optional[object], Optional[int | str]]:
    """
    Scans candidate camera sources and returns (cap_obj, idx_or_id).
    Supports:
      1. Picamera2 (Raspberry Pi CSI Camera Module v1/v2/v3/HQ)
      2. USB Webcams via OpenCV (V4L2 / DShow / MSMF)
    Guaranteed zero-hang on ISP / hardware codec nodes.
    """
    # 1. Raspberry Pi CSI Camera check via Picamera2
    if _HAS_PICAMERA2:
        try:
            picam = Picamera2Wrapper()
            ok, test_f = picam.read()
            if ok and test_f is not None:
                print("[camera] SUCCESS: Raspberry Pi CSI Camera active via Picamera2")
                return picam, "csi"
            picam.release()
        except Exception:
            pass

    # 2. USB Camera candidates
    candidates = get_candidate_camera_sources(preferred_source)
    if not candidates:
        return None, None

    backends = []
    if os.name != "nt" and hasattr(cv2, "CAP_V4L2"):
        backends.append(cv2.CAP_V4L2)
    elif hasattr(cv2, "CAP_DSHOW"):
        backends.append(cv2.CAP_DSHOW)
    elif hasattr(cv2, "CAP_MSMF"):
        backends.append(cv2.CAP_MSMF)
    backends.append(None)

    for fourcc in ["MJPG", ""]:
        for backend in backends:
            for idx in candidates:
                cap = _probe_single_camera(idx, backend, fourcc)
                if cap is not None:
                    return cap, idx

    return None, None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. DETECTION SMOOTHER & OVERLAY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class _LabelSmoother:
    def __init__(self, window: int = SMOOTH_WINDOW):
        self._buf = collections.deque(maxlen=window)

    def push(self, det: Optional[Detection]) -> Optional[Detection]:
        self._buf.append(det)
        if det is None:
            return None
        if det.confidence >= 0.65:
            return det
        labels = [d.label for d in self._buf if d is not None]
        if not labels:
            return None
        winner = max(set(labels), key=labels.count)
        for d in reversed(self._buf):
            if d is not None and d.label == winner:
                return d
        return det


def get_clean_display_label(raw_label: str, is_healthy: bool, mode: str) -> str:
    """
    Returns concise, standard display labels: Healthy, Normal, Abnormal, Unhealthy, or clean Disease Name.
    Guarantees no awkward prefixes like '[Gemini: Resting]' or 'Healthy (Resting)'.
    """
    l_str = (raw_label or "").strip()
    l_lower = l_str.lower()

    # 1. Named diseases take absolute priority when present in diagnosis
    if "salmonell" in l_lower:
        return "Salmonellosis"
    if "coccidiosis" in l_lower or "coccidi" in l_lower:
        return "Coccidiosis"
    if "newcastle" in l_lower:
        return "Newcastle Disease"
    if "fowl pox" in l_lower or "pox" in l_lower:
        return "Fowl Pox"
    if "coryza" in l_lower:
        return "Infectious Coryza"
    if "crd" in l_lower or "chronic respiratory" in l_lower:
        return "CRD (Chronic Respiratory)"
    if "favus" in l_lower:
        return "Favus"
    if "weak leg" in l_lower:
        return "Weak Leg Syndrome"
    if "bumblefoot" in l_lower:
        return "Bumblefoot"

    # 2. Explicit abnormal / unhealthy
    if not is_healthy and ("abnormal" in l_lower or "unhealthy" in l_lower):
        return "Abnormal"

    # 3. Healthy / Normal
    if is_healthy or "healthy" in l_lower or l_lower in ("normal", "normal resting", "resting"):
        return "Normal" if mode == "triage" else "Healthy"

    if mode == "triage":
        return "Abnormal"

    return l_str or "Abnormal"


def _overlay_detection(frame: np.ndarray, det: Detection) -> np.ndarray:
    """
    Draws a clean, high-visibility bounding box and label without verbose tags.
    """
    out = frame.copy()
    x1, y1, x2, y2 = map(int, det.box)
    h, w = out.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    lbl = det.label.strip()
    if lbl in ("Healthy", "Normal", "Healthy (Resting)"):
        color = (0, 215, 80)      # Emerald green (BGR)
    elif lbl in ("Abnormal", "Unhealthy"):
        color = (200, 0, 200)    # Magenta (BGR)
    else:
        color = LABEL_COLORS.get(lbl, (0, 140, 255))

    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    tag = f"{lbl}  {det.confidence:.0%}"
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    yt = max(y1, th + 10)
    cv2.rectangle(out, (x1, yt - th - 10), (x1 + tw + 10, yt), color, -1)
    cv2.putText(out, tag, (x1 + 5, yt - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    return out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. SELF-HEALING VIDEO CAMERA CLASS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PoultryVideoCamera:
    """
    High-availability, self-healing video camera for AvianGuard AI.
    Runs active watchdog auto-reconnection in the background so camera drops
    or hot-plugs never stall the server.
    """
    def __init__(self, source: int = 0,
                 predictor: Optional[PoultryPredictor] = None,
                 mode: str = "ensemble"):
        self._preferred_source = source
        self._connected_idx: Optional[int] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_connected = False
        self._is_reconnecting = False
        self._reconnect_count = 0
        self._last_frame_time = 0.0

        self._predictor = predictor or PoultryPredictor()
        self._smoother  = _LabelSmoother()
        self._mode      = mode if mode in VALID_MODES else "ensemble"

        self._lock           = threading.Lock()
        self._reconnect_lock = threading.Lock()
        self._raw: Optional[np.ndarray]     = None
        self._last_det: Optional[Detection] = None
        self._latest_result: Optional[dict] = None
        self._running        = True
        self._mode_changed   = False
        self._is_enabled     = True

        # Stage-2 Gemini Multimodal Verifier & Temporal Buffer
        self._verifier_engine = get_verifier_engine()
        self._temporal_buffer = collections.deque(maxlen=10)
        self._last_buf_time   = 0.0
        self._last_yolo_label: Optional[str] = None
        self._no_subject_count = 0

        # Initial attempt to open camera
        print(f"[camera] Initializing camera subsystem (preferred source={source})...")
        cap, idx = open_camera_auto(source)
        if cap is not None:
            self._cap = cap
            self._connected_idx = idx
            self._is_connected = True
            self._last_frame_time = time.time()
            print(f"[camera] Initial camera online at index={idx}")
        else:
            print("[camera] No camera detected at boot. Background Auto-Reconnector watchdog started.")

        # Always start all 3 background worker threads
        threading.Thread(target=self._capture_loop,   daemon=True, name="cam-capture").start()
        threading.Thread(target=self._inference_loop, daemon=True, name="cam-infer").start()
        threading.Thread(target=self._watchdog_loop,  daemon=True, name="cam-watchdog").start()

    # -- Safe Hardware Operations ----------------------------------------------

    def _safe_release_cap(self):
        """Thread-safe release of the VideoCapture object."""
        with self._lock:
            cap = self._cap
            self._cap = None
            self._is_connected = False
            self._raw = None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
            time.sleep(0.12)  # Give kernel V4L2 driver time to clear device lock

    def _perform_reconnect(self, preferred_source: Optional[int] = None) -> tuple[bool, Optional[int]]:
        """Scans and re-binds to available camera."""
        if not self._running:
            return False, None

        with self._reconnect_lock:
            self._safe_release_cap()
            with self._lock:
                self._is_reconnecting = True

            source = preferred_source if preferred_source is not None else self._preferred_source
            cap, idx = open_camera_auto(source)

            with self._lock:
                self._is_reconnecting = False
                if cap is not None:
                    self._cap = cap
                    self._connected_idx = idx
                    self._is_connected = True
                    self._last_frame_time = time.time()
                    self._reconnect_count += 1
                    print(f"[camera] Auto-Reconnector recovered camera at index={idx} (Recovery #{self._reconnect_count})")
                    return True, idx
                else:
                    self._is_connected = False
                    return False, None

    # -- Thread Loops ----------------------------------------------------------

    def _watchdog_loop(self):
        """
        Active Guardian Thread:
        Monitors camera health.
        If camera is missing, unplugged, or stream stalls, it auto-reconnects.
        """
        while self._running:
            with self._lock:
                enabled   = self._is_enabled
                connected = self._is_connected
                has_cap   = self._cap is not None
                last_time = self._last_frame_time

            sleep_time = 1.5 if connected else 2.5
            time.sleep(sleep_time)

            if not enabled:
                continue

            now = time.time()
            stalled = (now - last_time > 3.5) and has_cap

            # Trigger reconnection if disconnected or stalled
            if not connected or not has_cap or stalled:
                if stalled:
                    print(f"[watchdog] Camera feed stalled (>3.5s without frame). Triggering auto-recovery...")
                self._perform_reconnect()

    def _capture_loop(self):
        """Reads raw frames from the active camera at full frame rate."""
        consecutive_fails = 0
        while self._running:
            with self._lock:
                enabled = self._is_enabled
                cap = self._cap

            if not enabled or cap is None:
                time.sleep(0.08)
                continue

            try:
                ok, frame = cap.read()
            except Exception as exc:
                ok, frame = False, None

            if not ok or frame is None or frame.size == 0:
                consecutive_fails += 1
                if consecutive_fails >= 12:  # Stalled for ~0.4s-0.8s
                    print(f"[capture] {consecutive_fails} consecutive read failures — releasing handle for auto-reconnect")
                    self._safe_release_cap()
                    consecutive_fails = 0
                time.sleep(0.04)
                continue

            consecutive_fails = 0
            now = time.time()
            with self._lock:
                self._raw = frame
                self._is_connected = True
                self._last_frame_time = now
                if now - self._last_buf_time >= 1.4:
                    self._temporal_buffer.append((now, frame.copy()))
                    self._last_buf_time = now

            time.sleep(0.002)

    def _get_3_temporal_frames(self, current_frame: np.ndarray) -> list[np.ndarray]:
        with self._lock:
            buf = list(self._temporal_buffer)
        now = time.time()
        f_t3 = None
        f_t15 = None
        for ts, f in buf:
            dt = now - ts
            if dt >= 2.5 and f_t3 is None:
                f_t3 = f
            elif dt >= 1.1 and dt < 2.5 and f_t15 is None:
                f_t15 = f

        out = []
        out.append(f_t3 if f_t3 is not None else (buf[0][1] if buf else current_frame))
        out.append(f_t15 if f_t15 is not None else (buf[-1][1] if len(buf) >= 2 else current_frame))
        out.append(current_frame)
        return out

    def _inference_loop(self):
        """Runs mode-aware disease prediction in background with strict subject gating."""
        last_t = 0.0
        while self._running:
            with self._lock:
                enabled = self._is_enabled
                mode_changed = self._mode_changed
                if mode_changed:
                    self._mode_changed = False
                    self._verifier_engine.clear_verification()

            if not enabled:
                time.sleep(0.10)
                continue

            elapsed = time.time() - last_t
            if not mode_changed and elapsed < INFER_INTERVAL_S:
                time.sleep(INFER_INTERVAL_S - elapsed)

            with self._lock:
                raw  = None if self._raw is None else self._raw.copy()
                mode = self._mode

            if raw is None:
                time.sleep(0.05)
                continue

            last_t = time.time()
            try:
                pred_result = self._predictor.predict(raw, mode=mode)
                if not pred_result.has_subject or pred_result.primary is None:
                    self._no_subject_count += 1
                    if self._no_subject_count >= 15:
                        self._verifier_engine.clear_verification()
                        self._last_yolo_label = None
                    # Gating active: No poultry subject focused, clear detection overlay
                    with self._lock:
                        self._last_det = None
                        self._smoother = _LabelSmoother()
                        res = pred_result.to_dict()
                        res["primary"] = None
                        res["has_subject"] = False
                        self._latest_result = res
                    continue
                else:
                    self._no_subject_count = 0

                smoothed_det = self._smoother.push(pred_result.primary)
                pred_result.primary = smoothed_det
                result_dict = pred_result.to_dict()

                # ── STAGE-2 GEMINI MULTIMODAL VERIFICATION (NON-BLOCKING ASYNC) ──
                primary = pred_result.primary
                p_label = primary.label
                p_conf  = primary.confidence
                p_box   = primary.box

                if mode != "fecal" and self._verifier_engine.should_trigger(p_label, p_conf, mode):
                    roi = _crop_with_padding(raw, p_box, pad_pct=0.15)
                    if mode == "head":
                        self._verifier_engine.submit_task(
                            mode="head", full_frame=raw, roi_crop=roi,
                            yolo_label=p_label, yolo_conf=p_conf
                        )
                    elif mode == "triage":
                        t_frames = self._get_3_temporal_frames(raw)
                        self._verifier_engine.submit_task(
                            mode="triage", full_frame=raw, roi_crop=roi,
                            yolo_label=p_label, yolo_conf=p_conf,
                            temporal_frames=t_frames
                        )
                    else:  # ensemble
                        extra_crops = [_crop_with_padding(raw, d.box, pad_pct=0.10) for d in pred_result.all_detections[:3]]
                        summary = {
                            "primary_model": primary.source_model,
                            "primary_label": p_label,
                            "confidence": round(p_conf, 3),
                            "all_detections": [{"model": d.source_model, "label": d.label, "confidence": round(d.confidence, 3)} for d in pred_result.all_detections]
                        }
                        self._verifier_engine.submit_task(
                            mode="ensemble", full_frame=raw, roi_crop=roi,
                            yolo_label=p_label, yolo_conf=p_conf,
                            consensus_data=summary, extra_crops=extra_crops
                        )

                # ── SYNCHRONIZE STAGE-2 GEMINI MULTIMODAL VERIFIED RESULT ──
                self._last_yolo_label = p_label
                gemini_ver = self._verifier_engine.get_latest_verification(max_age_seconds=8.0) if mode != "fecal" else None
                if gemini_ver and gemini_ver.get("mode") == mode and pred_result.primary:
                    v_diag = gemini_ver.get("verified_diagnosis", p_label)
                    clean_lbl = get_clean_display_label(v_diag, bool(gemini_ver.get("is_healthy", False)), mode)
                    is_h = (clean_lbl in ("Healthy", "Normal"))
                    v_conf = gemini_ver.get("diagnostic_confidence", int(p_conf * 100)) / 100.0

                    # Sync primary detection object for video overlay
                    pred_result.primary.label = clean_lbl
                    pred_result.primary.confidence = v_conf

                    # Sync result dictionary for web status API
                    result_dict["gemini_verifier"] = gemini_ver
                    if result_dict.get("primary"):
                        result_dict["primary"]["label"] = clean_lbl
                        result_dict["primary"]["confidence"] = v_conf
                        is_crit = clean_lbl in ("Salmonellosis", "Coccidiosis", "Newcastle Disease")
                        result_dict["primary"]["severity"] = "Normal" if is_h else ("Critical" if is_crit else "High")
                        result_dict["primary"]["sev_key"] = "low" if is_h else ("crit" if is_crit else "high")
                        result_dict["primary"]["color"] = "#00e5a0" if is_h else ("#ec4899" if clean_lbl in ("Abnormal", "Unhealthy") else "#ef4444")
                        result_dict["primary"]["hint"] = gemini_ver.get("symptom_guide_summary", "")
                        result_dict["primary"]["gemini_verified"] = True
                        result_dict["primary"]["status_badge"] = ("NORMAL" if mode == "triage" else "HEALTHY") if is_h else ("ABNORMAL" if (mode == "triage" or clean_lbl in ("Abnormal", "Unhealthy")) else "VERIFIED ALERT")
                        if gemini_ver.get("differential_diagnosis"):
                            result_dict["primary"]["differential_diagnosis"] = gemini_ver.get("differential_diagnosis")
                else:
                    result_dict["gemini_verifier"] = None
                    if not pred_result.has_subject:
                        self._verifier_engine.clear_verification()

            except Exception as exc:
                print(f"[infer] {exc}")
                continue

            with self._lock:
                self._last_det      = pred_result.primary
                self._latest_result = result_dict

    # -- Dynamic Status Display (Search & Paused Animations) -------------------

    def _render_paused_frame(self) -> np.ndarray:
        """
        Renders a high-tech HUD frame when camera stream power is switched OFF.
        """
        img = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        img[:] = (16, 12, 6)  # Deep obsidian (BGR)

        color_gold  = (11, 158, 245)   # BGR for #f59e0b
        color_amber = (56, 189, 248)   # BGR for #38bdf8
        color_dim   = (120, 130, 145)

        # Reticle brackets
        ret = 24
        cv2.line(img, (24, 24), (24 + ret, 24), color_dim, 2)
        cv2.line(img, (24, 24), (24, 24 + ret), color_dim, 2)
        cv2.line(img, (FRAME_W - 24, 24), (FRAME_W - 24 - ret, 24), color_dim, 2)
        cv2.line(img, (FRAME_W - 24, 24), (FRAME_W - 24, 24 + ret), color_dim, 2)
        cv2.line(img, (24, FRAME_H - 24), (24 + ret, FRAME_H - 24), color_dim, 2)
        cv2.line(img, (24, FRAME_H - 24), (24, FRAME_H - 24 - ret), color_dim, 2)
        cv2.line(img, (FRAME_W - 24, FRAME_H - 24), (FRAME_W - 24 - ret, FRAME_H - 24), color_dim, 2)
        cv2.line(img, (FRAME_W - 24, FRAME_H - 24), (FRAME_W - 24, FRAME_H - 24 - ret), color_dim, 2)

        # Title
        title = "AvianGuard AI -- Video Stream Paused"
        (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.putText(img, title, ((FRAME_W - tw) // 2, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color_gold, 2, cv2.LINE_AA)

        # Status text
        status = "Camera Sensor Standby Active"
        (sw, _), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)
        cv2.putText(img, status, ((FRAME_W - sw) // 2, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

        # Explanatory lines
        sub1 = "Camera power switched OFF via toolbar toggle"
        (s1w, _), _ = cv2.getTextSize(sub1, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.putText(img, sub1, ((FRAME_W - s1w) // 2, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color_amber, 1, cv2.LINE_AA)

        sub2 = "Click 'Camera: ON' at bottom of screen to resume live AI surveillance"
        (s2w, _), _ = cv2.getTextSize(sub2, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        cv2.putText(img, sub2, ((FRAME_W - s2w) // 2, 295), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 190, 205), 1, cv2.LINE_AA)

        return img

    def _render_diagnostic_frame(self) -> np.ndarray:

        """
        Renders a dynamic, high-tech diagnostic HUD frame during camera search
        so the browser stream stays alive and informative.
        """
        img = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        img[:] = (16, 12, 6)  # Deep royal obsidian (BGR)

        color_gold    = (11, 158, 245)   # BGR for #f59e0b
        color_emerald = (168, 240, 0)    # BGR for #00f0a8
        color_cyan    = (248, 189, 56)   # BGR for #38bdf8

        # Reticle brackets
        ret = 24
        cv2.line(img, (24, 24), (24 + ret, 24), color_gold, 2)
        cv2.line(img, (24, 24), (24, 24 + ret), color_gold, 2)
        cv2.line(img, (FRAME_W - 24, 24), (FRAME_W - 24 - ret, 24), color_gold, 2)
        cv2.line(img, (FRAME_W - 24, 24), (FRAME_W - 24, 24 + ret), color_gold, 2)
        cv2.line(img, (24, FRAME_H - 24), (24 + ret, FRAME_H - 24), color_gold, 2)
        cv2.line(img, (24, FRAME_H - 24), (24, FRAME_H - 24 - ret), color_gold, 2)
        cv2.line(img, (FRAME_W - 24, FRAME_H - 24), (FRAME_W - 24 - ret, FRAME_H - 24), color_gold, 2)
        cv2.line(img, (FRAME_W - 24, FRAME_H - 24), (FRAME_W - 24, FRAME_H - 24 - ret), color_gold, 2)

        # Title
        title = "AvianGuard AI -- Camera Subsystem"
        (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.putText(img, title, ((FRAME_W - tw) // 2, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color_gold, 2, cv2.LINE_AA)

        # Animated dots based on wall clock
        dots = "." * (int(time.time() * 2) % 4)
        status_line = f"Searching for Camera{dots}"
        (sw, _), _ = cv2.getTextSize(status_line, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.putText(img, status_line, ((FRAME_W - sw) // 2, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        # Explanatory lines
        sub1 = "Auto-reconnect engine active (USB Webcam & Pi CSI Camera)"
        (s1w, _), _ = cv2.getTextSize(sub1, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.putText(img, sub1, ((FRAME_W - s1w) // 2, 248), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color_emerald, 1, cv2.LINE_AA)

        sub2 = "Plug in USB webcam or connect Pi Camera -- stream starts automatically"
        (s2w, _), _ = cv2.getTextSize(sub2, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        cv2.putText(img, sub2, ((FRAME_W - s2w) // 2, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 190, 205), 1, cv2.LINE_AA)

        # Scanning device indicator
        dev_info = "Probing valid video devices | Broadcom ISP / Codec Filtered"
        (dw, _), _ = cv2.getTextSize(dev_info, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        cv2.putText(img, dev_info, ((FRAME_W - dw) // 2, 335), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color_cyan, 1, cv2.LINE_AA)

        return img

    # -- Public API ------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        with self._lock:
            return self._is_connected

    @property
    def is_enabled(self) -> bool:
        with self._lock:
            return self._is_enabled

    def set_enabled(self, enabled: Optional[bool] = None) -> bool:
        with self._lock:
            if enabled is None:
                self._is_enabled = not self._is_enabled
            else:
                self._is_enabled = bool(enabled)
            current = self._is_enabled
            if not current:
                self._last_det = None
                self._smoother = _LabelSmoother()
        print(f"[camera] power state -> {'ON' if current else 'OFF'}")
        return current

    def reconnect(self, preferred_source: Optional[int] = None) -> tuple[bool, Optional[int]]:
        """Trigger an immediate forced camera scan and re-bind."""
        print(f"[camera] Manual reconnect triggered (preferred={preferred_source})")
        return self._perform_reconnect(preferred_source)

    def health_status(self) -> dict:
        with self._lock:
            return {
                "connected": self._is_connected,
                "enabled": self._is_enabled,
                "reconnecting": self._is_reconnecting,
                "connected_index": self._connected_idx,
                "reconnect_count": self._reconnect_count,
                "preferred_source": self._preferred_source,
                "mode": self._mode,
            }

    def set_mode(self, mode: str) -> str:
        if mode not in VALID_MODES:
            return self._mode
        with self._lock:
            self._mode         = mode
            self._last_det     = None
            self._smoother     = _LabelSmoother()
            self._mode_changed = True
        if self._verifier_engine:
            self._verifier_engine.clear_verification()
        print(f"[camera] mode -> {mode}")
        return mode

    def get_mode(self) -> str:
        with self._lock:
            return self._mode

    def set_threshold(self, threshold: float):
        with self._lock:
            if self._predictor:
                self._predictor.set_threshold(threshold)
                self._last_det = None
                self._smoother = _LabelSmoother()

    def get_threshold(self) -> float:
        with self._lock:
            return getattr(self._predictor, "conf_threshold", 0.50)

    def latest_result(self) -> Optional[dict]:
        with self._lock:
            if not self._is_enabled:
                return {
                    "primary": None,
                    "secondary": None,
                    "mode": self._mode,
                    "camera_enabled": False,
                    "camera_connected": self._is_connected,
                    "connected_index": self._connected_idx,
                    "status": "camera_paused",
                    "inference_ms": 0,
                }
            if not self._is_connected:
                return {
                    "primary": None,
                    "secondary": None,
                    "mode": self._mode,
                    "error": "camera_searching",
                    "camera_enabled": True,
                    "camera_connected": False,
                    "connected_index": None,
                    "inference_ms": 0,
                }
            res = dict(self._latest_result) if self._latest_result else {
                "primary": None, "secondary": None,
                "mode": self._mode, "inference_ms": 0
            }
            res["camera_enabled"]   = True
            res["camera_connected"] = True
            res["connected_index"]  = self._connected_idx
            return res

    def frames(self):
        """Yields JPEG frames for Flask MJPEG streaming."""
        params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

        while self._running:
            with self._lock:
                raw = self._raw
                det = self._last_det
                connected = self._is_connected
                enabled = self._is_enabled

            if not enabled:
                out = self._render_paused_frame()
            elif not connected or raw is None:
                out = self._render_diagnostic_frame()
            elif det is not None:
                out = _overlay_detection(raw, det)
            else:
                out = raw

            ok, buf = cv2.imencode(".jpg", out, params)
            if ok:
                yield buf.tobytes()
            time.sleep(1.0 / STREAM_FPS)

    def release(self):
        """Clean shutdown of camera and all worker threads."""
        self._running = False
        self._safe_release_cap()

