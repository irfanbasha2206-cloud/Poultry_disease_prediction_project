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

JPEG_QUALITY     = 50
INFER_INTERVAL_S = 0.50
SMOOTH_WINDOW    = 2
FRAME_W, FRAME_H = 640, 480
STREAM_FPS       = 30


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. INTELLIGENT HARDWARE ENUMERATION & PROBING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_candidate_camera_sources(preferred_source: int = 0) -> list[int]:
    """
    Returns an intelligent, ordered list of camera indices to probe.
    Prioritizes real USB webcams over Pi ISP/codec nodes.
    """
    candidates = []

    # 1. Preferred source first
    candidates.append(preferred_source)

    # 2. Linux / Raspberry Pi Hardware Inspection
    if os.name != "nt":
        # /dev/v4l/by-id/* symlinks point directly to physical USB webcams
        try:
            for p in sorted(glob.glob("/dev/v4l/by-id/*")):
                # In Linux, index0 is video capture, index1 is metadata
                if "index0" in p or not any(f"index{k}" in p for k in range(1, 10)):
                    real_path = os.path.realpath(p)
                    m = re.search(r"video(\d+)", real_path)
                    if m:
                        idx = int(m.group(1))
                        if idx not in candidates:
                            candidates.append(idx)
        except Exception:
            pass

        # /sys/class/video4linux to identify real camera names vs bcm2835-codec
        try:
            usb_nodes = []
            other_nodes = []
            for node in glob.glob("/sys/class/video4linux/video*"):
                m = re.search(r"video(\d+)", node)
                if not m:
                    continue
                idx = int(m.group(1))
                name_file = os.path.join(node, "name")
                name = ""
                if os.path.exists(name_file):
                    try:
                        with open(name_file, "r") as f:
                            name = f.read().lower()
                    except Exception:
                        pass
                # Check for camera / webcam keywords (not codec / isp / unicam-dummy)
                if any(k in name for k in ["usb", "camera", "webcam", "uvc", "cam"]):
                    usb_nodes.append(idx)
                else:
                    other_nodes.append(idx)

            for idx in sorted(usb_nodes):
                if idx not in candidates:
                    candidates.append(idx)
            for idx in sorted(other_nodes):
                if idx not in candidates and idx < 16:
                    candidates.append(idx)
        except Exception:
            pass

        # Direct /dev/video* fallback
        try:
            for dev in sorted(glob.glob("/dev/video*")):
                m = re.search(r"video(\d+)", dev)
                if m:
                    idx = int(m.group(1))
                    if idx not in candidates and idx < 16:
                        candidates.append(idx)
        except Exception:
            pass

    # 3. Numeric sweep fallback (0 to 10)
    for i in range(10):
        if i not in candidates:
            candidates.append(i)

    return candidates


def _probe_single_camera(idx: int, backend=None, fourcc_str: str = "MJPG") -> Optional[cv2.VideoCapture]:
    """
    Attempts to open a camera index with a specific backend and FOURCC.
    Verifies actual frame delivery before declaring success.
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

        # Warm up sensor: some USB cameras take 2-4 read attempts before delivering the first frame
        for _ in range(5):
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                b_name = "V4L2" if backend == getattr(cv2, "CAP_V4L2", None) else (str(backend) if backend else "default")
                print(f"[camera] SUCCESS: Bound to index={idx} ({b_name}, {fourcc_str or 'raw'}, {actual_w}x{actual_h})")
                return cap
            time.sleep(0.04)

        cap.release()
    except Exception as exc:
        if cap:
            try:
                cap.release()
            except Exception:
                pass
    return None


def open_camera_auto(preferred_source: int = 0) -> tuple[Optional[cv2.VideoCapture], Optional[int]]:
    """
    Scans all candidate indices with multiple backends and formats.
    Returns (cap, idx) on success, or (None, None) if no camera is available.
    """
    backends = []
    if os.name != "nt" and hasattr(cv2, "CAP_V4L2"):
        backends.append(cv2.CAP_V4L2)
    if hasattr(cv2, "CAP_DSHOW"):
        backends.append(cv2.CAP_DSHOW)
    if hasattr(cv2, "CAP_MSMF"):
        backends.append(cv2.CAP_MSMF)
    backends.append(None)  # cv2.CAP_ANY / default

    candidates = get_candidate_camera_sources(preferred_source)
    # First pass: try with MJPG (fastest on USB 2.0 / Pi 4)
    # Second pass: fallback to uncompressed / raw
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


def _overlay_detection(frame: np.ndarray, det: Detection) -> np.ndarray:
    out = frame.copy()
    x1, y1, x2, y2 = map(int, det.box)
    h, w = out.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    color = LABEL_COLORS.get(det.label, DEFAULT_COLOR)
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    tag = f"{det.label}  {det.confidence:.0%}"
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    yt = max(y1, th + 10)
    cv2.rectangle(out, (x1, yt - th - 10), (x1 + tw + 8, yt), color, -1)
    cv2.putText(out, tag, (x1 + 4, yt - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
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
        Monitors camera health every 1.5 seconds.
        If camera is missing, unplugged, or stream stalls, it auto-reconnects.
        """
        while self._running:
            time.sleep(1.5)

            with self._lock:
                enabled   = self._is_enabled
                connected = self._is_connected
                has_cap   = self._cap is not None
                last_time = self._last_frame_time

            if not enabled:
                continue

            now = time.time()
            stalled = (now - last_time > 3.0) and has_cap

            # Trigger reconnection if disconnected or stalled
            if not connected or not has_cap or stalled:
                if stalled:
                    print(f"[watchdog] Camera feed stalled (>3.0s without frame). Triggering auto-recovery...")
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
            with self._lock:
                self._raw = frame
                self._is_connected = True
                self._last_frame_time = time.time()

            time.sleep(0.002)

    def _inference_loop(self):
        """Runs mode-aware disease prediction in background with strict subject gating."""
        last_t = 0.0
        while self._running:
            with self._lock:
                enabled = self._is_enabled
                mode_changed = self._mode_changed
                if mode_changed:
                    self._mode_changed = False

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
                    # Gating active: No poultry subject focused, clear detection overlay
                    with self._lock:
                        self._last_det = None
                        self._smoother = _LabelSmoother()
                        res = pred_result.to_dict()
                        res["primary"] = None
                        res["has_subject"] = False
                        self._latest_result = res
                    continue

                smoothed_det = self._smoother.push(pred_result.primary)
                pred_result.primary = smoothed_det
                result_dict = pred_result.to_dict()
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
        status_line = f"Searching for USB Camera{dots}"
        (sw, _), _ = cv2.getTextSize(status_line, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.putText(img, status_line, ((FRAME_W - sw) // 2, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        # Explanatory lines
        sub1 = "Auto-reconnect engine active (/dev/video*)"
        (s1w, _), _ = cv2.getTextSize(sub1, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        cv2.putText(img, sub1, ((FRAME_W - s1w) // 2, 248), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color_emerald, 1, cv2.LINE_AA)

        sub2 = "Plug in or reposition USB webcam -- stream starts automatically"
        (s2w, _), _ = cv2.getTextSize(sub2, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        cv2.putText(img, sub2, ((FRAME_W - s2w) // 2, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 190, 205), 1, cv2.LINE_AA)

        # Scanning device indicator
        dev_info = "Probing video indices 0-15 | V4L2 Hardware Watchdog Active"
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

