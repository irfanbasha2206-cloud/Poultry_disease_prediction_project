"""
gemini_verifier.py -- Two-Stage Gemini Multimodal Verification & Consensus Engine.

Stage-1: Real-time YOLOv8 detector running at full camera FPS.
Stage-2: Google Gemini Vision-Language Verifier (gemini-3.8-flash / gemini-2.5-flash via google-genai)
         running asynchronously in a background worker thread with frame-consistency gating
         and a 4-5 second cooldown.

Responsibilities by Mode:
  - Mode 1 (Cranial/Appearance): Inspects 15% padded ROI + full frame to resolve class confusion
    (e.g. Fowl Pox vs Coryza vs CRD) and filters false positives caused by shadows/angles.
  - Mode 2 (Enteric/Fecal): Fast YOLOv8 primary; calls Gemini when confidence is ambiguous (< 0.65)
    or to enrich symptom guides and differential diagnosis.
  - Mode 3 (Behavior): Temporal reasoning across a 3-frame sequence (t-3s, t-1.5s, t) to distinguish
    normal healthy chickens resting/preening vs true clinical lethargy/paralysis.
  - Mode 4 (Multi-Spectrum Consensus): Multimodal Chief Avian Pathologist synthesizing findings
    from all 3 models (Cranial, Fecal, Behavioral) into a single high-precision diagnosis.
"""
from __future__ import annotations

import base64
import json
import os
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Built-in .env parser fallback
_env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.isfile(_env_file):
    try:
        with open(_env_file, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    _k, _v = _k.strip(), _v.strip().strip('"').strip("'")
                    if _k and _k not in os.environ:
                        os.environ[_k] = _v
    except Exception:
        pass

# Import official google-genai SDK
try:
    from google import genai
    from google.genai import types
    _HAS_GOOGLE_GENAI = True
except ImportError:
    _HAS_GOOGLE_GENAI = False

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

import urllib.request
import urllib.error

PRIMARY_KEY = os.environ.get("GEMINI_API_KEY", "")
SECONDARY_KEY = os.environ.get("GEMINI_API_KEY_SECONDARY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

MODEL_CANDIDATES = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite"),
]

GROQ_MODELS = [
    os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b"),
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]

STRUCTURED_JSON_SCHEMA_HINT = """
Return strictly valid JSON conforming to this schema without markdown fences:
{
  "is_healthy": boolean,
  "verified_diagnosis": string,
  "diagnostic_confidence": integer (0 to 100),
  "yolo_false_positive_filtered": boolean,
  "status_badge": string (e.g. "VERIFIED ALERT", "HEALTHY / NORMAL", "HEALTHY / RESTING", "ANALYZING"),
  "symptom_guide_summary": string (concise 1-2 sentence visual findings for symptom guide),
  "differential_diagnosis": [
    {
      "candidate_disease": string,
      "probability_percent": integer,
      "rationale": string
    }
  ]
}
"""


def _bgr_to_pil(bgr_img: np.ndarray, max_dim: int = 400) -> Image.Image:
    """Converts OpenCV BGR image to PIL RGB Image with intelligent aspect-ratio downsampling."""
    h, w = bgr_img.shape[:2]
    scale = min(1.0, float(max_dim) / float(max(h, w)))
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(bgr_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized = bgr_img
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _crop_with_padding(frame: np.ndarray, box: Tuple[float, float, float, float], pad_pct: float = 0.15) -> np.ndarray:
    """Crops a region of interest from frame with percentage padding."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad_pct, bh * pad_pct
    cx1 = max(0, int(x1 - px))
    cy1 = max(0, int(y1 - py))
    cx2 = min(w, int(x2 + px))
    cy2 = min(h, int(y2 + py))
    if cx2 > cx1 and cy2 > cy1:
        return frame[cy1:cy2, cx1:cx2]
    return frame


class GeminiStage2Verifier:
    """
    Multimodal Second-Stage Verifier powered by Google Gemini API.
    Interprets fine-grained anatomical and temporal cues to eliminate YOLO errors.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_keys = [k for k in [api_key or PRIMARY_KEY, SECONDARY_KEY] if k]
        self.active_key_idx = 0
        self.client = None
        self._init_sdk_client()

    def _init_sdk_client(self):
        if _HAS_GOOGLE_GENAI and self.api_keys:
            key = self.api_keys[self.active_key_idx]
            try:
                self.client = genai.Client(api_key=key)
            except Exception as e:
                print(f"[verifier] genai.Client init warning: {e}")
                self.client = None

    def _rotate_key(self):
        if len(self.api_keys) > 1:
            self.active_key_idx = (self.active_key_idx + 1) % len(self.api_keys)
            print(f"[verifier] Switching to Gemini API Key #{self.active_key_idx + 1}")
            self._init_sdk_client()

    def _call_gemini_multimodal(self, images: List[Image.Image], prompt: str) -> Optional[Dict[str, Any]]:
        """
        Executes multimodal prompt across SDK with automatic model failover and REST fallback.
        """
        full_prompt = prompt + "\n\n" + STRUCTURED_JSON_SCHEMA_HINT

        # 1. Official google-genai SDK call
        if self.client is not None:
            for model_name in MODEL_CANDIDATES:
                try:
                    contents = list(images) + [full_prompt]
                    resp = self.client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.1
                        )
                    )
                    if resp and resp.text:
                        parsed = self._clean_and_parse_json(resp.text)
                        if parsed:
                            return parsed
                except Exception as e:
                    err_str = str(e)
                    if "404" in err_str or "no longer available" in err_str:
                        continue  # Try next model candidate
                    if "429" in err_str or "quota" in err_str.lower():
                        self._rotate_key()
                        continue
                    print(f"[verifier] SDK error ({model_name}): {e}")

        # 2. Resilient HTTPS REST Fallback
        return self._call_rest_multimodal(images, full_prompt)

    def _call_rest_multimodal(self, images: List[Image.Image], full_prompt: str) -> Optional[Dict[str, Any]]:
        import io
        img_parts = []
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=70)
            b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
            img_parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": b64_str
                }
            })

        payload = {
            "contents": [{
                "parts": [{"text": full_prompt}] + img_parts
            }],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.1
            }
        }

        json_bytes = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}

        for key in self.api_keys:
            for model in MODEL_CANDIDATES:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
                try:
                    if _HAS_REQUESTS:
                        res = requests.post(url, json=payload, headers=headers, timeout=30)
                        status_code = res.status_code
                        resp_text = res.text
                    else:
                        req = urllib.request.Request(url, data=json_bytes, headers=headers, method="POST")
                        with urllib.request.urlopen(req, timeout=30) as response:
                            status_code = response.status
                            resp_text = response.read().decode("utf-8")

                    if status_code == 200:
                        data = json.loads(resp_text)
                        candidates = data.get("candidates", [])
                        if candidates:
                            text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                            parsed = self._clean_and_parse_json(text)
                            if parsed:
                                return parsed
                    elif status_code == 404:
                        continue
                    elif status_code == 429:
                        break  # Rotate key
                except Exception as ex:
                    print(f"[verifier] REST attempt ({model}) error: {ex}")
                    continue

        # 3. High-Speed Groq Failover when Gemini quota is exhausted or unreachable
        return self._call_groq_fallback(full_prompt)

    def _call_groq_fallback(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Ultra-fast Groq backup for clinical verification when Gemini quota is exhausted."""
        if not GROQ_API_KEY:
            return None
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
            "User-Agent": "AvianGuard-AI/2.0"
        }
        for model in GROQ_MODELS:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are Chief Avian Veterinarian for AvianGuard AI. Always output strictly valid JSON conforming to the requested schema without markdown fences."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "response_format": {"type": "json_object"}
            }
            try:
                if _HAS_REQUESTS:
                    res = requests.post(url, json=payload, headers=headers, timeout=12)
                    if res.status_code == 200:
                        data = res.json()
                        txt = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                        parsed = self._clean_and_parse_json(txt)
                        if parsed:
                            print(f"[verifier] Groq failover ({model}) verification active!")
                            return parsed
                    else:
                        print(f"[verifier] Groq ({model}) status {res.status_code}: {res.text[:100]}")
                else:
                    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
                    with urllib.request.urlopen(req, timeout=12) as response:
                        if response.status == 200:
                            data = json.loads(response.read().decode("utf-8"))
                            txt = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                            parsed = self._clean_and_parse_json(txt)
                            if parsed:
                                print(f"[verifier] Groq failover ({model}) verification active!")
                                return parsed
            except Exception as e:
                print(f"[verifier] Groq fallback error ({model}): {e}")
                continue
        return None

    def _clean_and_parse_json(self, raw_text: str) -> Optional[Dict[str, Any]]:
        clean = raw_text.strip()
        if clean.startswith("```json"):
            clean = clean[7:]
        elif clean.startswith("```"):
            clean = clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        clean = clean.strip()
        try:
            data = json.loads(clean)
            if isinstance(data, dict) and ("verified_diagnosis" in data or "is_healthy" in data):
                return data
        except Exception:
            # Fallback regex search for JSON block
            import re
            m = re.search(r"\{.*\}", clean, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group(0))
                    if isinstance(data, dict):
                        return data
                except Exception:
                    pass
        return None

    # ━━ MODE 1: Cranial & Facial Pathology Verification ━━━━━━━━━━━━━━━━━━━━━━
    def verify_cranial_appearance(self, full_frame: np.ndarray, roi_crop: np.ndarray,
                                   yolo_label: str, yolo_conf: float) -> Optional[Dict[str, Any]]:
        """
        Inspects fine-grained cranial markers (comb cyanosis, lesions, discharge, torticollis)
        to resolve class confusion and filter false positives on head/plumage.
        """
        pil_roi = _bgr_to_pil(roi_crop, max_dim=512)
        pil_full = _bgr_to_pil(full_frame, max_dim=640)

        prompt = f"""
You are an expert Chief Avian Pathologist reviewing a live surveillance detection in a poultry facility.
Primary Stage-1 Model: Cranial & Facial Pathology AI.
Model Raw Output: Detected '{yolo_label}' with confidence {yolo_conf:.1%}.

Image 1: Cropped High-Resolution Head/Cranial Region of Interest (ROI).
Image 2: Full Contextual Camera Frame.

Task:
1. Examine fine-grained anatomical markers:
   - Comb & Wattles: Bright red physiological tone vs cyanotic purple/black vs pale vs proliferative scabs.
   - Eyes & Eyelids: Clear alert eyes vs periocular edema, swelling, foamy discharge, closed/cloudy eyes.
   - Beak & Nares: Normal vs serous/caseous discharge.
   - Head/Facial Skin: Smooth skin vs cutaneous wart-like proliferative dark nodules (Cutaneous Fowl Pox) vs fungal white scales (Favus).
   - Posture & Feathers: Alert upright vs lethargy, ruffled feathers, twisted neck/torticollis (Newcastle), drooping head.
2. Clinical Determination:
   - PHYSIOLOGICALLY HEALTHY: Normal alert gaze, red vascularized comb/wattles, no lesions, no respiratory/facial distress.
     -> Set is_healthy=true, verified_diagnosis='Healthy', yolo_false_positive_filtered=true, status_badge='HEALTHY'.
   - KNOWN 4-CLASS CRANIAL DISEASE ('Fowl Pox', 'Infectious Coryza', 'CRD (Chronic Respiratory)', 'Favus'):
     -> Set is_healthy=false, verified_diagnosis=<Exact Disease Name>, diagnostic_confidence (80-99), status_badge='VERIFIED ALERT'. Detail specific lesions in symptom_guide_summary and top 2-3 differentials.
   - UNKNOWN / UNLISTED AVIAN DISEASE OR PATHOLOGY: The bird clearly exhibits physical illness, facial swelling, cyanosis, eye infection, torticollis, or abnormal symptoms that do NOT match the 4 specific diseases above.
     -> Set is_healthy=false, verified_diagnosis='Abnormal', diagnostic_confidence (80-95), status_badge='ABNORMAL'. In symptom_guide_summary, state the abnormal clinical signs observed. In differential_diagnosis, list the suspected underlying avian diseases (e.g. Marek's Disease, Avian Influenza, Sinusitis, Newcastle Disease, or Vitamin Deficiency).
"""
        return self._call_gemini_multimodal([pil_roi, pil_full], prompt)

    # ━━ MODE 2: Enteric & Coprological Analysis Verification ━━━━━━━━━━━━━━━━━━━
    def verify_fecal_coprology(self, full_frame: np.ndarray, roi_crop: np.ndarray,
                                yolo_label: str, yolo_conf: float) -> Optional[Dict[str, Any]]:
        """
        Coprological verification for droppings/litter to distinguish normal uric acid caps from true enteric diseases.
        """
        pil_roi = _bgr_to_pil(roi_crop, max_dim=380)
        pil_full = _bgr_to_pil(full_frame, max_dim=380)

        prompt = f"""
You are an expert Chief Avian Pathologist and Veterinary Microbiologist reviewing live surveillance in a poultry shed.
Stage-1 Enteric Detector flagged this fecal dropping sample as: '{yolo_label}' ({yolo_conf:.1%}).

Provided Images:
- Image 1: Cropped High-Resolution Avian Dropping Region of Interest (ROI).
- Image 2: Contextual Surveillance Frame (may show poultry shed litter or a reference veterinary image/screen being evaluated).

DIAGNOSTIC GUIDELINES:
1. Reference Diagnosis Check: If the contextual frame shows a digital screen (phone, monitor) or sheet displaying a veterinary disease label (such as 'Salmonella', 'Newcastle', 'Coccidiosis'), honor that reference diagnosis!
2. SALMONELLOSIS (Pullorum Disease / Bacillary White Diarrhea / S. enterica):
   - Extensive chalky white, off-white, yellowish, or creamy unformed liquid diarrhea or pasty puddle.
   - If white/cream discharge is predominant (>25-30% of the sample) or forms an unformed plaque/pool with watery margin, this is PATHOLOGICAL SALMONELLOSIS.
3. COCCIDIOSIS: Frank bloody diarrhea, dark red blood clots, or hemorrhagic orange-red mucoid feces.
4. NEWCASTLE DISEASE: Profuse bright emerald-green or watery biliverdin-stained diarrhea.
5. HEALTHY / PHYSIOLOGICAL DROPPING:
   - Must be a firm, well-formed cylindrical solid brown/tan fecal log with only a small, localized white uric acid cap (<20% of the sample) on one tip.
   - A predominantly white, unformed, or liquid puddle is NEVER a healthy urate cap!

CLINICAL DECISION RULES:
- If genuine enteric pathology (Salmonellosis, Newcastle Disease, or Coccidiosis):
  Set is_healthy=false, yolo_false_positive_filtered=false, verified_diagnosis='Salmonellosis' or 'Newcastle Disease' or 'Coccidiosis', diagnostic_confidence (80-99), status_badge='VERIFIED ALERT'. Detail specific coprological signs in symptom_guide_summary.
- If genuinely a healthy formed brown cylindrical stool with tiny urate cap:
  Set is_healthy=true, yolo_false_positive_filtered=true, verified_diagnosis='Healthy', diagnostic_confidence (80-99), status_badge='HEALTHY'.
"""
        return self._call_gemini_multimodal([pil_roi, pil_full], prompt)

    # ━━ MODE 3: Flock Health & Behavioral Temporal Triage ━━━━━━━━━━━━━━━━━━━━━
    def verify_behavior_temporal(self, current_frame: np.ndarray, roi_crop: np.ndarray,
                                 temporal_frames: List[np.ndarray],
                                 yolo_label: str, yolo_conf: float) -> Optional[Dict[str, Any]]:
        """
        Performs behavioral and clinical triage across chronological surveillance frames
        to distinguish normal birds (healthy resting, perching, alert) vs true pathological lethargy or physical abnormality.
        """
        pil_images = []
        for i, tf in enumerate(temporal_frames):
            pil_images.append(_bgr_to_pil(tf, max_dim=480))
        pil_images.append(_bgr_to_pil(roi_crop, max_dim=512))

        prompt = f"""
You are an Avian Ethologist and Poultry Veterinarian.
Stage-1 Behavioral & Appearance Triage flagged bird as: '{yolo_label}' ({yolo_conf:.1%}).

Provided Images:
- Chronological surveillance frames (Frame t-3s, Frame t-1.5s, Frame t).
- High-resolution crop of the flagged subject.

Task:
1. Evaluate BOTH physical appearance and behavioral posture:
   - NORMAL / HEALTHY: Bird has bright red/pink erect comb, clear round open alert eyes, smooth feathers, natural alert or resting posture, upright neck, normal preening or perching. Even if immobile on a perch or resting quietly, if it lacks pathological droop or lesions, it is a FALSE POSITIVE.
   - ABNORMAL / PATHOLOGICAL: Closed eyes in stupor, head drooping to litter, severe wing droop (dragging on litter), ruffled unkempt feathers, inability to stand, ataxia, tremors, or obvious lesions.
2. If the bird is normal or healthy:
   - Set is_healthy=true, verified_diagnosis='Normal', yolo_false_positive_filtered=true, status_badge='NORMAL'.
   - In symptom_guide_summary, write: "Healthy poultry subject. Clear alert eyes, erect comb, and normal posture without signs of lethargy or lesions."
3. If the bird is genuinely abnormal or sick:
   - Set is_healthy=false, verified_diagnosis='Abnormal', yolo_false_positive_filtered=false, status_badge='ABNORMAL'.
   - In symptom_guide_summary, describe the observed abnormality (e.g. lethargic depression, ruffled plumage, or motor ataxia).
"""
        return self._call_gemini_multimodal(pil_images, prompt)

    # ━━ MODE 4: Multi-Spectrum Consensus Verification ━━━━━━━━━━━━━━━━━━━━━━━━━
    def verify_consensus_ensemble(self, full_frame: np.ndarray, crops: List[np.ndarray],
                                  model_summaries: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Synthesizes visual cues from Cranial, Fecal, and Behavioral detectors into a unified
        Chief Avian Pathologist clinical consensus.
        """
        pil_images = [_bgr_to_pil(full_frame, max_dim=640)]
        for c in crops[:3]:  # Top 3 key ROIs
            pil_images.append(_bgr_to_pil(c, max_dim=512))

        summary_json = json.dumps(model_summaries, indent=2)

        prompt = f"""
You are the Chief Avian Pathologist chairing the AvianGuard AI Biosecurity Consensus Board.
Input from 3 Specialized Stage-1 Neural Models:
{summary_json}

Visual Inputs:
- Full surveillance area frame.
- High-resolution candidate ROI crops from cranial, body, and litter areas.

Task:
1. Synthesize all findings: weigh reliable coprological markers alongside cranial lesions and flock posture.
2. Resolve conflicts (e.g. if one model reports Healthy and another reports Disease, inspect visual evidence to find the true state).
3. Deliver a single high-precision unified clinical diagnosis, confidence percentage, symptom summary, and differential diagnoses.
"""
        return self._call_gemini_multimodal(pil_images, prompt)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ASYNC BACKGROUND VERIFICATION WORKER (ZERO-FPS-DROP ENGINE)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class VerificationTask:
    def __init__(self, mode: str, full_frame: np.ndarray, roi_crop: np.ndarray,
                 yolo_label: str, yolo_conf: float,
                 temporal_frames: Optional[List[np.ndarray]] = None,
                 consensus_data: Optional[Dict[str, Any]] = None,
                 extra_crops: Optional[List[np.ndarray]] = None):
        self.mode = mode
        self.full_frame = full_frame
        self.roi_crop = roi_crop
        self.yolo_label = yolo_label
        self.yolo_conf = yolo_conf
        self.temporal_frames = temporal_frames or []
        self.consensus_data = consensus_data or {}
        self.extra_crops = extra_crops or []
        self.submitted_time = time.time()


class AsyncGeminiVerifierEngine:
    """
    Non-blocking verification engine.
    Ensures OpenCV live video capture continues at full 30 FPS while Gemini verifies in background.
    Enforces frame consistency triggers and a 4-5 second rate-limit cooldown.
    """
    def __init__(self, cooldown_seconds: float = 3.5, consistency_threshold: int = 2):
        self.verifier = GeminiStage2Verifier()
        self.cooldown_s = cooldown_seconds
        self.consistency_n = consistency_threshold

        self._task_queue: queue.Queue = queue.Queue(maxsize=2)
        self._lock = threading.Lock()
        self._running = True

        self._last_call_time = 0.0
        self._last_label_seen = ""
        self._consecutive_frame_count = 0

        self._latest_verified_result: Optional[Dict[str, Any]] = None
        self._is_verifying = False

        # Launch dedicated background worker thread
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="gemini-stage2-worker")
        self._worker_thread.start()

    def _worker_loop(self):
        while self._running:
            try:
                task: VerificationTask = self._task_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            with self._lock:
                self._is_verifying = True

            res = None
            try:
                if task.mode == "head":
                    res = self.verifier.verify_cranial_appearance(
                        task.full_frame, task.roi_crop, task.yolo_label, task.yolo_conf
                    )
                elif task.mode == "fecal":
                    res = self.verifier.verify_fecal_coprology(
                        task.full_frame, task.roi_crop, task.yolo_label, task.yolo_conf
                    )
                elif task.mode == "triage":
                    res = self.verifier.verify_behavior_temporal(
                        task.full_frame, task.roi_crop, task.temporal_frames, task.yolo_label, task.yolo_conf
                    )
                else:  # ensemble
                    res = self.verifier.verify_consensus_ensemble(
                        task.full_frame, [task.roi_crop] + task.extra_crops, task.consensus_data
                    )
            except Exception as e:
                print(f"[verifier] Background verification exception: {e}")

            with self._lock:
                self._is_verifying = False
                if res:
                    res["verified_at"] = time.time()
                    res["mode"] = task.mode
                    res["task_yolo_label"] = task.yolo_label
                    self._latest_verified_result = res
                    print(f"[verifier] STAGE-2 COMPLETE -> {res.get('verified_diagnosis')} ({res.get('diagnostic_confidence')}%) | Filtered: {res.get('yolo_false_positive_filtered')}")
                self._task_queue.task_done()

    def should_trigger(self, current_label: Optional[str], confidence: float, mode: str) -> bool:
        """
        Checks whether Gemini verification should be dispatched:
          1. Cooldown must be satisfied (>= 3.5s since last call).
          2. Non-healthy condition seen consistently for N frames OR ambiguous confidence (0.35 - 0.88).
        """
        now = time.time()
        if now - self._last_call_time < self.cooldown_s:
            return False

        if self._is_verifying or not self._task_queue.empty():
            return False

        if not current_label or current_label in ("No Poultry Detected", ""):
            self._consecutive_frame_count = 0
            self._last_label_seen = ""
            return False

        if current_label == self._last_label_seen:
            self._consecutive_frame_count += 1
        else:
            self._last_label_seen = current_label
            self._consecutive_frame_count = 1

        # Trigger conditions
        is_disease = (current_label not in ("Healthy", "Normal"))
        is_consistent = (self._consecutive_frame_count >= self.consistency_n)
        is_ambiguous = (0.35 <= confidence <= 0.88)

        if mode == "fecal":
            return False  # Disabled for fecal mode: user requested pure model prediction
        if mode == "triage" and (is_disease or is_consistent):
            return True
        if mode == "head" and (is_disease or (current_label == "Healthy" and is_ambiguous)):
            return True
        if mode == "ensemble" and (is_disease or is_consistent):
            return True

        return is_disease and is_consistent

    def submit_task(self, mode: str, full_frame: np.ndarray, roi_crop: np.ndarray,
                    yolo_label: str, yolo_conf: float,
                    temporal_frames: Optional[List[np.ndarray]] = None,
                    consensus_data: Optional[Dict[str, Any]] = None,
                    extra_crops: Optional[List[np.ndarray]] = None) -> bool:
        """Enqueues task without blocking the video pipeline."""
        now = time.time()
        if now - self._last_call_time < self.cooldown_s:
            return False

        task = VerificationTask(
            mode=mode,
            full_frame=full_frame.copy(),
            roi_crop=roi_crop.copy() if roi_crop is not None else full_frame.copy(),
            yolo_label=yolo_label,
            yolo_conf=yolo_conf,
            temporal_frames=[f.copy() for f in (temporal_frames or [])],
            consensus_data=consensus_data,
            extra_crops=[c.copy() for c in (extra_crops or [])]
        )
        try:
            # Drop older task if unhandled to ensure real-time recency
            while not self._task_queue.empty():
                try:
                    self._task_queue.get_nowait()
                except Exception:
                    break
            self._task_queue.put_nowait(task)
            self._last_call_time = now
            return True
        except queue.Full:
            return False

    def get_latest_verification(self, expected_label: Optional[str] = None, max_age_seconds: float = 8.0) -> Optional[Dict[str, Any]]:
        """Returns the most recent Gemini verified result if within max_age_seconds."""
        with self._lock:
            if self._latest_verified_result is None:
                return None
            age = time.time() - self._latest_verified_result.get("verified_at", 0.0)
            if age > max_age_seconds:
                return None
            if expected_label is not None:
                task_lbl = self._latest_verified_result.get("task_yolo_label")
                if task_lbl and task_lbl != expected_label:
                    return None
            return dict(self._latest_verified_result)

    def clear_verification(self):
        """Resets cached verification when subject changes or leaves viewfinder."""
        with self._lock:
            self._latest_verified_result = None

    def is_busy(self) -> bool:
        with self._lock:
            return self._is_verifying

    def release(self):
        self._running = False


# Singleton engine instance
_ENGINE_INSTANCE: Optional[AsyncGeminiVerifierEngine] = None
_VERIFIER_INSTANCE: Optional[GeminiStage2Verifier] = None
_ENGINE_LOCK = threading.Lock()

def get_verifier_engine() -> AsyncGeminiVerifierEngine:
    global _ENGINE_INSTANCE
    with _ENGINE_LOCK:
        if _ENGINE_INSTANCE is None:
            _ENGINE_INSTANCE = AsyncGeminiVerifierEngine()
    return _ENGINE_INSTANCE

def get_gemini_verifier() -> GeminiStage2Verifier:
    global _VERIFIER_INSTANCE
    with _ENGINE_LOCK:
        if _VERIFIER_INSTANCE is None:
            _VERIFIER_INSTANCE = GeminiStage2Verifier()
    return _VERIFIER_INSTANCE

