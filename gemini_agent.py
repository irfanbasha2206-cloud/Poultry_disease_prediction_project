"""
gemini_agent.py -- Intelligent AvianGuard AI Agent powered by Google Gemini.
Responsibilities:
  1. Multimodal Subject Gating: Evaluates uploaded images and strictly accepts ONLY
     poultry chickens/birds or chicken fecal matter. Rejects human hands, faces, doors,
     furniture, pets, and non-avian items.
  2. Perspective Model Routing: Routes valid birds to Cranial/Body models, and droppings
     to Coprological/Fecal models.
  3. Clinical Description & Treatment Synthesis: Generates rich veterinary insights,
     quarantine protocols, and medical treatments.
  4. Email Alert Dispatcher: Sends automated email alerts via Gmail SMTP when diseases are diagnosed.
"""
from __future__ import annotations

import base64
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import time
from typing import Optional, Tuple, Dict, Any

import cv2
import os

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

import urllib.request
import urllib.error

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

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_API_KEY_SECONDARY = os.environ.get("GEMINI_API_KEY_SECONDARY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

PRIMARY_MODELS = [
    os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite"),
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
]

GROQ_MODELS = [
    os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b"),
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]

SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", 465))
SMTP_SENDER = os.environ.get("SMTP_SENDER", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")


class PoultryAIAgent:
    def __init__(self, api_key: str = GEMINI_API_KEY):
        self.gemini_keys = [k for k in [api_key or GEMINI_API_KEY, GEMINI_API_KEY_SECONDARY] if k]
        self.groq_key = GROQ_API_KEY
        self.models = PRIMARY_MODELS
        self.groq_models = GROQ_MODELS
        self.session = requests.Session() if _HAS_REQUESTS else None

    def _call_groq(self, prompt: str, json_mode: bool = True) -> Optional[str]:
        """High-speed fallback reasoning engine powered by Groq API."""
        if not self.groq_key:
            return None
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.groq_key}",
            "Content-Type": "application/json",
            "User-Agent": "AvianGuard-AI/2.0"
        }
        for model in self.groq_models:
            payload: Dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are Chief Avian Veterinarian for AvianGuard AI. Always output strictly valid JSON conforming to the requested schema without markdown fences."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1
            }
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
            try:
                if _HAS_REQUESTS and self.session is not None:
                    resp = self.session.post(url, json=payload, headers=headers, timeout=12)
                    if resp.status_code == 200:
                        data = resp.json()
                        choices = data.get("choices", [])
                        if choices:
                            txt = choices[0].get("message", {}).get("content", "").strip()
                            if txt:
                                print(f"[agent] Groq failover ({model}) succeeded!")
                                return txt
                    else:
                        print(f"[agent] Groq ({model}) status {resp.status_code}: {resp.text[:100]}")
                else:
                    json_bytes = json.dumps(payload).encode("utf-8")
                    req = urllib.request.Request(url, data=json_bytes, headers=headers, method="POST")
                    with urllib.request.urlopen(req, timeout=12) as response:
                        if response.status == 200:
                            data = json.loads(response.read().decode("utf-8"))
                            choices = data.get("choices", [])
                            if choices:
                                txt = choices[0].get("message", {}).get("content", "").strip()
                                if txt:
                                    print(f"[agent] Groq failover ({model}) succeeded!")
                                    return txt
            except Exception as e:
                print(f"[agent] Groq error ({model}): {e}")
                continue
        return None

    def _call_gemini(self, prompt: str, image_bgr: Optional[np.ndarray] = None, json_mode: bool = True) -> Optional[str]:
        """Calls Gemini API with multi-key failover and seamless Groq backup trigger."""
        img_parts = []
        if image_bgr is not None and image_bgr.size > 0:
            h, w = image_bgr.shape[:2]
            scale = min(1.0, 512.0 / max(h, w))
            if scale < 1.0:
                resized = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))
            else:
                resized = image_bgr
            _, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
            img_b64 = base64.b64encode(buf).decode("utf-8")
            img_parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": img_b64
                }
            })

        payload: Dict[str, Any] = {
            "contents": [{
                "parts": [{"text": prompt}] + img_parts
            }]
        }
        if json_mode:
            payload["generationConfig"] = {"response_mime_type": "application/json"}

        # 1. Try Gemini across all available API keys & verified models
        for key in self.gemini_keys:
            for model_name in self.models:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"
                try:
                    if _HAS_REQUESTS and self.session is not None:
                        resp = self.session.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=12)
                        if resp.status_code == 200:
                            data = resp.json()
                            candidates = data.get("candidates", [])
                            if candidates:
                                parts = candidates[0].get("content", {}).get("parts", [])
                                if parts:
                                    return parts[0].get("text", "")
                        elif resp.status_code in (429, 503):
                            continue
                        else:
                            print(f"[agent] Gemini ({model_name}) returned status {resp.status_code}: {resp.text[:120]}")
                    else:
                        json_bytes = json.dumps(payload).encode("utf-8")
                        req = urllib.request.Request(url, data=json_bytes, headers={"Content-Type": "application/json"}, method="POST")
                        try:
                            with urllib.request.urlopen(req, timeout=12) as response:
                                if response.status == 200:
                                    data = json.loads(response.read().decode("utf-8"))
                                    candidates = data.get("candidates", [])
                                    if candidates:
                                        parts = candidates[0].get("content", {}).get("parts", [])
                                        if parts:
                                            return parts[0].get("text", "")
                        except urllib.error.HTTPError as he:
                            if he.code in (429, 503):
                                continue
                            print(f"[agent] Gemini ({model_name}) HTTP error {he.code}: {he}")
                except Exception as e:
                    print(f"[agent] Error querying {model_name}: {e}")
                    continue

        # 2. Automatic Groq Fallback if Gemini quota is exhausted or unreachable
        print("[agent] Gemini quota/rate-limits reached. Triggering Groq backup reasoning engine...")
        return self._call_groq(prompt, json_mode=json_mode)

    def evaluate_upload(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Agent Step 1: Evaluates whether the uploaded image is suitable for poultry disease detection.
        Strictly requires real chicken/poultry bird OR chicken fecal droppings.
        """
        if image_bgr is None or image_bgr.size == 0:
            return {
                "is_poultry": False,
                "category": "NON_POULTRY",
                "reason": "Empty or invalid image data.",
                "routed_model": None
            }

        prompt = (
            "You are the AvianGuard AI Gatekeeper Agent specializing in poultry veterinary health.\n"
            "Examine this uploaded image and determine if it belongs to ONE of the following valid project categories:\n"
            "1. 'CHICKEN_BIRD': A chicken, rooster, hen, chick, or poultry flock (head, comb, wattle, eye, plumage, or full bird).\n"
            "2. 'CHICKEN_FECAL': Chicken or poultry droppings / fecal matter.\n"
            "3. 'NON_POULTRY': ANY OTHER SUBJECT. This includes human faces, human hands, human bodies, wooden doors, "
            "furniture, dogs, cats, cars, plants, landscapes, or household objects.\n\n"
            "CRITICAL INSTRUCTION: You must strictly reject non-poultry subjects. Even if an object is red or textured, "
            "if it is NOT an actual poultry bird or droppings, classify as 'NON_POULTRY'.\n\n"
            "Respond strictly in JSON format:\n"
            "{\n"
            '  "is_poultry": true,\n'
            '  "category": "CHICKEN_BIRD",\n'
            '  "confidence": 0.98,\n'
            '  "reason": "Clear explanation of what was detected",\n'
            '  "routed_model": "head"\n'
            "}\n"
            "Where category is CHICKEN_BIRD (routed_model='head'), CHICKEN_FECAL (routed_model='fecal'), or NON_POULTRY (is_poultry=false, routed_model=null)."
        )

        res_text = self._call_gemini(prompt, image_bgr, json_mode=True)
        if res_text:
            try:
                res_json = json.loads(res_text)
                category = res_json.get("category", "NON_POULTRY").upper()
                is_poultry = bool(res_json.get("is_poultry", False)) and category in ("CHICKEN_BIRD", "CHICKEN_FECAL")
                routed = "head" if category == "CHICKEN_BIRD" else ("fecal" if category == "CHICKEN_FECAL" else None)
                return {
                    "is_poultry": is_poultry,
                    "category": category,
                    "confidence": float(res_json.get("confidence", 0.95)),
                    "reason": res_json.get("reason", "Subject evaluated by AvianGuard AI Agent."),
                    "routed_model": routed
                }
            except Exception as exc:
                print(f"[agent] Error parsing Gemini JSON: {exc}")

        # Resilient fallback: If Gemini API is unreachable, use local multi-spectral poultry validator
        try:
            from predictor import PoultryPredictor
            pred = PoultryPredictor()
            is_valid, reason = pred.validate_poultry_subject(image_bgr, [image_bgr])
            return {
                "is_poultry": is_valid,
                "category": "CHICKEN_BIRD" if is_valid else "NON_POULTRY",
                "confidence": 0.90 if is_valid else 0.98,
                "reason": f"AvianGuard Fallback Validator: {reason}",
                "routed_model": "head" if is_valid else None
            }
        except Exception:
            pass

        return {
            "is_poultry": False,
            "category": "NON_POULTRY",
            "confidence": 0.50,
            "reason": "Agent service offline or image verification failed.",
            "routed_model": None
        }

    def generate_clinical_consultation(self, disease_label: str, confidence: float, category: str, image_bgr: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Agent Step 2: Synthesizes expert veterinary descriptions, pathology notes,
        and biosecurity recommendations based on model classification.
        """
        prompt = (
            f"You are the Chief Avian Veterinarian and Diagnostic Agent for AvianGuard AI.\n"
            f"Our specialized neural network detected: '{disease_label}' with {confidence:.1%} confidence on a {category} subject.\n"
            "Generate a comprehensive, actionable veterinary diagnostic report strictly in valid JSON format:\n"
            "{\n"
            '  "diagnosis_title": "Disease Name & Clinical Subtype",\n'
            '  "pathology_overview": "Detailed description of the condition, causative pathogen (viral/bacterial/parasitic), and progression.",\n'
            '  "clinical_symptoms": ["Symptom 1", "Symptom 2", "Symptom 3"],\n'
            '  "biosecurity_protocol": "Immediate quarantine and coop disinfection guidelines to prevent flock outbreak.",\n'
            '  "treatment_regimen": "Standard veterinary intervention, medication, antiseptic care, or dietary supplements.",\n'
            '  "urgency_level": "High" or "Critical" or "Moderate" or "Normal"\n'
            "}"
        )

        res_text = self._call_gemini(prompt, image_bgr=image_bgr, json_mode=True)
        if res_text:
            try:
                return json.loads(res_text)
            except Exception as e:
                print(f"[agent] Failed to parse consultation JSON: {e}")

        # Fallback dictionary
        return {
            "diagnosis_title": disease_label,
            "pathology_overview": f"Automated avian pathology detected {disease_label} ({confidence:.0%}).",
            "clinical_symptoms": ["Monitor flock for lethargy, respiratory distress, or abnormal lesions."],
            "biosecurity_protocol": "Isolate the affected bird in a separate warm pen with dedicated water and feeders.",
            "treatment_regimen": "Consult a registered avian veterinarian for targeted antibiotic or supportive therapy.",
            "urgency_level": "High" if disease_label != "Healthy" else "Normal"
        }

    @staticmethod
    def send_email_alert(
        recipient_email: str,
        disease_label: str,
        confidence: float,
        consultation: Dict[str, Any],
        image_bgr: Optional[np.ndarray] = None
    ) -> Tuple[bool, str]:
        """
        Dispatches an automated diagnostic email alert to the specified recipient via Gmail SMTP.
        """
        if not recipient_email or "@" not in recipient_email:
            return False, "Invalid recipient email address."

        try:
            msg = MIMEMultipart("related")
            msg["Subject"] = f"🚨 AvianGuard AI Alert: {disease_label} Detected ({confidence:.0%})"
            msg["From"] = f"AvianGuard AI Surveillance <{SMTP_SENDER}>"
            msg["To"] = recipient_email

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            urgency = consultation.get("urgency_level", "High")
            color = "#ef4444" if urgency in ("Critical", "High") else "#f59e0b"

            symptoms_html = "".join([f"<li style='margin-bottom:4px;'>{s}</li>" for s in consultation.get("clinical_symptoms", [])])

            html = f"""
            <html>
            <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #0b0f17; color: #f1f5f9; padding: 24px;">
              <div style="max-width: 600px; margin: 0 auto; background: #131b2e; border: 1px solid rgba(255,255,255,0.12); border-radius: 14px; overflow: hidden; box-shadow: 0 10px 30px rgba(0,0,0,0.5);">
                
                <div style="background: linear-gradient(135deg, #1e293b, #0f172a); padding: 20px 24px; border-bottom: 2px solid {color};">
                  <div style="font-size: 13px; font-weight: 700; color: #00f0a8; letter-spacing: 1.5px; text-transform: uppercase;">AvianGuard AI Diagnostics</div>
                  <h1 style="margin: 8px 0 0; font-size: 24px; color: #ffffff;">Automated Disease Alert</h1>
                </div>

                <div style="padding: 24px;">
                  <div style="display: flex; align-items: center; justify-content: space-between; background: rgba(255,255,255,0.04); border-radius: 8px; padding: 14px 18px; margin-bottom: 20px;">
                    <div>
                      <div style="font-size: 12px; color: #94a3b8; text-transform: uppercase;">Condition Diagnosed</div>
                      <div style="font-size: 20px; font-weight: 800; color: {color}; margin-top: 2px;">{disease_label}</div>
                    </div>
                    <div style="text-align: right;">
                      <div style="font-size: 12px; color: #94a3b8; text-transform: uppercase;">Confidence</div>
                      <div style="font-size: 20px; font-weight: 800; color: #ffffff; margin-top: 2px;">{confidence:.1%}</div>
                    </div>
                  </div>

                  <div style="margin-bottom: 18px;">
                    <h3 style="font-size: 14px; color: #00f0a8; text-transform: uppercase; margin-bottom: 6px;">Pathology Overview</h3>
                    <p style="font-size: 14px; line-height: 1.6; color: #cbd5e1; margin: 0;">
                      {consultation.get("pathology_overview", "N/A")}
                    </p>
                  </div>

                  {f'''<div style="margin-bottom: 18px;">
                    <h3 style="font-size: 14px; color: #38bdf8; text-transform: uppercase; margin-bottom: 6px;">Observed Clinical Symptoms</h3>
                    <ul style="font-size: 14px; color: #cbd5e1; padding-left: 20px; margin: 0;">
                      {symptoms_html}
                    </ul>
                  </div>''' if symptoms_html else ''}

                  <div style="background: rgba(239, 68, 68, 0.1); border-left: 4px solid #ef4444; padding: 12px 16px; border-radius: 4px; margin-bottom: 18px;">
                    <h3 style="font-size: 13px; color: #f87171; text-transform: uppercase; margin: 0 0 4px;">Biosecurity & Quarantine Directive</h3>
                    <p style="font-size: 13px; line-height: 1.5; color: #fecaca; margin: 0;">
                      {consultation.get("biosecurity_protocol", "Isolate affected birds immediately.")}
                    </p>
                  </div>

                  <div style="background: rgba(16, 185, 129, 0.1); border-left: 4px solid #10b981; padding: 12px 16px; border-radius: 4px; margin-bottom: 20px;">
                    <h3 style="font-size: 13px; color: #34d399; text-transform: uppercase; margin: 0 0 4px;">Treatment & Intervention</h3>
                    <p style="font-size: 13px; line-height: 1.5; color: #a7f3d0; margin: 0;">
                      {consultation.get("treatment_regimen", "Consult veterinarian.")}
                    </p>
                  </div>

                  <div style="font-size: 12px; color: #64748b; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 14px; display: flex; justify-content: space-between;">
                    <span>Generated: {timestamp}</span>
                    <span>Host: AvianGuard Raspberry Pi Edge Node</span>
                  </div>
                </div>
              </div>
            </body>
            </html>
            """

            msg.attach(MIMEText(html, "html"))

            if image_bgr is not None and image_bgr.size > 0:
                _, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                img_part = MIMEImage(buf.tobytes(), name=f"{disease_label.replace(' ', '_')}.jpg")
                img_part.add_header("Content-Disposition", "attachment", filename=f"{disease_label.replace(' ', '_')}.jpg")
                msg.attach(img_part)

            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=10) as server:
                server.login(SMTP_SENDER, SMTP_PASSWORD)
                server.send_message(msg)

            print(f"[agent] Alert email successfully sent to {recipient_email} for {disease_label}")
            return True, f"Alert email successfully dispatched to {recipient_email}."
        except Exception as exc:
            print(f"[agent] Failed to send email alert: {exc}")
            return False, f"Failed to send email alert: {str(exc)}"


# Global singleton instance
_agent_instance: Optional[PoultryAIAgent] = None

def get_poultry_agent() -> PoultryAIAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = PoultryAIAgent()
    return _agent_instance
