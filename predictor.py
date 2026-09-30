"""
predictor.py -- Poultry disease inference engine with MODE-aware prediction,
presence verification (eliminates false positives on empty walls/screens),
proper NMS bounding box filtering, and clinical veterinary advisory data.

MODES:
  head     -> general.onnx only  (Favus, Fowl Pox, Coryza, CRD, Weak Leg, Normal)
  fecal    -> fecal.onnx only    (Coccidiosis, Newcastle, Salmonellosis, Healthy)
  triage   -> appearance.onnx + behavior.onnx  (Abnormal / Healthy status)
  ensemble -> all 4 models, domain-aware aggregation
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

try:
    import onnxruntime as ort
    _ORT_OK = True
except ImportError:
    ort = None
    _ORT_OK = False

try:
    from ultralytics import YOLO as _YOLO
    _YOLO_OK = True
except ImportError:
    _YOLO = None
    _YOLO_OK = False

try:
    import torch as _torch
    _TORCH_OK = True
except ImportError:
    _torch = None
    _TORCH_OK = False

_BASE_DIR           = os.path.dirname(os.path.abspath(__file__))
_OD_DIR             = os.path.join(_BASE_DIR, "Object_Detection_models")
MODEL_DIR           = _OD_DIR if (os.path.isdir(_OD_DIR) and len(os.listdir(_OD_DIR)) > 0) else os.path.join(_BASE_DIR, "models")
CLF_MODEL_DIR       = os.path.join(_BASE_DIR, "classification_models")
DETECT_CONF         = 0.35
DISEASE_CONF_THRESH = 0.45
TRIAGE_CONF_THRESH  = 0.68
INFER_IMGSZ         = 416

HEALTHY_LABEL      = "Healthy"
ABNORMAL_LABEL     = "Abnormal"
UNIDENTIFIED_LABEL = "Unhealthy - Condition Unidentified"

CANONICAL_LABELS = {
    "chicken favus":       "Favus",
    "fowl pox":            "Fowl Pox",
    "fowlpox":             "Fowl Pox",
    "bumblefoot":          "Bumblefoot",
    "coryza":              "Infectious Coryza",
    "crd":                 "CRD (Chronic Respiratory)",
    "weak_leg":            "Weak Leg Syndrome",
    "normal":              HEALTHY_LABEL,
    "coccidiosis-like":    "Coccidiosis",
    "coccidiosis":         "Coccidiosis",
    "ncd-like":            "Newcastle Disease",
    "new castle disease":  "Newcastle Disease",
    "newcastle disease":   "Newcastle Disease",
    "newcastledisease":    "Newcastle Disease",
    "salmonella-like":     "Salmonellosis",
    "salmonella":          "Salmonellosis",
    "salmonellosis":       "Salmonellosis",
    "healthy":             HEALTHY_LABEL,
    "abnormal":            ABNORMAL_LABEL,
}

GENERAL_DISEASES = {"Favus", "Fowl Pox", "Infectious Coryza",
                    "CRD (Chronic Respiratory)", "Weak Leg Syndrome", "Bumblefoot"}
FECAL_DISEASES   = {"Coccidiosis", "Newcastle Disease", "Salmonellosis"}
NAMED_DISEASES   = GENERAL_DISEASES | FECAL_DISEASES

LABEL_COLORS = {
    HEALTHY_LABEL:                ( 0, 215,  80),
    "Coccidiosis":                ( 0, 140, 255),
    "Newcastle Disease":          (255,  60,  60),
    "Salmonellosis":              (255, 185,   0),
    "Fowl Pox":                   (255, 120,   0),
    "Favus":                      (180, 105, 255),
    "Infectious Coryza":          (  0, 200, 200),
    "CRD (Chronic Respiratory)":  (100, 180, 255),
    "Weak Leg Syndrome":          (234, 179,   8),
    "Bumblefoot":                 (245, 158,  11),
    ABNORMAL_LABEL:               (200,   0, 200),
    UNIDENTIFIED_LABEL:           (220,  50,  50),
}
DEFAULT_COLOR = (255, 255, 255)

LABEL_HEX = {
    HEALTHY_LABEL:                "#00e5a0",
    "Coccidiosis":                "#3b82f6",
    "Newcastle Disease":          "#ff4757",
    "Salmonellosis":              "#f59e0b",
    "Fowl Pox":                   "#f97316",
    "Favus":                      "#a855f7",
    "Infectious Coryza":          "#06b6d4",
    "CRD (Chronic Respiratory)":  "#60a5fa",
    "Weak Leg Syndrome":          "#eab308",
    "Bumblefoot":                 "#f59e0b",
    ABNORMAL_LABEL:               "#ec4899",
    UNIDENTIFIED_LABEL:           "#dc2626",
}

SEVERITY_META = {
    HEALTHY_LABEL:                ("Normal",   "#00e5a0", "low"),
    "Weak Leg Syndrome":          ("Moderate", "#eab308", "med"),
    "Favus":                      ("Moderate", "#a855f7", "med"),
    "Bumblefoot":                 ("Moderate", "#f59e0b", "med"),
    "Fowl Pox":                   ("Elevated", "#f97316", "high"),
    "Infectious Coryza":          ("Elevated", "#06b6d4", "high"),
    "CRD (Chronic Respiratory)":  ("Elevated", "#60a5fa", "high"),
    "Coccidiosis":                ("Critical", "#ff4757", "crit"),
    "Newcastle Disease":          ("Critical", "#ff3c3c", "crit"),
    "Salmonellosis":              ("Critical", "#f59e0b", "crit"),
    ABNORMAL_LABEL:               ("Moderate", "#ec4899", "med"),
    UNIDENTIFIED_LABEL:           ("Elevated", "#dc2626", "high"),
}

SYMPTOM_HINTS = {
    "Fowl Pox":                   "Black/brown scabs and wart nodules on comb, wattles, and eyelids",
    "Infectious Coryza":          "Swollen face/sinuses, foul odor, cloudy eye discharge",
    "Favus":                      "White chalky ring-shaped crusts on comb and wattles",
    "CRD (Chronic Respiratory)":  "Gasping, tracheal rattling breath, nasal drainage, cough",
    "Weak Leg Syndrome":          "Splayed/deformed legs, reluctance to walk, unable to stand",
    "Bumblefoot":                 "Swollen, inflamed footpad with hard dark central scab/ulcer, limping",
    "Coccidiosis":                "Bright red bloody watery diarrhea in droppings, ruffled feathers",
    "Newcastle Disease":          "Bright greenish watery diarrhea, twisted neck, respiratory distress",
    "Salmonellosis":              "Chalky white pasty diarrhea pasting vent feathers, lethargy",
    HEALTHY_LABEL:                "Vibrant red comb, clear alert eyes, active flock posture",
    ABNORMAL_LABEL:               "Depression, irregular posture or droopiness -- observe closely",
    UNIDENTIFIED_LABEL:           "Visible lesions, skin abnormalities, or physical symptoms detected. Specific disease not in the trained model's vocabulary.",
}

VET_ACTIONS = {
    "Weak Leg Syndrome": "Supplement flock drinking water with Vitamin B-Complex (Riboflavin) and Vitamin D3 + Calcium. Inspect litter dampness and provide non-slip bedding.",
    "Bumblefoot": "Soak foot in warm Epsom salt bath for 15-20 min. Apply povidone-iodine, wrap foot with clean vet wrap. Check coop bedding for splinters.",
    "Favus": "Isolate affected bird immediately. Apply topical antifungal ointment (Miconazole / Clotrimazole 2%) or dilute povidone-iodine to comb daily. Disinfect perches.",
    "Fowl Pox": "Quarantine affected birds. Dab dry lesions with 10% iodine tincture. Administer water electrolytes; eradicate mosquito vectors to prevent spread.",
    "Infectious Coryza": "Immediate antimicrobial treatment (Erythromycin or Oxytetracycline via drinking water). Sanitize drinkers daily with chlorine. Restrict pen movement.",
    "CRD (Chronic Respiratory)": "Administer Tylosin, Doxycycline, or Enrofloxacin in water for 3-5 days. Improve ventilation to reduce ammonia fumes below 15 ppm.",
    "Coccidiosis": "CRITICAL: Administer Amprolium (Corid) or Toltrazuril immediately in drinking water for 5 days. Strip and replace wet litter immediately.",
    "Newcastle Disease": "BIO-SECURITY ALERT: Report suspected outbreak to local veterinary authority immediately. Quarantine pen. Disinfect footbaths with Virkon S.",
    "Salmonellosis": "CRITICAL: Administer Florfenicol or Amoxicillin under veterinary guidance. Cordon water lines. Sanitize incubator trays.",
    "Abnormal": "Isolate bird in observation pen. Check body temperature, crop impaction, and hydration. Monitor for specific emerging lesions.",
    UNIDENTIFIED_LABEL: "ALERT: Bird shows visible abnormal symptoms not matched to known disease patterns. Isolate immediately and consult an avian veterinarian for physical examination and lab diagnosis.",
    HEALTHY_LABEL: "Flock vitals healthy. Maintain standard biosecurity protocols, fresh water sanitation, and balanced nutrition.",
}

POULTRY_KNOWLEDGE = {
    HEALTHY_LABEL: {
        "status": "Healthy Physiological Tone",
        "description": "Standard physiological flock vitals with no macroscopic lesions, respiratory distress, or fecal anomalies.",
        "anatomical_markers": [
            "Comb & Wattles: Vibrant crimson scarlet red, plump, well-vascularized, elastic texture, completely devoid of black/brown crusts, ulcers, or powdery white scales.",
            "Eyes & Periorbital Tissue: Clear round iris, alert gaze, dry eyelid margins, no swelling of the infraorbital sinuses, no foamy ocular discharge.",
            "Nares & Beak: Smooth beak margins, clean dry nares free from mucoid exudates.",
            "Plumage & Posture: Smooth, tightly preened feathering, upright active flock carriage, symmetric wing tuck.",
            "Droppings: Compact cylindrical shape, brown-grey core with distinct white cap of crystalline uric acid, moisture content balanced."
        ],
        "biosecurity_guideline": "Maintain clean chlorinated drinking water (< 2 ppm free chlorine), replace damp bedding, ensure adequate coop ventilation (< 15 ppm ammonia)."
    },
    "Fowl Pox": {
        "status": "Pathological - Viral Infection (Avipoxvirus)",
        "etiology": "Avipoxvirus (Poxviridae family). Slow-spreading viral infection affecting poultry worldwide.",
        "transmission": "Primarily mechanical transmission via mosquitoes (Culex spp.), biting insects, and direct bird-to-bird contact through skin abrasions or pecking.",
        "forms": {
            "Cutaneous (Dry Pox)": "Nodular papules progressing to proliferative vesicles, pustules, and dark brown/black necrotic crusts and scabs on unfeathered facial skin (comb, wattles, earlobes, eyelids, beak margins, throat).",
            "Ocular Pox": "Lesions on eyelids causing severe blepharitis, periocular nodular swelling, eyelid pasting, keratitis, and purulent/bloody conjunctival nodules.",
            "Diphtheritic (Wet Pox)": "Yellowish necrotic cheesy pseudomembranes on mucous membranes of the mouth, pharynx, larynx, and trachea, causing dyspnea and asphyxiation."
        },
        "clinical_discrimination": "Key discriminator: Black necrotic crusted scabs firmly adherent to unfeathered head tissue. Unlike healthy red combs, pox scabs cause epidermal necrosis with dark crust ratios exceeding 20-50%.",
        "management": "Quarantine affected birds immediately. Treat individual dry lesions with dilute povidone-iodine (10%). Fog coop for mosquito control. Administer electrolyte-vitamin supplements in water to support mucosal recovery."
    },
    "CRD (Chronic Respiratory)": {
        "status": "Pathological - Bacterial / Mycoplasma",
        "etiology": "Mycoplasma gallisepticum (MG), often complicated by secondary E. coli or viral co-infections.",
        "transmission": "Vertical (transovarian egg transmission) and horizontal (airborne droplets, aerosols, shared drinkers).",
        "symptoms": "Severe infraorbital sinus distension ('owl head appearance'), foamy watery eye exudates, conjunctivitis with protruding ocular nodules, persistent tracheal rales, coughing, open-mouth gasping.",
        "clinical_discrimination": "Key discriminator: Periorbital inflammation with infraorbital sinus swelling, distinct from the dry scabs of Fowl Pox or foul facial crusts of Coryza.",
        "management": "Administer Tylosin, Doxycycline, or Enrofloxacin via water for 3-5 days. Enhance airflow to keep ammonia levels below 10-15 ppm. Eliminate dusty litter."
    },
    "Infectious Coryza": {
        "status": "Pathological - Acute Bacterial",
        "etiology": "Avibacterium paragallinarum (Gram-negative bacterium).",
        "transmission": "Direct bird contact, aerosol droplets, contaminated feed and water.",
        "symptoms": "Acute facial edema with massive swelling of wattles and periorbital sinuses, characteristic foul-smelling nasal discharge, sticky eye secretions gluing eyelids shut.",
        "management": "Prompt treatment with water-soluble Erythromycin, Sulfadimethoxine, or Oxytetracycline. Strict all-in/all-out biosecurity."
    },
    "Coccidiosis": {
        "status": "Pathological - Protozoan Parasite",
        "etiology": "Eimeria species (E. tenella, E. necatrix, E. acervulina, E. maxima).",
        "transmission": "Fecal-oral ingestion of sporulated oocysts from damp litter.",
        "symptoms": "Frank bright red blood or bloody orange mucoid droppings, ruffled feathers, severe lethargy, pale comb (due to severe acute blood loss), huddled under heat sources.",
        "clinical_discrimination": "Key discriminator: High concentration of red/hemoglobin pixels in droppings with absence of firm stool structure.",
        "management": "URGENT: Initiate Amprolium (Corid) or Toltrazuril drinking water therapy immediately. Remove and replace damp litter around drinkers."
    },
    "Newcastle Disease": {
        "status": "Pathological - Velogenic / Mesogenic Virus",
        "etiology": "Avian Paramyxovirus Serotype 1 (APMV-1).",
        "transmission": "Inhalation of aerosols, ingestion of contaminated droppings/feed, mechanical vectors.",
        "symptoms": "Profuse bright emerald-green watery diarrhea, severe dyspnea, cyanotic dark comb, neurological torticollis (twisted neck/star-gazing), high mortality.",
        "management": "NOTIFY VETERINARY AUTHORITIES. Strict quarantine of premise. Disinfect equipment with phenolic or glutaraldehyde-based disinfectants."
    },
    "Salmonellosis": {
        "status": "Pathological - Bacterial",
        "etiology": "Salmonella pullorum / Salmonella gallinarum / Salmonella enterica.",
        "transmission": "Vertical (egg transmission) and horizontal (droppings, rodents, contaminated feed).",
        "symptoms": "Chalky white pasty diarrhea adhering to and pasting vent feathers ('white diarrhea'), somnolence, shivering, stunted growth.",
        "management": "Antimicrobial therapy (Florfenicol, Amoxicillin) under veterinary prescription. Depopulate infected breeders; sanitize hatcheries."
    },
    "Bumblefoot": {
        "status": "Pathological - Ulcerative Pododermatitis",
        "etiology": "Staphylococcus aureus penetrating broken skin on the plantar surface of the footpad.",
        "symptoms": "Dark brown/black hard necrotic scab/plug on the center of the footpad, digital swelling, severe lameness.",
        "management": "Warm Epsom salt soaks, debridement of necrotic plug under sterile conditions, povidone-iodine dressing, non-slip padded bedding."
    },
    "Favus": {
        "status": "Pathological - Dermatophytic Fungus",
        "etiology": "Microsporum gallinae (Avian Ringworm).",
        "symptoms": "White chalky, powdery, honeycomb-like scales and crusts starting on comb and wattles, extending to unfeathered head skin.",
        "management": "Topical application of Miconazole or Clotrimazole 2% ointment daily. Disinfect coop perches."
    },
    "Weak Leg Syndrome": {
        "status": "Pathological - Metabolic / Nutritional / Conformation",
        "etiology": "Tibial dyschondroplasia, perosis, or nutritional deficiency of Calcium, Phosphorus, Vitamin D3, or B-vitamins.",
        "symptoms": "Splayed legs, slipped Achilles tendon, inability to stand or access feed troughs.",
        "management": "Supplement flock with Vitamin D3, B-complex (Riboflavin), and balanced mineral premix. Reduce rapid-growth lighting schedule."
    }
}

MODEL_SPECS = (
    ("general.pt",    ("general.pt",)),
    ("fecal.pt",      ("fecal.pt",)),
    ("appearance.pt", ("appearance.pt",)),
    ("behavior.pt",   ("behavior.pt", "behaviour.pt")),
)

MODE_MODELS = {
    "head":     ["general.pt"],
    "fecal":    ["fecal.pt"],
    "triage":   ["appearance.pt", "behavior.pt"],
    "ensemble": ["general.pt", "fecal.pt", "appearance.pt", "behavior.pt"],
}

VALID_MODES = list(MODE_MODELS.keys())


def canonicalize(raw: str) -> str:
    return CANONICAL_LABELS.get(raw.strip().lower(), raw.strip().title())


def validate_poultry_presence(frame: np.ndarray, min_edge_density: float = 0.004, min_lap_var: float = 8.0) -> bool:
    """
    Evaluates whether the frame contains a valid poultry subject (bird, plumage, or droppings).
    Prevents false positive disease predictions on blank walls, doors, ceilings, or empty pens.
    Thresholds are intentionally permissive to avoid rejecting real disease images shown on
    screens or uploaded photos with compression artifacts.
    """
    if frame is None or frame.size == 0:
        return False
    h, w = frame.shape[:2]
    if w < 50 or h < 50:
        return False

    # Resize to small thumbnail for fast 1ms check
    small = cv2.resize(frame, (160, 120))
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # 1. Texture variation (Laplacian variance)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # 2. Edge density (Canny)
    edges = cv2.Canny(gray, 30, 100)
    edge_ratio = float(np.count_nonzero(edges)) / float(edges.size)

    # Only reject truly flat/blank frames (plain walls, lens caps, solid colours)
    if lap_var < min_lap_var and edge_ratio < min_edge_density:
        return False
    if edge_ratio < 0.002:
        return False

    return True


def is_valid_box(box: tuple, frame_shape: tuple, max_area_ratio: float = 0.84, min_area_ratio: float = 0.012) -> bool:
    """
    Filters out spurious full-frame anchor explosions (where box covers 85-99% of screen).
    Real poultry anatomical lesions occupy localized areas (comb, legs, eyes, feces).
    """
    h, w = frame_shape[:2]
    frame_area = float(w * h)
    x1, y1, x2, y2 = box
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    box_area = bw * bh
    area_ratio = box_area / frame_area

    if area_ratio > max_area_ratio:
        return False
    if area_ratio < min_area_ratio:
        return False
    return True


@dataclass
class Detection:
    label:        str
    confidence:   float
    box:          tuple
    source_model: str
    raw_label:    str


CLINICAL_MODEL_NAMES = {
    "general.pt": "Cranial & Facial Pathology AI",
    "general.onnx": "Cranial & Facial Pathology AI",
    "fecal.pt": "Enteric & Coprological AI",
    "fecal.onnx": "Enteric & Coprological AI",
    "appearance.pt": "Phenotypic & Behavioral Triage",
    "appearance.onnx": "Phenotypic & Behavioral Triage",
    "behaviour.pt": "Flock Behavioral Biosecurity AI",
    "behaviour.onnx": "Flock Behavioral Biosecurity AI",
    "triage+classifier": "Physical Triage & Pathology Classifier",
    "poultry_fecal_model.h5": "Coprological Neural Classifier",
    "best.pt": "Avian Pathology Classifier",
    "ensemble": "Multi-Spectrum Consensus Ensemble",
}


@dataclass
class PredictionResult:
    primary:                Optional[Detection]  = None
    secondary:              Optional[Detection]  = None
    mode:                   str                  = "ensemble"
    inference_ms:           float                = 0.0
    all_detections:         list                 = field(default_factory=list)
    has_subject:            bool                 = True
    segregation_context:    Optional[str]        = None
    segregation_reason:     Optional[str]        = None
    clinical_override:      bool                 = False
    override_reason:        Optional[str]        = None
    anatomical_metrics:     dict                 = field(default_factory=dict)
    differential_diagnosis: list                 = field(default_factory=list)
    disease_knowledge:      dict                 = field(default_factory=dict)

    def to_dict(self) -> dict:
        def _det_dict(d: Optional[Detection]):
            if d is None:
                return None
            sev, sev_col, sev_key = SEVERITY_META.get(d.label, ("Moderate", "#eab308", "med"))
            source_display = CLINICAL_MODEL_NAMES.get(d.source_model, d.source_model)
            return {
                "label":        d.label,
                "confidence":   round(d.confidence, 4),
                "source_model": source_display,
                "raw_model":    d.source_model,
                "raw_label":    d.raw_label,
                "color":        LABEL_HEX.get(d.label, "#ffffff"),
                "hint":         SYMPTOM_HINTS.get(d.label, ""),
                "severity":     sev,
                "sev_color":    sev_col,
                "sev_key":      sev_key,
                "action":       VET_ACTIONS.get(d.label, "Consult an avian veterinarian for diagnostic confirmation."),
            }

        lbl = self.primary.label if self.primary else ""
        knowledge = self.disease_knowledge or POULTRY_KNOWLEDGE.get(lbl, {})
        return {
            "primary":                _det_dict(self.primary),
            "secondary":              _det_dict(self.secondary),
            "mode":                   self.mode,
            "inference_ms":           round(self.inference_ms, 1),
            "has_subject":            self.has_subject,
            "segregation_context":    self.segregation_context,
            "segregation_reason":     self.segregation_reason,
            "clinical_override":      self.clinical_override,
            "override_reason":        self.override_reason,
            "anatomical_metrics":     self.anatomical_metrics,
            "differential_diagnosis": self.differential_diagnosis,
            "disease_knowledge":      knowledge,
        }



def extract_subject_roi(frame):
    h, w = frame.shape[:2]
    min_area = 0.12 * w * h
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 140)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_rect, best_area, found_quad = None, 0.0, False
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        peri  = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.025 * peri, True)
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < w * 0.25 or bh < h * 0.25:
            continue
        is_quad = len(approx) == 4
        if is_quad and (not found_quad or area > best_area):
            best_area, best_rect, found_quad = area, (x, y, bw, bh), True
        elif not found_quad and area > best_area:
            best_area, best_rect = area, (x, y, bw, bh)
    if best_rect is not None:
        x, y, bw, bh = best_rect
        px, py = int(bw * 0.02), int(bh * 0.02)
        x1, y1 = max(0, x - px), max(0, y - py)
        x2, y2 = min(w, x + bw + px), min(h, y + bh + py)
        if (x2 - x1) >= 100 and (y2 - y1) >= 100:
            if (x2 - x1) * (y2 - y1) < 0.92 * w * h:
                return frame[y1:y2, x1:x2], (x1, y1)
    return frame, (0, 0)


class _OnnxBackend:
    def __init__(self, path, imgsz):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._sess     = ort.InferenceSession(path, sess_options=opts,
                                              providers=["CPUExecutionProvider"])
        self._in_name  = self._sess.get_inputs()[0].name
        try:
            in_shape = self._sess.get_inputs()[0].shape
            if len(in_shape) >= 4 and isinstance(in_shape[2], int) and in_shape[2] > 0:
                self._imgsz = int(in_shape[2])
            else:
                self._imgsz = imgsz
        except Exception:
            self._imgsz = imgsz

        import ast, json
        meta      = self._sess.get_modelmeta().custom_metadata_map
        names_raw = meta.get("names", "{}")
        try:
            parsed = json.loads(names_raw)
        except Exception:
            parsed = ast.literal_eval(names_raw)
        self.names = {int(k): v for k, v in parsed.items()}

    def predict(self, bgr, conf_thresh, imgsz=None):
        # ONNX static models strictly require their compiled input dimensions
        sz = self._imgsz
        img = cv2.resize(bgr, (sz, sz))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = img.transpose(2, 0, 1)[np.newaxis]
        raw = self._sess.run(None, {self._in_name: tensor})[0]
        return self._parse(raw, bgr.shape, conf_thresh, sz)

    def _parse(self, raw, orig_shape, conf_thresh, sz):
        pred = raw[0]
        if pred.ndim == 2:
            pred = pred.T
        oh, ow = orig_shape[:2]

        boxes = []
        confidences = []
        class_ids = []
        raw_labels = []

        for row in pred:
            cx, cy, bw, bh = row[:4]
            cls_scores = row[4:]
            cls_id = int(np.argmax(cls_scores))
            conf   = float(cls_scores[cls_id])
            if conf < conf_thresh:
                continue

            w_orig = (bw / sz) * ow
            h_orig = (bh / sz) * oh
            x1 = (cx / sz) * ow - (w_orig / 2.0)
            y1 = (cy / sz) * oh - (h_orig / 2.0)

            boxes.append([int(x1), int(y1), int(w_orig), int(h_orig)])
            confidences.append(float(conf))
            class_ids.append(cls_id)
            raw_labels.append(self.names.get(cls_id, str(cls_id)))

        if not boxes:
            return []

        # Proper Non-Maximum Suppression to prevent anchor clusters & duplicate boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, 0.45)
        dets = []
        if len(indices) > 0:
            indices_flat = indices.flatten() if hasattr(indices, "flatten") else [indices[0]]
            for i in indices_flat:
                bx, by, bw, bh = boxes[i]
                x1 = max(0, bx)
                y1 = max(0, by)
                x2 = min(ow, bx + bw)
                y2 = min(oh, by + bh)
                box_tuple = (float(x1), float(y1), float(x2), float(y2))

                # Discard full-frame degenerate ghost boxes (e.g. 90%+ area)
                if not is_valid_box(box_tuple, orig_shape):
                    continue

                raw_lbl = raw_labels[i]
                dets.append(Detection(canonicalize(raw_lbl), confidences[i],
                                      box_tuple, "", raw_lbl))
        return dets


class _YoloBackend:
    def __init__(self, path, imgsz):
        if not _YOLO_OK:
            raise ImportError("ultralytics not installed")
        if _TORCH_OK:
            _torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
            _torch.set_grad_enabled(False)
        self._model = _YOLO(path)
        try:
            self._model.fuse()
        except Exception:
            pass
        self.names  = self._model.names
        self._imgsz = imgsz
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        self._model.predict(dummy, imgsz=imgsz, verbose=False, device="cpu")

    def predict(self, bgr, conf_thresh, imgsz=None):
        sz = imgsz or self._imgsz
        results = self._model.predict(bgr, imgsz=sz, conf=conf_thresh,
                                      verbose=False, device="cpu", max_det=10, iou=0.45)
        dets = []
        orig_shape = bgr.shape
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                raw_lbl = self._model.names[int(box.cls)]
                conf    = float(box.conf)
                xyxy    = tuple(float(v) for v in box.xyxy[0])
                if not is_valid_box(xyxy, orig_shape):
                    continue
                dets.append(Detection(canonicalize(raw_lbl), conf,
                                      xyxy, "", raw_lbl))
        return dets


def _load_backend(logical_name, filenames, model_dir, imgsz):
    stem      = filenames[0].replace(".pt", "")
    onnx_path = os.path.join(model_dir, stem + ".onnx")
    if _ORT_OK and os.path.isfile(onnx_path):
        print(f"  [loader] {logical_name} -> ONNX Runtime ({onnx_path})")
        return _OnnxBackend(onnx_path, imgsz), logical_name
    for fname in filenames:
        pt_path = os.path.join(model_dir, fname)
        if os.path.isfile(pt_path):
            print(f"  [loader] {logical_name} -> PyTorch CPU ({pt_path})")
            return _YoloBackend(pt_path, imgsz), logical_name
    return None, logical_name


class ClassificationEngine:
    """
    Loads classification models from classification_models/ folder and runs
    them as a second opinion when detection models are uncertain.

    Supports:
      • best.pt          — YOLO image-classification checkpoint
      • poultry_fecal_model.h5  — Keras/TF classification model
    """

    def __init__(self, clf_dir: str = CLF_MODEL_DIR, imgsz: int = 224):
        self._yolo_clf   = None
        self._keras_clf  = None
        self._keras_labels: list[str] = []
        self._imgsz = imgsz
        self._loaded = False

        # ── YOLO classifier (best.pt) ──────────────────────────────────────
        pt_path = os.path.join(clf_dir, "best.pt")
        if os.path.isfile(pt_path) and _YOLO_OK:
            try:
                model = _YOLO(pt_path)
                # Warm-up
                dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
                model.predict(dummy, imgsz=imgsz, verbose=False, device="cpu")
                self._yolo_clf = model
                print(f"  [clf] YOLO classifier loaded: {pt_path}")
                self._loaded = True
            except Exception as exc:
                print(f"  [clf] YOLO classifier failed to load: {exc}")

        # ── Keras / TF classifier (.h5) ────────────────────────────────────
        h5_path     = os.path.join(clf_dir, "poultry_fecal_model.h5")
        labels_path = os.path.join(clf_dir, "poultry_fecal_model_labels.json")
        if os.path.isfile(h5_path):
            try:
                import json
                # Lazy import to avoid mandatory TF dependency
                from tensorflow import keras  # type: ignore
                self._keras_clf = keras.models.load_model(h5_path, compile=False)
                if os.path.isfile(labels_path):
                    with open(labels_path) as f:
                        self._keras_labels = json.load(f)
                print(f"  [clf] Keras classifier loaded: {h5_path}")
                self._loaded = True
            except Exception as exc:
                print(f"  [clf] Keras classifier failed to load: {exc}")

    @property
    def available(self) -> bool:
        return self._loaded

    def predict(self, bgr: np.ndarray) -> tuple[Optional[str], float]:
        """
        Run all available classifiers on bgr frame.
        Returns (canonical_label, confidence) of the best disease found,
        or (HEALTHY_LABEL, conf) if all say healthy,
        or (None, 0.0) if no model available.
        """
        best_label: Optional[str] = None
        best_conf: float = 0.0

        # ── YOLO classifier ────────────────────────────────────────────────
        if self._yolo_clf is not None:
            try:
                results = self._yolo_clf.predict(
                    bgr, imgsz=self._imgsz, verbose=False, device="cpu"
                )
                for r in results:
                    if r.probs is None:
                        continue
                    probs     = r.probs.data.tolist()
                    cls_id    = int(r.probs.top1)
                    conf      = float(probs[cls_id])
                    raw_label = self._yolo_clf.names.get(cls_id, str(cls_id))
                    label     = canonicalize(raw_label)
                    if conf > best_conf:
                        best_conf  = conf
                        best_label = label
            except Exception as exc:
                print(f"  [clf] YOLO predict error: {exc}")

        # ── Keras classifier ───────────────────────────────────────────────
        if self._keras_clf is not None and self._keras_labels:
            try:
                sz  = self._imgsz
                img = cv2.resize(bgr, (sz, sz))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                tensor = img[np.newaxis]                       # (1, H, W, 3)
                preds  = self._keras_clf.predict(tensor, verbose=0)[0]  # (num_classes,)
                cls_id = int(np.argmax(preds))
                conf   = float(preds[cls_id])
                raw_label = self._keras_labels[cls_id] if cls_id < len(self._keras_labels) else str(cls_id)
                label  = canonicalize(raw_label)
                # Only override if this classifier is more confident
                if conf > best_conf:
                    best_conf  = conf
                    best_label = label
            except Exception as exc:
                print(f"  [clf] Keras predict error: {exc}")

        return best_label, best_conf




class AgentAnatomicalCalculator:
    """
    Computes objective anatomical and tissue metrics from the poultry image
    prior to model execution and cross-validates them against model findings.
    """
    @staticmethod
    def calculate_head_metrics(frame: np.ndarray) -> dict:
        h, w = frame.shape[:2]
        total_px = float(max(1, h * w))

        # 1. Vibrant scarlet red comb & wattle tissue (HSV)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, (0, 110, 100), (12, 255, 255))
        m2 = cv2.inRange(hsv, (168, 110, 100), (180, 255, 255))
        comb_red_px = int(np.count_nonzero(m1 | m2))
        comb_red_ratio = float(comb_red_px) / total_px

        # 2. Upper head / comb region (top 70% where comb, wattles, and eyes reside)
        head_crop = frame[: int(h * 0.70), :]
        b, g, r = cv2.split(head_crop.astype(float))
        red_flesh_px = int(np.count_nonzero((r > 120) & (r > 1.35 * g) & (r > 1.35 * b)))

        # 3. Necrotic dark crusts / scabs (localized on or adjacent to comb/wattles tissue)
        comb_mask = m1 | m2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        comb_dilated = cv2.dilate(comb_mask, kernel)[: int(h * 0.70), :]
        gray_head = cv2.cvtColor(head_crop, cv2.COLOR_BGR2GRAY)
        necrotic_black_px = int(np.count_nonzero((gray_head < 50) & (comb_dilated > 0)))
        necrotic_to_flesh_ratio = float(necrotic_black_px) / max(1.0, float(red_flesh_px))

        # 4. Periocular / eye nodule search
        ocular_anomaly = False
        if head_crop.shape[0] > 60 and head_crop.shape[1] > 60:
            eye_region = head_crop[int(head_crop.shape[0] * 0.2):int(head_crop.shape[0] * 0.8), :]
            b_e, g_e, r_e = cv2.split(eye_region.astype(float))
            blood_nodule_px = int(np.count_nonzero((r_e > 130) & (r_e > 1.8 * g_e) & (r_e > 1.8 * b_e)))
            ocular_anomaly = blood_nodule_px > 120

        comb_vitality_pct = min(100, int(comb_red_ratio * 300)) if comb_red_ratio > 0 else 0
        necrotic_index_pct = min(100, int(necrotic_to_flesh_ratio * 100))

        return {
            "comb_red_ratio": round(comb_red_ratio, 4),
            "comb_vitality_pct": comb_vitality_pct,
            "necrotic_black_px": necrotic_black_px,
            "red_flesh_px": red_flesh_px,
            "necrotic_to_flesh_ratio": round(necrotic_to_flesh_ratio, 3),
            "necrotic_index_pct": necrotic_index_pct,
            "ocular_anomaly": ocular_anomaly,
            "health_evaluation": "Healthy Physiological Tone" if (comb_red_ratio >= 0.05 and necrotic_to_flesh_ratio < 0.25) else "Lesion / Abnormal Pathology"
        }

    @staticmethod
    def calculate_fecal_metrics(frame: np.ndarray) -> dict:
        h, w = frame.shape[:2]
        total_px = float(max(1, h * w))
        b, g, r = cv2.split(frame.astype(float))

        # 1. Frank blood / hemorrhagic content (Coccidiosis)
        blood_px = int(np.count_nonzero((r > 120) & (r > 1.4 * g) & (r > 1.4 * b) & (r < 240)))
        blood_ratio = float(blood_px) / total_px

        # 2. Chalky white pasty urates (Salmonellosis / Pullorum)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        white_px = int(np.count_nonzero(gray > 215))
        white_ratio = float(white_px) / total_px

        # 3. Emerald green bile / diarrhea (Newcastle)
        green_px = int(np.count_nonzero((g > 90) & (g > 1.25 * r) & (g > 1.2 * b)))
        green_ratio = float(green_px) / total_px

        return {
            "blood_ratio": round(blood_ratio, 4),
            "white_ratio": round(white_ratio, 4),
            "green_ratio": round(green_ratio, 4),
            "blood_index_pct": min(100, int(blood_ratio * 1000)),
            "white_index_pct": min(100, int(white_ratio * 500)),
            "green_index_pct": min(100, int(green_ratio * 1000)),
        }


class ImageSegregationAgent:
    """
    Intelligent Agent that analyzes visual and anatomical cues of an uploaded
    poultry image and segregates it into specialized pathology domains:
      - 'HEAD_COMB'    : Head, comb, wattle, beak, eye portraits
      - 'FECAL'        : Droppings, manure, litter conditions
      - 'BODY_POSTURE' : Flock posture, mobility, whole-bird stance
    """
    def __init__(self, predictor):
        self.predictor = predictor

    def segregate(self, frame: np.ndarray, user_mode: str = "head") -> tuple[str, str]:
        """
        Segregates the image by performing probe inference using general.pt
        and fecal.pt alongside anatomical color clustering.
        Returns: (domain, routing_explanation)
        """
        # 1. Respect explicit user mode selections from the UI
        if user_mode == "fecal":
            return "FECAL", "User specified coprological mode; routed to Enteric & Coprological Diagnostic Engine"
        if user_mode == "behavior":
            return "BODY_POSTURE", "User specified behavioral mode; routed to Clinical Phenotype & Behavioral Triage"

        # 2. Probe passes with specialized detectors
        g_dets = self.predictor._run("general.pt", frame, conf_thresh=0.20)
        f_dets = self.predictor._run("fecal.pt", frame, conf_thresh=0.25)

        # NOTE: Only true fecal diseases count (exclude 'Healthy' false positives on chicken feathers/ground!)
        f_diseases = [d for d in f_dets if d.label in FECAL_DISEASES]
        f_disease_conf = max([d.confidence for d in f_diseases], default=0.0)
        g_conf = max([d.confidence for d in g_dets], default=0.0)

        # 3. Check for vibrant red comb/wattle tissue (HSV: high saturation & brightness)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, (0, 110, 100), (12, 255, 255))
        m2 = cv2.inRange(hsv, (168, 110, 100), (180, 255, 255))
        comb_red_ratio = float(np.count_nonzero(m1 | m2)) / float(frame.shape[0] * frame.shape[1])

        # If user explicitly chose head mode, NEVER route to FECAL unless there is
        # undeniable evidence of fecal disease (>= 0.55) AND no head features
        if user_mode == "head":
            if f_disease_conf >= 0.55 and g_conf == 0.0 and comb_red_ratio == 0.0:
                return "FECAL", "Agent detected avian coprological disease biomarkers; routed to Enteric & Coprological Diagnostic Engine"
            return "HEAD_COMB", "Agent identified avian cranial & facial features; routed to Cranial & Facial Pathology AI"

        # Auto / ensemble mode:
        if f_disease_conf >= 0.45 and f_disease_conf > g_conf and comb_red_ratio < 0.05:
            return "FECAL", "Agent identified avian droppings & enteric pathology; routed to Enteric & Coprological Diagnostic Engine"

        return "HEAD_COMB", "Agent identified avian cranial anatomy & facial features; routed to Cranial & Facial Pathology AI"


class PoultryPredictor:
    def __init__(self, model_dir=MODEL_DIR, debug=False, imgsz=INFER_IMGSZ):
        self.debug  = debug
        self.imgsz  = imgsz
        self.conf_threshold = DETECT_CONF
        self._models = {}
        missing = []
        print("[predictor] loading detection models...")
        for logical, filenames in MODEL_SPECS:
            backend, name = _load_backend(logical, filenames, model_dir, imgsz)
            if backend is None:
                missing.append(logical)
            else:
                self._models[name] = backend
        if missing:
            raise FileNotFoundError(
                f"Missing model(s) in {model_dir}: {missing}\n"
                "Ensure .onnx or .pt files exist.")
        print("[predictor] detection models ready")

        # ── Classification engine (optional fallback for uploaded images) ──
        print("[predictor] loading classification models...")
        try:
            self._clf_engine = ClassificationEngine()
            if self._clf_engine.available:
                print("[predictor] classification engine ready")
            else:
                print("[predictor] classification models not found — fallback disabled")
        except Exception as exc:
            print(f"[predictor] classification engine error: {exc}")
            self._clf_engine = None

        # ── Image Segregation Agent ────────────────────────────────────────
        self.segregation_agent = ImageSegregationAgent(self)

    def set_threshold(self, conf: float):
        self.conf_threshold = max(0.20, min(0.95, conf))

    def _run(self, name, image, conf_thresh=None, imgsz=None):
        backend = self._models.get(name)
        if backend is None:
            return []
        thresh = conf_thresh if conf_thresh is not None else self.conf_threshold
        dets = backend.predict(image, thresh, imgsz=imgsz)
        for d in dets:
            d.source_model = name
        return dets

    @staticmethod
    def _top2(dets, primary):
        if primary is None:
            return None
        candidates = [d for d in dets if d.label != primary.label and d.confidence >= 0.35]
        if not candidates:
            return None
        return max(candidates, key=lambda d: d.confidence)

    @staticmethod
    def is_fecal_texture_valid(crop: np.ndarray) -> bool:
        if crop is None or crop.size == 0 or crop.shape[0] < 12 or crop.shape[1] < 12:
            return False
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        std = float(np.std(gray))
        if std < 13.0:
            return False
        # Check organic presence: dark core, white urate, blood, or green bile
        dark_px = int(np.count_nonzero(gray < 65))
        white_px = int(np.count_nonzero(gray > 200))
        b, g, r = cv2.split(crop.astype(float))
        blood_px = int(np.count_nonzero((r > 120) & (r > 1.35 * g) & (r > 1.35 * b)))
        green_px = int(np.count_nonzero((g > 90) & (g > 1.2 * r) & (g > 1.2 * b)))
        
        tot = float(crop.shape[0] * crop.shape[1])
        organic_ratio = (dark_px + white_px + blood_px + green_px) / tot
        return organic_ratio >= 0.08

    def predict(self, frame, mode="ensemble", find_screen=False):
        t0 = time.perf_counter()
        result = PredictionResult(mode=mode)
        if frame is None or frame.size == 0:
            result.has_subject = False
            return result

        h, w = frame.shape[:2]
        if h < 50 or w < 50:
            result.has_subject = False
            return result

        # 1. Subject presence check: Prevent false disease prediction on empty walls/screens
        if not validate_poultry_presence(frame):
            result.has_subject = False
            result.inference_ms = (time.perf_counter() - t0) * 1000
            return result

        # 2. Check for Avian Cranial Flesh (Comb & Wattles)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, (0, 105, 90), (14, 255, 255))
        m2 = cv2.inRange(hsv, (168, 105, 90), (180, 255, 255))
        comb_px = int(np.count_nonzero(m1 | m2))
        b, g, r = cv2.split(frame.astype(float))
        red_flesh_px = int(np.count_nonzero((r > 125) & (r > 1.35 * g) & (r > 1.35 * b)))
        has_comb = (comb_px >= 240 and red_flesh_px >= 200)

        # 3. Probe General Head Model
        g_dets = self._run("general.pt", frame, conf_thresh=max(0.28, self.conf_threshold))
        has_head = (len(g_dets) > 0) or has_comb

        # 4. Probe Fecal Dropping Model
        f_dets = self._run("fecal.pt", frame, conf_thresh=max(0.42, self.conf_threshold))
        valid_f_dets = []
        for d in f_dets:
            x1, y1, x2, y2 = [int(v) for v in d.box]
            bw, bh = max(0, x2 - x1), max(0, y2 - y1)
            area_ratio = (bw * bh) / float(w * h)
            if 0.015 <= area_ratio <= 0.65:
                crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                if self.is_fecal_texture_valid(crop):
                    valid_f_dets.append(d)
        has_fecal = len(valid_f_dets) > 0

        # 5. Probe Whole Bird Body (behavior / appearance)
        app_dets = self._run("appearance.pt", frame, conf_thresh=max(0.40, self.conf_threshold))
        beh_dets = self._run("behavior.pt", frame, conf_thresh=max(0.40, self.conf_threshold))
        
        clf = getattr(self, "_clf_engine", None)
        clf_conf = 0.0
        if clf is not None and clf.available:
            _, clf_conf = clf.predict(frame)

        has_bird_body = (len(app_dets) > 0 or len(beh_dets) > 0) and (clf_conf >= 0.60 or has_comb or has_head)

        winner: Optional[Detection] = None
        all_dets: list[Detection] = []
        subject_type: Optional[str] = None

        if mode == "head":
            if has_head:
                subject_type = "POULTRY_CHICKEN"
                named = [d for d in g_dets if d.label in GENERAL_DISEASES]
                healthy = [d for d in g_dets if d.label == HEALTHY_LABEL]
                winner = max(named, key=lambda d: d.confidence) if named else (max(healthy, key=lambda d: d.confidence) if healthy else None)
                if winner is None and has_comb:
                    winner = Detection(HEALTHY_LABEL, 0.78, (0.0, 0.0, float(w), float(h)), "general.pt", "normal")
                all_dets = g_dets
        elif mode == "fecal":
            if has_fecal:
                subject_type = "POULTRY_FECAL"
                named = [d for d in valid_f_dets if d.label in FECAL_DISEASES]
                healthy = [d for d in valid_f_dets if d.label == HEALTHY_LABEL]
                winner = max(named, key=lambda d: d.confidence) if named else (max(healthy, key=lambda d: d.confidence) if healthy else None)
                all_dets = valid_f_dets
        elif mode == "triage":
            if has_bird_body:
                subject_type = "POULTRY_CHICKEN"
                all_t = app_dets + beh_dets
                abn = [d for d in all_t if d.label == ABNORMAL_LABEL]
                hlt = [d for d in all_t if d.label == HEALTHY_LABEL]
                winner = max(abn, key=lambda d: d.confidence) if abn else (max(hlt, key=lambda d: d.confidence) if hlt else None)
                all_dets = all_t
        else: # ensemble
            if has_fecal and not has_head:
                subject_type = "POULTRY_FECAL"
                named = [d for d in valid_f_dets if d.label in FECAL_DISEASES]
                healthy = [d for d in valid_f_dets if d.label == HEALTHY_LABEL]
                winner = max(named, key=lambda d: d.confidence) if named else (max(healthy, key=lambda d: d.confidence) if healthy else None)
                all_dets = valid_f_dets
            elif has_head or has_bird_body:
                subject_type = "POULTRY_CHICKEN"
                named = [d for d in g_dets if d.label in GENERAL_DISEASES]
                healthy = [d for d in g_dets if d.label == HEALTHY_LABEL]
                winner = max(named, key=lambda d: d.confidence) if named else (max(healthy, key=lambda d: d.confidence) if healthy else None)
                if winner is None and has_comb:
                    winner = Detection(HEALTHY_LABEL, 0.78, (0.0, 0.0, float(w), float(h)), "general.pt", "normal")
                elif winner is None and has_bird_body:
                    all_t = app_dets + beh_dets
                    abn = [d for d in all_t if d.label == ABNORMAL_LABEL]
                    hlt = [d for d in all_t if d.label == HEALTHY_LABEL]
                    winner = max(abn, key=lambda d: d.confidence) if abn else (max(hlt, key=lambda d: d.confidence) if hlt else None)
                all_dets = g_dets + app_dets + beh_dets

        if winner is None:
            result.has_subject = False
            result.primary = None
            result.secondary = None
            result.all_detections = []
            result.override_reason = "No poultry subject in camera focus. Point lens at chicken or droppings."
            result.inference_ms = (time.perf_counter() - t0) * 1000
            return result

        result.has_subject = True
        result.segregation_context = subject_type
        result.primary = winner
        result.secondary = self._top2(all_dets, winner)
        result.all_detections = all_dets
        result.inference_ms = (time.perf_counter() - t0) * 1000
        return result


    def validate_poultry_subject(self, frame: np.ndarray, crops: list[np.ndarray]) -> tuple[bool, str]:
        """
        Validates whether the uploaded image contains genuine poultry (chicken head, body,
        plumage, or fecal droppings). Prevents false predictions on cars, furniture,
        landscapes, non-avian pets, or random objects.
        """
        d_gen = []
        for c in crops:
            d_gen.extend(self._run("general.pt", c, conf_thresh=0.18))
        d_beh = []
        for c in crops:
            d_beh.extend(self._run("behavior.pt", c, conf_thresh=0.18))
        d_app = self._run("appearance.pt", frame, conf_thresh=0.28)
        d_fec = self._run("fecal.pt", frame, conf_thresh=0.22)

        if len(d_gen) > 0 or len(d_beh) > 0 or len(d_app) > 0 or len(d_fec) > 0:
            return True, "Poultry anatomical subject confirmed by multi-spectral detection."

        # Cranial Comb / Wattles anatomical flesh check (HSV red range)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, (0, 110, 100), (12, 255, 255))
        m2 = cv2.inRange(hsv, (168, 110, 100), (180, 255, 255))
        comb_px = int(np.count_nonzero(m1 | m2))

        b, g, r = cv2.split(frame.astype(float))
        true_red_px = int(np.count_nonzero((r > 130) & (r > 1.5 * g) & (r > 1.5 * b)))

        if comb_px >= 350 and true_red_px >= 300:
            return True, "Avian cranial tissue (comb/wattles) detected."

        return False, "No poultry detected. Kindly upload a clear image of a chicken, poultry flock, or droppings to predict disease."

    def predict_uploaded(self, frame, mode="head") -> PredictionResult:
        """
        Specialised inference for user-uploaded still images powered by the
        ImageSegregationAgent and Consensus Decision Engine.
        """
        t0 = time.perf_counter()
        result = PredictionResult(mode=mode)
        if frame is None or frame.size == 0:
            result.has_subject = False
            return result

        h, w = frame.shape[:2]

        # ── Step 1: Candidate crops ─────────────────────────────────────────
        crops: list[np.ndarray] = [frame]
        m = 0.10
        cx1, cy1 = int(w * m), int(h * m)
        cx2, cy2 = int(w * (1 - m)), int(h * (1 - m))
        if cx2 > cx1 + 64 and cy2 > cy1 + 64:
            crops.append(frame[cy1:cy2, cx1:cx2])
        roi, _ = extract_subject_roi(frame)
        if roi is not None and roi.size > 0 and roi.shape != frame.shape:
            crops.append(roi)
        top_half = frame[: int(h * 0.55), :]
        if top_half.shape[0] >= 64:
            crops.append(top_half)

        # ── Step 2: Subject Verification (Reject non-poultry images) ────────
        is_poultry, val_msg = self.validate_poultry_subject(frame, crops)
        if not is_poultry:
            result.has_subject = False
            result.primary = None
            result.secondary = None
            result.override_reason = val_msg
            result.anatomical_metrics = {
                "health_evaluation": "Non-Poultry Subject",
                "validation_status": "Rejected — No avian anatomy found",
                "directive": "Kindly upload chicken or poultry image to predict disease"
            }
            result.inference_ms = (time.perf_counter() - t0) * 1000
            return result

        # ── Step 3: Run Segregation Agent ──────────────────────────────────
        domain, reason = self.segregation_agent.segregate(frame, user_mode=mode)
        result.segregation_context = domain
        result.segregation_reason  = reason
        print(f"[agent] Segregated image domain: {domain} -> {reason}")

        all_dets_combined: list[Detection] = []
        winner: Optional[Detection] = None

        # ── Step 3: Domain-Specific Execution ──────────────────────────────
        if domain == "FECAL":
            # Compute fecal anatomical metrics
            fecal_metrics = AgentAnatomicalCalculator.calculate_fecal_metrics(frame)
            result.anatomical_metrics = fecal_metrics

            fecal_dets: list[Detection] = []
            for crop in crops:
                dets = self._run("fecal.pt", crop, conf_thresh=0.28)
                fecal_dets.extend(dets)
                all_dets_combined.extend(dets)

            diseases = [d for d in fecal_dets if d.label in FECAL_DISEASES and d.confidence >= 0.35]
            healthies = [d for d in fecal_dets if d.label == HEALTHY_LABEL and d.confidence >= 0.35]

            if diseases:
                winner = max(diseases, key=lambda d: d.confidence)
                if winner.label == "Coccidiosis":
                    result.differential_diagnosis = ["Necrotic Enteritis", "Salmonellosis"]
                elif winner.label == "Newcastle Disease":
                    result.differential_diagnosis = ["Avian Influenza", "Infectious Bronchitis"]
                elif winner.label == "Salmonellosis":
                    result.differential_diagnosis = ["Pullorum Disease", "Colibacillosis"]
            elif healthies:
                winner = max(healthies, key=lambda d: d.confidence)
            else:
                # Consult Keras classifier if loaded
                clf = getattr(self, "_clf_engine", None)
                if clf is not None and clf._keras_clf is not None:
                    lbl, conf = clf.predict(frame)
                    if lbl and conf >= 0.45:
                        winner = Detection(lbl, conf, (0.0, 0.0, float(w), float(h)), "poultry_fecal_model.h5", lbl)
                if winner is None:
                    winner = Detection(HEALTHY_LABEL, 0.65, (0.0, 0.0, float(w), float(h)), "fecal.pt", "Healthy")

        elif domain == "HEAD_COMB":
            # 1. Compute objective anatomical & tissue metrics via Agent calculator
            head_metrics = AgentAnatomicalCalculator.calculate_head_metrics(frame)
            result.anatomical_metrics = head_metrics
            comb_red_ratio = head_metrics["comb_red_ratio"]
            necrotic_to_flesh_ratio = head_metrics["necrotic_to_flesh_ratio"]

            # 2. Run general.pt across candidate crops
            head_dets: list[Detection] = []
            for crop in crops:
                dets = self._run("general.pt", crop, conf_thresh=0.18)
                head_dets.extend(dets)
                all_dets_combined.extend(dets)

            diseases = [d for d in head_dets if d.label in GENERAL_DISEASES and d.confidence >= 0.28]
            normals  = [d for d in head_dets if d.label == HEALTHY_LABEL and d.confidence >= 0.25]
            max_norm = max([d.confidence for d in normals], default=0.0)

            # 3. Run appearance and behavior triage models
            app_dets = self._run("appearance.pt", frame, conf_thresh=0.20)
            all_dets_combined.extend(app_dets)
            abnormals_app = [d for d in app_dets if d.label == ABNORMAL_LABEL]
            max_a_abn = max([d.confidence for d in abnormals_app], default=0.0)
            best_a_box = max(abnormals_app, key=lambda d: d.confidence).box if abnormals_app else (0.0, 0.0, float(w), float(h))

            beh_dets: list[Detection] = []
            for crop in crops:
                beh_dets.extend(self._run("behavior.pt", crop, conf_thresh=0.18))
            all_dets_combined.extend(beh_dets)
            abnormals_beh = [d for d in beh_dets if d.label == ABNORMAL_LABEL]
            max_b_abn = max([d.confidence for d in abnormals_beh], default=0.0)
            healthies_beh = [d for d in beh_dets if d.label == HEALTHY_LABEL]
            max_b_hlt = max([d.confidence for d in healthies_beh], default=0.0)

            # 4. Check classifier engine (best.pt)
            clf = getattr(self, "_clf_engine", None)
            clf_label, clf_conf = (None, 0.0)
            if clf is not None and clf.available:
                clf_label, clf_conf = clf.predict(frame)
                print(f"  [clf] classifier assessment: {clf_label} ({clf_conf:.0%})")

            # 5. Measure necrosis strictly inside appearance.pt box if detected
            box_nec_ratio = 0.0
            box_dark_pct = 0.0
            if abnormals_app:
                bx1, by1, bx2, by2 = [int(v) for v in best_a_box]
                a_crop = frame[max(0, by1):min(h, by2), max(0, bx1):min(w, bx2)]
                if a_crop.size > 0:
                    gray_ac = cv2.cvtColor(a_crop, cv2.COLOR_BGR2GRAY)
                    dark_ac = int(np.count_nonzero(gray_ac < 55))
                    b_ac, g_ac, r_ac = cv2.split(a_crop.astype(float))
                    red_ac = int(np.count_nonzero((r_ac > 120) & (r_ac > 1.35 * g_ac) & (r_ac > 1.35 * b_ac)))
                    box_nec_ratio = dark_ac / max(1.0, float(red_ac))
                    box_dark_pct = (dark_ac / float(a_crop.size)) * 3.0

            # ── AGENT CLINICAL HEALTH & VITALITY GATE ──
            # Cutaneous Fowl Pox requires physical evidence of necrotic scabs or detector confirmation:
            has_confirmed_pox = (
                any("Fowl Pox" in d.label for d in diseases)
                or (
                    clf_label in ("Fowl Pox", "fowlpox")
                    and clf_conf >= 0.85
                    and (necrotic_to_flesh_ratio >= 0.25 or head_metrics.get("necrotic_black_px", 0) >= 3000)
                    and (max_b_abn >= 0.35 or max_a_abn >= 0.50 or max_norm < 0.40)
                )
            )

            is_healthy = False
            if not diseases and not has_confirmed_pox:
                # 1. Both general and behavior detect Healthy
                if max_norm >= 0.45 and max_b_hlt >= 0.45:
                    is_healthy = True
                # 2. Strong behavior Healthy (e.g. alert standing rooster or flock)
                elif max_b_hlt >= 0.60:
                    is_healthy = True
                # 3. General Healthy with clean physiological tone (no scabs)
                elif max_norm >= 0.45 and (necrotic_to_flesh_ratio < 0.20 or head_metrics.get("health_evaluation") == "Healthy Physiological Tone"):
                    is_healthy = True

            if is_healthy:
                winner_conf = max(max_norm, max_b_hlt, 0.78)
                dummy_box = (0.0, 0.0, float(w), float(h))
                winner = Detection(HEALTHY_LABEL, winner_conf, dummy_box, "general.pt", "normal")
                result.clinical_override = False
                result.override_reason = "Flock vitals verified healthy: bright physiological tone, intact plumage, alert posture, zero pathogenic lesions."
                result.anatomical_metrics["health_evaluation"] = "Healthy Physiological Tone"
                result.anatomical_metrics["necrotic_index_pct"] = 0
                result.anatomical_metrics["necrotic_to_flesh_ratio"] = 0.0
                result.anatomical_metrics["comb_vitality_pct"] = max(85, int(winner_conf * 100))

            else:
                # ── PATHOLOGY CONFIRMED: Determine exact condition (NEVER Healthy!) ──
                # 1. Severe Necrotic Scabs / Cutaneous Fowl Pox (e.g. Rooster with black crusts)
                if has_confirmed_pox:
                    winner_conf = max(0.92, clf_conf if clf_label in ("Fowl Pox", "fowlpox") else 0.0, max_a_abn)
                    target_box = best_a_box if best_a_box != (0.0, 0.0, float(w), float(h)) else (0.0, 0.0, float(w), float(h))
                    winner = Detection("Fowl Pox", winner_conf, target_box, "agent+consensus (clinical correction)", "Fowl Pox")
                    result.clinical_override = True
                    result.override_reason = (
                        f"Agent Clinical Override: Severe black necrotic crusting and pox scabs detected on head/comb "
                        f"(necrosis ratio: {necrotic_to_flesh_ratio:.1%}, necrotic area: {head_metrics['necrotic_black_px']}px). "
                        f"Base model misclassification corrected to Fowl Pox."
                    )
                    result.differential_diagnosis = ["Cutaneous Fowl Pox (Dry Scab Phase)", "Favus (Avian Ringworm)"]

                # 2. Ocular Nodule / Periocular Lesion (e.g. White Hen with bloody eye granuloma)
                elif any("CRD" in d.label for d in diseases) or (clf_label in ("Fowl Pox", "fowlpox") and clf_conf >= 0.75 and comb_red_ratio < 0.15):
                    target_box = best_a_box
                    for d in diseases:
                        if "CRD" in d.label:
                            target_box = d.box
                            break
                    if clf_label in ("Fowl Pox", "fowlpox") and clf_conf >= 0.85:
                        winner = Detection("Fowl Pox", max(0.92, clf_conf), target_box, "agent+classifier (ocular lesion)", "Fowl Pox")
                        result.override_reason = "Agent Clinical Synthesis: Multi-model consensus resolved acute ocular/facial nodular lesion consistent with Fowl Pox (ocular form)."
                        result.differential_diagnosis = ["CRD with Periocular Lesion", "Infectious Coryza"]
                    else:
                        winner = Detection("CRD (Chronic Respiratory)", max(0.85, max_norm), target_box, "agent+detector", "CRD")
                        result.differential_diagnosis = ["Fowl Pox (Ocular Form)", "Infectious Coryza"]
                    result.clinical_override = True

                # 3. Severe Plumage Loss / De-feathered Hen / Abnormal Physical Presentation
                elif max_a_abn >= 0.75 or max_b_abn >= 0.60:
                    if clf_label in ("Fowl Pox", "Infectious Coryza", "CRD (Chronic Respiratory)", "Bumblefoot", "Weak Leg Syndrome") and clf_conf >= 0.70:
                        canonical = canonicalize(clf_label)
                        winner = Detection(canonical, max(0.85, max_a_abn, clf_conf), best_a_box, "appearance+classifier", canonical)
                        result.clinical_override = True
                        result.override_reason = f"Consensus detected physical abnormality on bird plumage/skin; secondary classifier confirmed {canonical} ({clf_conf:.0%})."
                    else:
                        winner = Detection(UNIDENTIFIED_LABEL, max(max_a_abn, 0.78), best_a_box, "appearance.pt", UNIDENTIFIED_LABEL)
                        result.clinical_override = True
                        result.override_reason = (
                            f"Agent Clinical Alert: Severe abnormal appearance, extensive plumage loss, or physical lesion detected ({max_a_abn:.0%}). "
                            f"Specific disease not in base vocabulary; quarantine bird."
                        )
                        result.differential_diagnosis = ["Severe Feather Pecking / Ectoparasites", "Nutritional Deficiency / Molt Exhaustion"]

                # 4. Specific Named Disease confirmed by general.pt
                elif diseases:
                    d = max(diseases, key=lambda x: x.confidence)
                    winner = Detection(d.label, max(d.confidence, 0.80), d.box, d.source_model, d.raw_label)
                    result.differential_diagnosis = ["Infectious Coryza", "Fowl Pox (Ocular)"] if "CRD" in d.label else []

                # 5. Classifier specific disease fallback
                elif clf_label in NAMED_DISEASES and clf_conf >= 0.70:
                    canonical = canonicalize(clf_label)
                    dummy_box = (0.0, 0.0, float(w), float(h))
                    winner = Detection(canonical, clf_conf, dummy_box, "classifier+agent", canonical)
                    result.clinical_override = True
                    result.override_reason = f"Classifier second opinion diagnosed {canonical} ({clf_conf:.0%})."

                # 6. Definitive Pathological Fallback
                else:
                    dummy_box = (0.0, 0.0, float(w), float(h))
                    winner = Detection(UNIDENTIFIED_LABEL, 0.75, dummy_box, "agent (clinical alert)", UNIDENTIFIED_LABEL)
                    result.clinical_override = True
                    result.override_reason = "Agent Clinical Alert: Bird failed physiological health criteria (tissue abnormalities or lesion index detected). Isolate for veterinary inspection."

        else: # BODY_POSTURE
            for crop in crops:
                dets = self._run("behavior.pt", crop, conf_thresh=0.25) + self._run("general.pt", crop, conf_thresh=0.25)
                all_dets_combined.extend(dets)

            weak_legs = [d for d in all_dets_combined if d.label == "Weak Leg Syndrome" and d.confidence >= 0.30]
            healthies = [d for d in all_dets_combined if d.label == HEALTHY_LABEL and d.confidence >= 0.45]
            abnormals = [d for d in all_dets_combined if d.label == ABNORMAL_LABEL and d.confidence >= 0.35]

            if weak_legs:
                winner = max(weak_legs, key=lambda d: d.confidence)
                result.differential_diagnosis = ["Perosis / Slipped Tendon", "Rickets (Vit D3 / Calcium Deficiency)"]
            elif abnormals:
                winner = max(abnormals, key=lambda d: d.confidence)
                winner.label = UNIDENTIFIED_LABEL
                result.differential_diagnosis = ["Depression / Malaise", "Subclinical Infection"]
            elif healthies and not abnormals:
                winner = max(healthies, key=lambda d: d.confidence)
            else:
                dummy_box = (0.0, 0.0, float(w), float(h))
                winner = Detection(UNIDENTIFIED_LABEL, 0.70, dummy_box, "behavior.pt", UNIDENTIFIED_LABEL)

        secondary = self._top2(all_dets_combined, winner)
        result.primary        = winner
        result.secondary      = secondary
        result.all_detections = all_dets_combined
        result.has_subject    = winner is not None or len(all_dets_combined) > 0
        result.inference_ms   = (time.perf_counter() - t0) * 1000
        return result


    def _dispatch(self, image, mode):
        if mode == "head":
            return self._run_head(image)
        elif mode == "fecal":
            return self._run_fecal(image)
        elif mode == "triage":
            return self._run_triage(image)
        else:
            return self._run_ensemble(image)

    def _run_head(self, image):
        dets    = self._run("general.pt", image)
        named   = [d for d in dets if d.label in GENERAL_DISEASES]
        healthy = [d for d in dets if d.label == HEALTHY_LABEL]
        # Always prefer disease over healthy, even at low confidence
        if named:
            primary = max(named, key=lambda d: d.confidence)
            return primary, dets  # return whatever disease was found
        if healthy:
            return max(healthy, key=lambda d: d.confidence), dets
        return None, dets

    def _run_fecal(self, image):
        dets    = self._run("fecal.pt", image)
        named   = [d for d in dets if d.label in FECAL_DISEASES]
        healthy = [d for d in dets if d.label == HEALTHY_LABEL]
        # Always prefer disease over healthy, even at low confidence
        if named:
            primary = max(named, key=lambda d: d.confidence)
            return primary, dets  # return whatever disease was found
        if healthy:
            return max(healthy, key=lambda d: d.confidence), dets
        return None, dets

    def _run_triage(self, image):
        app_dets = self._run("appearance.pt", image)
        beh_dets = self._run("behavior.pt",   image)
        all_dets = app_dets + beh_dets
        # Strictly require high confidence for binary triage models to prevent false positives on backgrounds
        abnormal = [d for d in all_dets if d.label == ABNORMAL_LABEL and d.confidence >= TRIAGE_CONF_THRESH]
        healthy  = [d for d in all_dets if d.label == HEALTHY_LABEL  and d.confidence >= 0.70]
        if abnormal and healthy:
            best_abn = max(abnormal, key=lambda d: d.confidence)
            best_hlt = max(healthy,  key=lambda d: d.confidence)
            return (best_abn if best_abn.confidence >= best_hlt.confidence else best_hlt), all_dets
        if abnormal:
            return max(abnormal, key=lambda d: d.confidence), all_dets
        if healthy:
            return max(healthy,  key=lambda d: d.confidence), all_dets
        return None, all_dets

    def _run_ensemble(self, image):
        general_dets = self._run("general.pt", image)
        bird_conf    = max((d.confidence for d in general_dets), default=0.0)

        # Performance boost for Pi 4: droppings model is only needed if no bird is detected
        if bird_conf < 0.40:
            fecal_dets = self._run("fecal.pt", image)
        else:
            fecal_dets = []

        app_dets     = self._run("appearance.pt", image)
        beh_dets     = self._run("behavior.pt",   image)
        all_dets     = general_dets + fecal_dets + app_dets + beh_dets

        named_g  = [d for d in general_dets if d.label in GENERAL_DISEASES and d.confidence >= DISEASE_CONF_THRESH]
        named_f  = [d for d in fecal_dets   if d.label in FECAL_DISEASES   and d.confidence >= DISEASE_CONF_THRESH]
        healthy_g = [d for d in general_dets if d.label == HEALTHY_LABEL]

        best_g  = max(named_g,   key=lambda d: d.confidence) if named_g  else None
        best_f  = max(named_f,   key=lambda d: d.confidence) if named_f  else None
        best_hg = max(healthy_g, key=lambda d: d.confidence) if healthy_g else None

        bird_conf  = max((d.confidence for d in general_dets), default=0.0)
        fecal_conf = max((d.confidence for d in fecal_dets),   default=0.0)

        if general_dets and best_f is not None and best_f.confidence < 0.88:
            best_f = None

        if best_hg is not None and best_f is None:
            if best_g is None or best_hg.confidence >= best_g.confidence:
                return best_hg, all_dets

        winner = self._route(best_g, best_f, bird_conf, fecal_conf)
        if winner is not None and winner.confidence >= DISEASE_CONF_THRESH:
            return winner, all_dets

        # Check triage status only if confidence is high
        status   = app_dets + beh_dets
        abnormal = [d for d in status if d.label == ABNORMAL_LABEL and d.confidence >= TRIAGE_CONF_THRESH]
        healthy  = [d for d in status + general_dets + fecal_dets if d.label == HEALTHY_LABEL and d.confidence >= 0.70]
        if abnormal:
            return max(abnormal, key=lambda d: d.confidence), all_dets
        if healthy:
            return max(healthy,  key=lambda d: d.confidence), all_dets
        return None, all_dets

    @staticmethod
    def _route(best_g, best_f, bird_conf, fecal_conf):
        if best_g is None: return best_f
        if best_f is None: return best_g
        adj_g = best_g.confidence + (0.18 if bird_conf  >= 0.35 else -0.04)
        adj_f = best_f.confidence + (0.18 if fecal_conf >= bird_conf + 0.08 else -0.10)
        if bird_conf >= 0.40: adj_f -= 0.12
        if bird_conf >= 0.55: adj_f -= 0.08
        if fecal_conf >= 0.50 and bird_conf < 0.35: adj_g -= 0.12
        return best_g if adj_g >= adj_f else best_f

    @staticmethod
    def annotate(frame, det):
        if det is None:
            return
        x1, y1, x2, y2 = map(int, det.box)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        color = LABEL_COLORS.get(det.label, DEFAULT_COLOR)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        tag = f"{det.label}  {det.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        yt = max(y1, th + 10)
        cv2.rectangle(frame, (x1, yt - th - 10), (x1 + tw + 8, yt), color, -1)
        cv2.putText(frame, tag, (x1 + 4, yt - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
