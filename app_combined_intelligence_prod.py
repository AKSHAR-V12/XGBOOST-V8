# app_combined_intelligence_prod.py
"""
Deployment-ready FastAPI app combining:
- XGBoost model inference (joblib .pkl)
- Deterministic humanized coaching engine
- Slider constraints, delta detection, per-user prev storage (in-memory; Redis optional)
- WebSocket for live updates (0.5s tick)
- Robust logging, CORS, graceful startup/shutdown

NOTE: Only change from your original file:
- Added human-friendly "message" field to both REST /predict response and WebSocket response.
  The message now provides a clear sentence for both label == 1 and label == 0.
All other code is unchanged.3+
"""

import os
import json
import time
import math
import logging
import joblib
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from fastapi import FastAPI, WebSocket, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np

# -------------------------
# Configuration (env-friendly)
# -------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "models/muscle_growth_model.pkl")
FEATURES_PATH = os.getenv("FEATURES_PATH", "feature_names.json")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
USE_REDIS = os.getenv("USE_REDIS", "false").lower() in ("1", "true", "yes")
WEBSOCKET_TICK = float(os.getenv("WEBSOCKET_TICK", "0.5"))
MAX_WS_CONNS = int(os.getenv("MAX_WS_CONNS", "200"))

# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("app_combined_intelligence")

# -------------------------
# Load model and feature order
# -------------------------
if not os.path.exists(MODEL_PATH):
    logger.error("Model file not found at %s", MODEL_PATH)
    raise SystemExit("Model file missing")

if not os.path.exists(FEATURES_PATH):
    logger.error("Feature names file not found at %s", FEATURES_PATH)
    raise SystemExit("Feature names file missing")

try:
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, "r") as f:
        FEATURE_NAMES = json.load(f)
    logger.info("Loaded model and feature names successfully")
except Exception as e:
    logger.exception("Failed to load model or features: %s", e)
    raise SystemExit(e)

# -------------------------
# Feature metadata (frontend sliders)
# -------------------------
FEATURE_METADATA = {
    "sleep_hours": {"min": 0.0, "max": 12.0, "step": 0.25, "cadence": "daily", "delta_threshold": 0.5},
    "training_frequency_per_week": {"min": 0.0, "max": 14.0, "step": 0.5, "cadence": "weekly", "delta_threshold": 0.5},
    "calorie_surplus": {"min": -2000, "max": 2000, "step": 50, "cadence": "daily", "delta_threshold": 100},
    "progressive_overload_score": {"min": 0.0, "max": 10.0, "step": 0.5, "cadence": "weekly", "delta_threshold": 0.5},
    "training_experience_years": {"min": 0.0, "max": 80.0, "step": 0.25, "cadence": "yearly", "delta_threshold": 0.25},
    "body_fat_percentage": {"min": 3.0, "max": 60.0, "step": 0.5, "cadence": "monthly", "delta_threshold": 1.0},
    "protein_intake_g": {"min": 0.0, "max": 1000.0, "step": 5.0, "cadence": "daily", "delta_threshold": 10},
    "stress_level": {"min": 0.0, "max": 10.0, "step": 0.5, "cadence": "daily", "delta_threshold": 0.5}
}

# -------------------------
# Deterministic engine dataclasses & helpers (from your upgraded engine)
# -------------------------
@dataclass(frozen=True)
class Thresholds:
    sleep_low: float = 6.0
    sleep_very_low: float = 4.0
    sleep_high: float = 9.0
    protein_low_per_kg: float = 1.6
    protein_target_per_kg: float = 2.2
    protein_high_per_kg: float = 2.8
    overload_very_low: float = 3.0
    overload_moderate: float = 6.0
    freq_low: float = 2.0
    freq_high: float = 6.0
    calorie_small_surplus: float = 200.0
    calorie_good_surplus_low: float = 250.0
    calorie_good_surplus_high: float = 500.0
    calorie_very_high_surplus: float = 1500.0
    bodyfat_very_low: float = 6.0
    bodyfat_high: float = 30.0
    stress_moderate: float = 4.0
    stress_high: float = 7.0
    experience_novice: float = 0.5
    experience_early_intermediate: float = 2.0
    experience_intermediate: float = 5.0

@dataclass
class AdviceItem:
    tag: str
    advice: str
    priority: int
    severity: float = 0.0
    category: str = "general"
    evidence: Dict[str, Any] = field(default_factory=dict)
    tone: str = "neutral"
    actionability: int = 70

@dataclass
class CoachingProfile:
    tone: str = "friendly"
    max_items: int = 4
    include_rationale: bool = True
    include_positive_reinforcement: bool = True
    include_interaction_notes: bool = True
    humanize_language: bool = True

def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if isinstance(value, bool):
            return float(int(value))
        return float(value)
    except (TypeError, ValueError):
        return None

def make_item(tag: str, advice: str, priority: int, severity: float = 0.0,
              category: str = "general", evidence: Optional[Dict[str, Any]] = None,
              tone: str = "neutral", actionability: int = 70) -> AdviceItem:
    return AdviceItem(tag=tag, advice=advice, priority=priority, severity=severity,
                      category=category, evidence=evidence or {}, tone=tone, actionability=actionability)

def extract_signals(inputs: Dict[str, Any], thresholds: Thresholds) -> Dict[str, float]:
    sleep = safe_float(inputs.get("sleep_hours"))
    stress = safe_float(inputs.get("stress_level"))
    overload = safe_float(inputs.get("progressive_overload_score"))
    freq = safe_float(inputs.get("training_frequency_per_week"))
    calorie_surplus = safe_float(inputs.get("calorie_surplus"))
    bodyfat = safe_float(inputs.get("body_fat_percentage"))
    training_years = safe_float(inputs.get("training_experience_years"))
    protein = safe_float(inputs.get("protein_intake_g"))
    bw = safe_float(inputs.get("body_weight_kg")) or 70.0
    protein_per_kg = (protein / max(1.0, bw)) if protein is not None else None

    signals: Dict[str, float] = {
        "sleep": sleep if sleep is not None else float("nan"),
        "stress": stress if stress is not None else float("nan"),
        "overload": overload if overload is not None else float("nan"),
        "freq": freq if freq is not None else float("nan"),
        "calorie_surplus": calorie_surplus if calorie_surplus is not None else float("nan"),
        "bodyfat": bodyfat if bodyfat is not None else float("nan"),
        "training_years": training_years if training_years is not None else float("nan"),
        "protein_per_kg": protein_per_kg if protein_per_kg is not None else float("nan"),
    }

    signals["sleep_deficit"] = max(0.0, thresholds.sleep_low - sleep) if sleep is not None else float("nan")
    signals["sleep_severe_deficit"] = max(0.0, thresholds.sleep_very_low - sleep) if sleep is not None else float("nan")
    signals["stress_excess"] = max(0.0, stress - thresholds.stress_high) if stress is not None else float("nan")
    signals["stress_moderate_excess"] = max(0.0, stress - thresholds.stress_moderate) if stress is not None else float("nan")
    signals["overload_gap"] = max(0.0, thresholds.overload_moderate - overload) if overload is not None else float("nan")
    signals["freq_gap"] = max(0.0, thresholds.freq_low - freq) if freq is not None else float("nan")
    signals["protein_gap"] = max(0.0, thresholds.protein_low_per_kg - protein_per_kg) if protein_per_kg is not None else float("nan")
    signals["calorie_gap"] = max(0.0, thresholds.calorie_good_surplus_low - calorie_surplus) if calorie_surplus is not None else float("nan")
    signals["calorie_excess"] = max(0.0, calorie_surplus - thresholds.calorie_very_high_surplus) if calorie_surplus is not None else float("nan")
    signals["bodyfat_excess"] = max(0.0, bodyfat - thresholds.bodyfat_high) if bodyfat is not None else float("nan")
    signals["bodyfat_too_low"] = max(0.0, thresholds.bodyfat_very_low - bodyfat) if bodyfat is not None else float("nan")
    return signals

# Per-feature advice functions (concise)
def sleep_advice(signals, thresholds, profile):
    items = []
    sleep = signals.get("sleep")
    if math.isnan(sleep):
        return items
    deficit = signals.get("sleep_deficit", 0.0)
    severe = signals.get("sleep_severe_deficit", 0.0)
    if sleep < thresholds.sleep_very_low:
        items.append(make_item("sleep_critical",
            "Sleep compromised. Aim 7–8 h, cut late caffeine, set wind-down routine.",
            98, severity=0.95+0.05*min(1.0,severe), category="recovery", evidence={"sleep_hours": sleep}, tone="firm", actionability=98))
    elif sleep < thresholds.sleep_low:
        items.append(make_item("sleep_low",
            "Sleep slightly low. Add 30–60 min nightly to improve recovery and training response.",
            90, severity=0.65+0.08*min(1.0,deficit), category="recovery", evidence={"sleep_hours": sleep}, tone="friendly", actionability=92))
    elif sleep <= thresholds.sleep_high:
        items.append(make_item("sleep_good",
            "Sleep in healthy range. Maintain consistency.",
            72, severity=0.18, category="recovery", evidence={"sleep_hours": sleep}, tone="positive", actionability=75))
    else:
        items.append(make_item("sleep_high",
            "Sleep high side — if not planned recovery, check daytime activity and routine.",
            62, severity=0.12, category="recovery", evidence={"sleep_hours": sleep}, tone="neutral", actionability=60))
    return items

def experience_advice(signals, thresholds, profile):
    items = []
    te = signals.get("training_years")
    if math.isnan(te):
        return items
    if te < thresholds.experience_novice:
        items.append(make_item("novice","Newbie — focus on technique and consistency.",95,category="experience",evidence={"training_experience_years":te},tone="encouraging"))
    elif te < thresholds.experience_early_intermediate:
        items.append(make_item("early_intermediate","Early intermediate — structured overload + recovery discipline will help.",90,category="experience",evidence={"training_experience_years":te},tone="encouraging"))
    elif te < thresholds.experience_intermediate:
        items.append(make_item("intermediate","Intermediate — progress needs smarter planning: overload, volume, nutrition.",84,category="experience",evidence={"training_experience_years":te},tone="neutral"))
    else:
        items.append(make_item("advanced","Advanced — gains slower; periodization and fatigue management are key.",80,category="experience",evidence={"training_experience_years":te},tone="firm"))
    return items

def protein_advice(signals, thresholds, profile):
    items = []
    ppk = signals.get("protein_per_kg")
    if math.isnan(ppk):
        return items
    gap = signals.get("protein_gap",0.0)
    if ppk < thresholds.protein_low_per_kg:
        items.append(make_item("protein_low","Protein low. Target 1.6–2.2 g/kg; distribute 20–40 g per meal.",96, severity=0.85+0.05*min(1.0,gap), category="nutrition", evidence={"protein_per_kg":ppk}, tone="firm"))
    elif ppk <= thresholds.protein_target_per_kg:
        items.append(make_item("protein_ok","Protein in target range. Maintain distribution and consistency.",76,category="nutrition",evidence={"protein_per_kg":ppk},tone="positive"))
    elif ppk <= thresholds.protein_high_per_kg:
        items.append(make_item("protein_high_ok","Protein upper-optimal side; ensure overall calorie balance.",72,category="nutrition",evidence={"protein_per_kg":ppk},tone="neutral"))
    else:
        items.append(make_item("protein_very_high","Protein very high; check digestion and overall balance.",68,category="nutrition",evidence={"protein_per_kg":ppk},tone="neutral"))
    return items

def overload_advice(signals, thresholds, profile):
    items = []
    pos = signals.get("overload")
    if math.isnan(pos):
        return items
    gap = signals.get("overload_gap",0.0)
    if pos < thresholds.overload_very_low:
        items.append(make_item("overload_very_low","Progressive overload weak. Start small weekly increases and track.",98, severity=0.90+0.03*min(1.0,gap), category="training", evidence={"progressive_overload_score":pos}, tone="firm"))
    elif pos < thresholds.overload_moderate:
        items.append(make_item("overload_ok","Overload okay; improve microprogressions and logging.",86,category="training",evidence={"progressive_overload_score":pos},tone="neutral"))
    else:
        items.append(make_item("overload_good","Overload strong; focus on recovery and fatigue management.",72,category="training",evidence={"progressive_overload_score":pos},tone="positive"))
    return items

def frequency_advice(signals, thresholds, profile):
    items = []
    tf = signals.get("freq")
    if math.isnan(tf):
        return items
    if tf < thresholds.freq_low:
        items.append(make_item("freq_too_low","Training frequency low. Aim 2–4 quality sessions/week.",90,category="training",evidence={"training_frequency_per_week":tf},tone="firm"))
    elif tf <= thresholds.freq_high:
        items.append(make_item("freq_good","Training frequency reasonable. Balance volume and recovery.",74,category="training",evidence={"training_frequency_per_week":tf},tone="positive"))
    else:
        items.append(make_item("freq_high","High frequency — periodize intensity if performance or sleep dips.",82,category="training",evidence={"training_frequency_per_week":tf},tone="firm"))
    return items

def calorie_advice(signals, thresholds, profile):
    items = []
    cs = signals.get("calorie_surplus")
    if math.isnan(cs):
        return items
    gap = signals.get("calorie_gap",0.0)
    excess = signals.get("calorie_excess",0.0)
    if cs < thresholds.calorie_small_surplus:
        items.append(make_item("calorie_small","Small surplus — consider +250–500 kcal/day for steady gains.",89, severity=0.68+0.04*min(1.0,gap/250.0), category="nutrition", evidence={"calorie_surplus":cs}, tone="firm"))
    elif cs > thresholds.calorie_very_high_surplus:
        items.append(make_item("calorie_very_high","Very aggressive surplus — risk of excess fat gain; reduce to 300–500 kcal/day.",95, severity=0.90+0.03*min(1.0,excess/500.0), category="nutrition", evidence={"calorie_surplus":cs}, tone="firm"))
    else:
        items.append(make_item("calorie_ok","Calorie surplus in workable range for lean gains.",70,category="nutrition",evidence={"calorie_surplus":cs},tone="positive"))
    return items

def bodyfat_advice(signals, thresholds, profile):
    items = []
    bf = signals.get("bodyfat")
    if math.isnan(bf):
        return items
    if bf > thresholds.bodyfat_high:
        items.append(make_item("bf_high","Body fat higher side — prefer modest surplus + conditioning.",85,category="body_comp",evidence={"body_fat_percentage":bf},tone="firm"))
    elif bf < thresholds.bodyfat_very_low:
        items.append(make_item("bf_very_low","Very low body fat — manage calories and recovery carefully.",95,category="body_comp",evidence={"body_fat_percentage":bf},tone="firm"))
    else:
        items.append(make_item("bf_ok","Body fat reasonable — focus on training quality and recovery.",70,category="body_comp",evidence={"body_fat_percentage":bf},tone="positive"))
    return items

def stress_advice(signals, thresholds, profile):
    items = []
    sl = signals.get("stress")
    if math.isnan(sl):
        return items
    if sl > thresholds.stress_high:
        items.append(make_item("stress_high","Stress high — de-escalate workload and prioritize sleep.",97,category="recovery",evidence={"stress_level":sl},tone="firm"))
    elif sl > thresholds.stress_moderate:
        items.append(make_item("stress_moderate","Stress moderate — monitor sleep and fatigue.",80,category="recovery",evidence={"stress_level":sl},tone="neutral"))
    else:
        items.append(make_item("stress_ok","Stress acceptable — supports recovery if other variables aligned.",72,category="recovery",evidence={"stress_level":sl},tone="positive"))
    return items

def interaction_advice(signals, thresholds, profile):
    items = []
    sleep = signals.get("sleep"); stress = signals.get("stress"); overload = signals.get("overload")
    freq = signals.get("freq"); cs = signals.get("calorie_surplus"); ppk = signals.get("protein_per_kg")
    bf = signals.get("bodyfat"); te = signals.get("training_years")
    if not math.isnan(sleep) and not math.isnan(stress):
        if sleep < thresholds.sleep_low and stress > thresholds.stress_high:
            items.append(make_item("recovery_collision","Low sleep + high stress squeeze recovery — control fatigue before heavy progression.",99,category="interaction",evidence={"sleep_hours":sleep,"stress_level":stress},tone="firm"))
        elif sleep < thresholds.sleep_low and stress > thresholds.stress_moderate:
            items.append(make_item("recovery_pressure","Low sleep + moderate stress shrink recovery margin — control workload.",94,category="interaction",evidence={"sleep_hours":sleep,"stress_level":stress},tone="neutral"))
    if not math.isnan(freq) and not math.isnan(overload):
        if freq > thresholds.freq_high and overload < thresholds.overload_moderate:
            items.append(make_item("junk_volume_risk","High frequency but weak overload — risk of junk volume; prioritize productive stimulus.",93,category="interaction",evidence={"training_frequency_per_week":freq,"progressive_overload_score":overload},tone="firm"))
    if not math.isnan(cs) and not math.isnan(ppk):
        if cs >= thresholds.calorie_good_surplus_low and ppk < thresholds.protein_low_per_kg:
            items.append(make_item("calories_without_protein","Calories okay but protein low — fix protein distribution to sharpen lean gains.",96,category="interaction",evidence={"calorie_surplus":cs,"protein_per_kg":ppk},tone="firm"))
    if not math.isnan(bf):
        if bf < thresholds.bodyfat_very_low:
            if (not math.isnan(freq) and freq >= thresholds.freq_high) or (not math.isnan(stress) and stress > thresholds.stress_high):
                items.append(make_item("fragile_recovery_state","Very low body fat + high frequency/stress creates fragile recovery — be cautious.",97,category="interaction",evidence={"body_fat_percentage":bf,"training_frequency_per_week":freq,"stress_level":stress},tone="firm"))
    if not math.isnan(te) and not math.isnan(overload):
        if te < thresholds.experience_novice and overload < thresholds.overload_moderate:
            items.append(make_item("novice_structure","Beginner + weak overload — big opportunity: simple plan + tracking yields fast improvements.",88,category="interaction",evidence={"training_experience_years":te,"progressive_overload_score":overload},tone="encouraging"))
    return items

def positive_reinforcement(signals, thresholds, profile):
    items = []
    if not profile.include_positive_reinforcement:
        return items
    sleep = signals.get("sleep"); protein = signals.get("protein_per_kg")
    if not math.isnan(sleep) and 7.0 <= sleep <= thresholds.sleep_high:
        items.append(make_item("sleep_strength","Sleep foundation solid — good base for adaptation.",58,category="positive",evidence={"sleep_hours":sleep},tone="positive"))
    if not math.isnan(protein) and 1.8 <= protein <= 2.2:
        items.append(make_item("protein_strength","Protein structure strong — supports muscle-building.",56,category="positive",evidence={"protein_per_kg":protein},tone="positive"))
    return items

# Delta detection
def detect_deltas(prev: Dict[str, float], curr: Dict[str, float]) -> List[Tuple[str, float, float]]:
    deltas = []
    for feat, meta in FEATURE_METADATA.items():
        thresh = meta.get("delta_threshold", 0.0)
        p = prev.get(feat)
        c = curr.get(feat)
        if p is None or c is None:
            continue
        if math.isclose(p, c, rel_tol=0.0, abs_tol=0.0):
            continue
        if abs(c - p) >= thresh:
            deltas.append((feat, p, c))
    return deltas

def generate_delta_messages(deltas: List[Tuple[str, float, float]]) -> List[Dict[str, Any]]:
    msgs = []
    deltas_sorted = sorted(deltas, key=lambda x: abs(x[2] - x[1]), reverse=True)
    for feat, p, c in deltas_sorted[:2]:
        label = feat.replace("_"," ").title()
        diff = c - p
        if diff > 0:
            if feat == "training_experience_years":
                text = f"Experience increased by {diff:.2f} years — you're progressing; consistency will compound gains."
            elif feat == "sleep_hours":
                text = f"Sleep increased by {diff:.1f} h. Keep consistent bedtime to reach 7–8 h."
            elif feat == "protein_intake_g":
                text = f"Protein up by {diff:.0f} g. Maintain for 2 weeks; aim 1.6–2.2 g/kg."
            else:
                text = f"{label} increased by {diff:.2f}. Keep consistent and monitor trend."
            msgs.append({"type":"delta_positive","tag":feat,"text":text})
        else:
            if feat == "training_experience_years":
                text = f"Experience decreased by {abs(diff):.2f} years — check input for typos."
            else:
                text = f"{label} decreased by {abs(diff):.2f}. Consider adjusting plan."
            msgs.append({"type":"delta_negative","tag":feat,"text":text})
    if len(deltas_sorted) >= 2:
        f1,p1,c1 = deltas_sorted[0]; f2,p2,c2 = deltas_sorted[1]
        if (c1-p1)>0 and (c2-p2)>0:
            msgs.insert(0, {"type":"delta_synergy","tag":"combo","text":"Good combo — these changes together improve recovery and adaptation. Keep consistent for 2 weeks."})
    return msgs

# Simple in-memory prev store (replace with Redis in production)
IN_MEMORY_PREV: Dict[str, Dict[str, Any]] = {}
def get_prev(user_id: str) -> Dict[str, Any]:
    return IN_MEMORY_PREV.get(user_id, {})
def set_prev(user_id: str, payload: Dict[str, Any]):
    IN_MEMORY_PREV[user_id] = payload

# Pydantic schema
class InputData(BaseModel):
    sleep_hours: float = Field(..., ge=FEATURE_METADATA["sleep_hours"]["min"], le=FEATURE_METADATA["sleep_hours"]["max"])
    training_frequency_per_week: float = Field(..., ge=FEATURE_METADATA["training_frequency_per_week"]["min"], le=FEATURE_METADATA["training_frequency_per_week"]["max"])
    calorie_surplus: float = Field(..., ge=FEATURE_METADATA["calorie_surplus"]["min"], le=FEATURE_METADATA["calorie_surplus"]["max"])
    progressive_overload_score: float = Field(..., ge=FEATURE_METADATA["progressive_overload_score"]["min"], le=FEATURE_METADATA["progressive_overload_score"]["max"])
    training_experience_years: float = Field(..., ge=FEATURE_METADATA["training_experience_years"]["min"], le=FEATURE_METADATA["training_experience_years"]["max"])
    body_fat_percentage: float = Field(..., ge=FEATURE_METADATA["body_fat_percentage"]["min"], le=FEATURE_METADATA["body_fat_percentage"]["max"])
    protein_intake_g: float = Field(..., ge=FEATURE_METADATA["protein_intake_g"]["min"], le=FEATURE_METADATA["protein_intake_g"]["max"])
    stress_level: float = Field(..., ge=FEATURE_METADATA["stress_level"]["min"], le=FEATURE_METADATA["stress_level"]["max"])
    body_weight_kg: Optional[float] = Field(None, ge=30, le=300)
    user_type: Optional[str] = "normal"

# FastAPI app
app = FastAPI(title="Muscle Growth Predictor + Humanized Deterministic Intelligence (Prod)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/feature-metadata")
def feature_metadata():
    return FEATURE_METADATA

@app.post("/predict")
def predict(payload: InputData, user_id: str = Query("anon", description="unique user id for session prev storage")):
    curr = payload.dict()
    for f, meta in FEATURE_METADATA.items():
        v = curr.get(f)
        if v is None:
            continue
        curr[f] = max(meta["min"], min(meta["max"], v))
        step = meta.get("step")
        if step:
            curr[f] = round(round(curr[f] / step) * step, 3)

    try:
        row = [curr.get(f, 0) for f in FEATURE_NAMES]
        X = np.array(row).reshape(1, -1)
        proba = float(model.predict_proba(X)[0, 1])
    except Exception as e:
        logger.exception("Model inference failed: %s", e)
        raise HTTPException(status_code=500, detail="Model inference error")

    label = int(proba > 0.5)

    # -------------------------
    # Added human-friendly message (only change)
    # - label == 1: positive, concise message
    # - label == 0: constructive guidance message
    # -------------------------
    if label == 1:
        message = "Most likely to gain muscle — current plan and nutrition support adaptation."
    else:
        message = "Less likely to gain muscle — consider improving protein, progressive overload, and recovery."

    thresholds = Thresholds()
    profile = CoachingProfile()
    signals = extract_signals(curr, thresholds)
    advice_items: List[AdviceItem] = []
    advice_items += sleep_advice(signals, thresholds, profile)
    advice_items += experience_advice(signals, thresholds, profile)
    advice_items += protein_advice(signals, thresholds, profile)
    advice_items += overload_advice(signals, thresholds, profile)
    advice_items += frequency_advice(signals, thresholds, profile)
    advice_items += calorie_advice(signals, thresholds, profile)
    advice_items += bodyfat_advice(signals, thresholds, profile)
    advice_items += stress_advice(signals, thresholds, profile)
    if profile.include_interaction_notes:
        advice_items += interaction_advice(signals, thresholds, profile)
    if profile.include_positive_reinforcement:
        advice_items += positive_reinforcement(signals, thresholds, profile)

    advice_items_sorted = sorted(advice_items, key=lambda x: x.priority, reverse=True)[:profile.max_items]
    advice_out = [a.__dict__ for a in advice_items_sorted]

    prev = get_prev(user_id)
    deltas = detect_deltas(prev, curr)
    delta_msgs = generate_delta_messages(deltas)
    set_prev(user_id, curr)

    return {
        "probability": proba,
        "label": label,
        "message": message,   # <-- newly added human-friendly message
        "advice": advice_out,
        "delta_messages": delta_msgs,
        "feature_order": FEATURE_NAMES
    }

# WebSocket for live updates
active_ws = set()
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    conn_id = id(ws)
    if len(active_ws) >= MAX_WS_CONNS:
        await ws.send_json({"error": "too many connections"})
        await ws.close()
        return
    active_ws.add(conn_id)
    try:
        while True:
            data = await ws.receive_json()
            user_id = data.get("user_id", "anon")
            try:
                payload = InputData(**data)
            except Exception as e:
                await ws.send_json({"error": "invalid input", "detail": str(e)})
                continue
            curr = payload.dict()
            for f, meta in FEATURE_METADATA.items():
                v = curr.get(f)
                if v is None:
                    continue
                curr[f] = max(meta["min"], min(meta["max"], v))
                step = meta.get("step")
                if step:
                    curr[f] = round(round(curr[f] / step) * step, 3)
            try:
                row = [curr.get(f, 0) for f in FEATURE_NAMES]
                X = np.array(row).reshape(1, -1)
                proba = float(model.predict_proba(X)[0, 1])
            except Exception as e:
                await ws.send_json({"error": "model error", "detail": str(e)})
                continue
            thresholds = Thresholds()
            profile = CoachingProfile()
            signals = extract_signals(curr, thresholds)
            advice_items = []
            advice_items += sleep_advice(signals, thresholds, profile)
            advice_items += experience_advice(signals, thresholds, profile)
            advice_items += protein_advice(signals, thresholds, profile)
            advice_items += overload_advice(signals, thresholds, profile)
            advice_items += frequency_advice(signals, thresholds, profile)
            advice_items += calorie_advice(signals, thresholds, profile)
            advice_items += bodyfat_advice(signals, thresholds, profile)
            advice_items += stress_advice(signals, thresholds, profile)
            if profile.include_interaction_notes:
                advice_items += interaction_advice(signals, thresholds, profile)
            if profile.include_positive_reinforcement:
                advice_items += positive_reinforcement(signals, thresholds, profile)
            advice_items_sorted = sorted(advice_items, key=lambda x: x.priority, reverse=True)[:profile.max_items]
            advice_out = [a.__dict__ for a in advice_items_sorted]
            prev = get_prev(user_id)
            deltas = detect_deltas(prev, curr)
            delta_msgs = generate_delta_messages(deltas)
            set_prev(user_id, curr)

            # -------------------------
            # Added human-friendly message to WebSocket response as well
            # -------------------------
            ws_label = int(proba > 0.5)
            if ws_label == 1:
                ws_message = "Most likely to gain muscle — current plan and nutrition support adaptation."
            else:
                ws_message = "Less likely to gain muscle — consider improving protein, progressive overload, and recovery."

            await ws.send_json({
                "probability": proba,
                "label": ws_label,
                "message": ws_message,   # <-- newly added human-friendly message for WS
                "advice": advice_out,
                "delta_messages": delta_msgs,
                "timestamp": time.time()
            })
            await asyncio.sleep(WEBSOCKET_TICK)
    except Exception as e:
        logger.debug("WebSocket closed: %s", e)
    finally:
        active_ws.discard(conn_id)
        try:
            await ws.close()
        except Exception:
            pass
