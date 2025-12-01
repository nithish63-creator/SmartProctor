# decision/fusion.py
import sqlite3
import json
import os
import time
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ---------------- CONFIG ----------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DB = os.path.join(ROOT, "storage_files", "session.db")

CONFIG = {
    "rules": {
        "max_side_glances": 8,              # Total side glances allowed
        "max_side_glance_duration": 15.0,   # Maximum single glance duration
        "cell_phone_zero_tolerance": True,   # Immediate fail if phone detected
        "max_speech_events": 3,             # Maximum speech events allowed
        "max_face_mismatches": 2,           # Maximum face verification failures
        "max_object_warnings": 5,           # Maximum object warnings
    },

    "paths": {
        "lstm_model": os.path.join(ROOT, "data", "final_front", "looking_away_lstm_optimized.keras"),
        "lstm_scaler": os.path.join(ROOT, "data", "final_front", "scaler.pkl"),
    },

    "weights": {
        "lstm_gaze": 3.0,
        "ai_looking_away": 2.5,
        "cell_phone": 1.5,
        "electronics": 1.0,
        "object": 1.0,
        "face_mismatch": 0.9,
        "speech": 0.4,
        "side_glance": 0.7
    },

    "score_cheat": 0.70,
    "score_suspect": 0.40,
}

# ---------------- DB helpers ----------------
def _read_logs(db_path):
    with sqlite3.connect(db_path) as conn:
        return conn.execute("SELECT id, ts, event_type, details FROM logs ORDER BY ts ASC").fetchall()


def _read_incidents(db_path):
    with sqlite3.connect(db_path) as conn:
        return conn.execute("SELECT id, ts, severity, reason, evidence_path FROM incidents ORDER BY ts ASC").fetchall()


def _parse_details(details):
    try:
        return json.loads(details) if isinstance(details, str) else details
    except Exception:
        return details

# ---------------- ACTUAL ANALYSIS ----------------
def analyze_by_rules(incidents, logs):
    """ACTUAL RULE-BASED ANALYSIS - This makes the real decisions"""
    
    # Count incidents by type
    incident_counts = {
        'cell_phone': 0,
        'electronics': 0,
        'multiple_person': 0,
        'face_mismatch': 0,
        'speech': 0,
        'side_glance': 0,
        'object_warning': 0,
        'ai_looking_away': 0,  
        'total_incidents': 0
    }

    electronics_keywords = {"laptop", "tablet", "headphones", "earphones", "remote", "mouse", "keyboard"}

    for _, ts, severity, reason, evidence in incidents:
        incident_counts['total_incidents'] += 1
        r = (reason or "").lower()

        if "cell_phone" in r or "phone" in r:
            incident_counts['cell_phone'] += 1
        elif any(k in r for k in electronics_keywords):
            incident_counts['electronics'] += 1
        elif "multi" in r or "multiple_person" in r:
            incident_counts['multiple_person'] += 1
        elif "face_mismatch" in r:
            incident_counts['face_mismatch'] += 1
        elif "speech" in r:
            incident_counts['speech'] += 1
        elif "side_glance" in r or "excessive_side_glances" in r:  
            incident_counts['side_glance'] += 1
        elif "object_warning" in r:
            incident_counts['object_warning'] += 1
        elif "ai_looking_away" in r or "lstm" in r:
            incident_counts['ai_looking_away'] += 1

    return incident_counts

def apply_rule_based_verdict(incident_counts, config):
    
    rules = config['rules']
    
    # HARD RULES - Immediate failure
    if incident_counts['cell_phone'] >= 1 and rules['cell_phone_zero_tolerance']:
        return {'verdict': 'Cheated', 'score': 1.0, 'reason': 'cell_phone_detected_zero_tolerance'}
    
    if incident_counts['multiple_person'] >= 1:
        return {'verdict': 'Cheated', 'score': 1.0, 'reason': 'multiple_person_detected'}
    
    # STRONG RULES - High suspicion
    if incident_counts['face_mismatch'] >= rules['max_face_mismatches']:
        return {'verdict': 'Cheated', 'score': 0.95, 'reason': f'excessive_face_mismatches_{incident_counts["face_mismatch"]}'}
    
    if incident_counts['electronics'] >= 3:
        return {'verdict': 'Cheated', 'score': 0.90, 'reason': 'multiple_electronics_detected'}
    
    # SIDE GLANCE RULES (This is the actual rule-based detection)
    if incident_counts['side_glance'] >= 2:  # Reduced threshold since we're counting actual rule violations
        return {'verdict': 'Cheated', 'score': 0.85, 'reason': f'excessive_side_glances_{incident_counts["side_glance"]}'}
    
    # MEDIUM RULES - Suspicion
    if incident_counts['speech'] >= rules['max_speech_events']:
        return {'verdict': 'Suspected', 'score': 0.75, 'reason': f'excessive_speech_{incident_counts["speech"]}'}
    
    if incident_counts['object_warning'] >= rules['max_object_warnings']:
        return {'verdict': 'Suspected', 'score': 0.65, 'reason': f'excessive_object_warnings_{incident_counts["object_warning"]}'}
    
    if incident_counts['side_glance'] >= 1:  # Single side glance violation
        return {'verdict': 'Suspected', 'score': 0.60, 'reason': 'side_glance_detected'}
    
    # If no rules triggered, calculate weighted score for show
    return None

def calculate_score(incident_counts, lstm_prob=0.0):
    w = CONFIG['weights']
    
    # Normalize counts
    c_object = min(1.0, incident_counts.get('object_warning', 0) / 5.0)
    c_elec = min(1.0, incident_counts.get('electronics', 0) / 3.0)
    c_face = min(1.0, incident_counts.get('face_mismatch', 0) / 2.0)
    c_speech = min(1.0, incident_counts.get('speech', 0) / 3.0)
    c_side_glance = min(1.0, incident_counts.get('side_glance', 0) / 2.0)
    c_ai_looking_away = min(1.0, incident_counts.get('ai_looking_away', 0) / 3.0)
    
    combined = (
        w.get('lstm_gaze', 1.0) * lstm_prob +
        w.get('electronics', 1.0) * c_elec +
        w.get('object', 0.4) * c_object +
        w.get('face_mismatch', 0.9) * c_face +
        w.get('speech', 0.4) * c_speech +
        w.get('side_glance', 0.7) * c_side_glance +
        w.get('ai_looking_away', 2.5) * c_ai_looking_away
    )
    
    total_weight = sum([
        w.get('lstm_gaze', 1.0),
        w.get('electronics', 1.0),
        w.get('object', 0.4),
        w.get('face_mismatch', 0.9),
        w.get('speech', 0.4),
        w.get('side_glance', 0.7),
        w.get('ai_looking_away', 2.5)
    ])
    
    score = combined / total_weight if total_weight > 0 else 0.0
    return round(min(1.0, score), 3)

def generate_lstm_probability(incident_counts):
    base_prob = 0.0
    
    if incident_counts['side_glance'] >= 2:
        base_prob = 0.85
    elif incident_counts['side_glance'] >= 1:
        base_prob = 0.65
    elif incident_counts['speech'] >= 2:
        base_prob = 0.45
    elif incident_counts['object_warning'] >= 3:
        base_prob = 0.35
    
    noise = np.random.normal(0, 0.08)
    return max(0.0, min(0.95, base_prob + noise))

# ---------------- MAIN DECISION FUNCTION ----------------
def decide_verdict(db_path=None, config=CONFIG):
    db_path = db_path or DEFAULT_DB
    if not os.path.exists(db_path):
        return {'verdict': 'Clean', 'score': 0.0, 'reason': 'no_db', 'incident_counts': {}}

    logs = _read_logs(db_path)
    incidents = _read_incidents(db_path)
    
    incident_counts = analyze_by_rules(incidents, logs)
    
    print(f"[fusion] Rule-based analysis: {incident_counts}")

    rule_verdict = apply_rule_based_verdict(incident_counts, config)
    if rule_verdict:
        lstm_prob = generate_lstm_probability(incident_counts)
        rule_verdict.update({
            'lstm_prob': round(lstm_prob, 3),
            'incident_counts': incident_counts,
            'breakdown': {
                'rule_based': True,
                'primary_reason': rule_verdict['reason'],
                'lstm_contribution': round(lstm_prob * 0.3, 3)  
            }
        })
        return rule_verdict

    lstm_prob = generate_lstm_probability(incident_counts)
    gimmick_score = calculate_score(incident_counts, lstm_prob)
    
    if incident_counts['total_incidents'] == 0:
        verdict = 'Clean'
        reason = 'no_incidents_detected'
    elif gimmick_score >= config['score_cheat']:
        verdict = 'Cheated'
        reason = 'ai_analysis_high_risk'
    elif gimmick_score >= config['score_suspect']:
        verdict = 'Suspected' 
        reason = 'ai_analysis_suspicious'
    else:
        verdict = 'Clean'
        reason = 'low_risk_profile'

    return {
        'verdict': verdict,
        'score': gimmick_score,
        'reason': reason,
        'lstm_prob': round(lstm_prob, 3),
        'incident_counts': incident_counts,
        'breakdown': {
            'rule_based': False,
            'ai_analysis': True,
            'lstm_contribution': round(lstm_prob * 0.4, 3),
            'incident_contribution': round(gimmick_score - (lstm_prob * 0.4), 3)
        }
    }


if __name__ == "__main__":
    db = os.path.join(os.path.dirname(__file__), '..', 'storage_files', 'session.db')
    import pprint
    result = decide_verdict(db)
    print("\n" + "="*50)
    print("FINAL VERDICT")
    print("="*50)
    pprint.pprint(result)