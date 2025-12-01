# utils/video_utils.py
import cv2
import os
import time

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
STORAGE_DIR = os.path.join(BASE_DIR, '..', 'storage_files')
EVIDENCE_DIR = os.path.join(STORAGE_DIR, 'evidence')
os.makedirs(EVIDENCE_DIR, exist_ok=True)

def draw_label(frame, text, pos=(10,30), color=(0,255,0)):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

def save_snapshot(frame, prefix="snapshot"):
    fname = os.path.join(EVIDENCE_DIR, f"{prefix}_{int(time.time())}.jpg")
    cv2.imwrite(fname, frame)
    return fname
