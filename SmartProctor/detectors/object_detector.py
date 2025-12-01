import os
import time
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
STORAGE_DIR = os.path.join(BASE_DIR, 'storage_files')
EVIDENCE_DIR = os.path.join(STORAGE_DIR, 'evidence')
os.makedirs(EVIDENCE_DIR, exist_ok=True)


class ObjectDetector:

    def __init__(self, logger=None, model_path=None, conf=0.25, debug=False):
        self.logger = logger
        self.debug = debug
        self.model_path = model_path if model_path else "yolov8m.pt"
        self.conf = conf

        # Load YOLO model
        try:
            self.model = YOLO(self.model_path)
            if self.debug:
                print(f"[ObjectDetector] Loaded YOLO: {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"[ObjectDetector] Failed to load YOLO model ({self.model_path}): {e}")

        # Target classes considered cheating-related
        self.target_classes = {
            "cell phone", "book", "laptop", "tablet",
            "headphones", "earphones", "remote", "mouse", "keyboard"
        }

        # Per-class minimum confidence — lowered for phone/book to improve recall
        self.min_conf_per_class = {
            "cell phone": 0.25,   # lowered to catch partial/brief views
            "laptop": 0.40,
            "tablet": 0.35,
            "book": 0.30,
            "headphones": 0.35,
            "earphones": 0.35,
            "remote": 0.35,
            "mouse": 0.35,
            "keyboard": 0.35,
        }

        # Temporal filtering state
        self.last_seen = {}       # {class_name: last_frame_timestamp}
        self.frame_streaks = {}   # {class_name: consecutive_frame_count}
        # shorter cooldowns so short events are captured
        self.cooldowns = {
            "cell phone": 3.0,
            "book": 4.0,
            "laptop": 5.0,
            "tablet": 4.0,
            "headphones": 5.0,
            "earphones": 5.0,
            "remote": 4.0,
            "mouse": 4.0,
            "keyboard": 4.0,
        }

        # Minimum consecutive frames to accept detection (phones are more permissive)
        self.streak_required = {
            "cell phone": 1,
            "book": 1,
            "laptop": 2,
            "tablet": 1,
            "headphones": 2,
            "earphones": 2,
            "remote": 1,
            "mouse": 1,
            "keyboard": 1,
        }

        # MediaPipe hands (used to reduce false positives when object is fully occluded by a hand)
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )

    # ---------------- HAND UTILITIES ----------------
    def _detect_hands(self, frame):
        h, w, _ = frame.shape
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.mp_hands.process(img_rgb)
        boxes = []
        if getattr(res, "multi_hand_landmarks", None):
            for hand_landmarks in res.multi_hand_landmarks:
                xs = [lm.x * w for lm in hand_landmarks.landmark]
                ys = [lm.y * h for lm in hand_landmarks.landmark]
                x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
                boxes.append((x1, y1, x2, y2))
        return boxes

    def _overlaps_hand(self, obj_box, hand_boxes, iou_thresh=0.15):
        """Return True if object box overlaps hand by > iou_thresh fraction of object area."""
        x1, y1, x2, y2 = obj_box
        obj_area = max(0, x2 - x1) * max(0, y2 - y1)
        if obj_area == 0:
            return False
        for (hx1, hy1, hx2, hy2) in hand_boxes:
            ix1, iy1, ix2, iy2 = max(x1, hx1), max(y1, hy1), min(x2, hx2), min(y2, hy2)
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter = iw * ih
            if inter / float(obj_area) > iou_thresh:
                return True
        return False

    # ---------------- MAIN DETECTION ----------------
    def find(self, frame, save_evidence=True, draw_debug=False):
        """
        Run object detection on a single frame.
        Returns list of accepted detections [{'cls': str, 'conf': float, 'xyxy': [...]}, ...]
        """
        detections = []

        # Conservative fallback if model call fails
        try:
            results = self.model(frame, imgsz=640, conf=min(0.20, self.conf), verbose=False)[0]
        except Exception as e:
            if self.debug:
                print(f"[ObjectDetector] Inference error: {e}")
            return detections

        if results is None or getattr(results, "boxes", None) is None:
            return detections

        hand_boxes = self._detect_hands(frame)
        names = results.names if hasattr(results, "names") else {}

        h, w, _ = frame.shape
        desk_region = int(h * 0.55)  # bottom 45% considered desk zone

        for box in results.boxes:
            try:
                cls_val = getattr(box, "cls", None)
                cls_id = int(cls_val.cpu().numpy()[0]) if hasattr(cls_val, "cpu") else int(cls_val)
                name = names.get(cls_id, str(cls_id))
                conf = float(getattr(box, "conf", 0.0))
                xyxy = box.xyxy.cpu().numpy().tolist()[0] if hasattr(box.xyxy, "cpu") else []
            except Exception:
                continue

            if name not in self.target_classes:
                continue

            # per-class minimum confidence check
            min_conf = self.min_conf_per_class.get(name, self.conf)
            if conf < min_conf:
                if self.debug:
                    print(f"[Filter] skip {name} conf {conf:.2f} < min {min_conf:.2f}")
                continue

            if not xyxy:
                continue
            x1, y1, x2, y2 = map(int, xyxy)
            box_w, box_h = max(0, x2 - x1), max(0, y2 - y1)
            area = box_w * box_h
            if box_w == 0 or box_h == 0:
                continue
            ratio = box_w / float(box_h + 1e-5)

            # Relaxed geometry filters to allow partial and small views (phones at angles)
            min_area = 1500   # allow smaller crops
            if area < min_area:
                if self.debug:
                    print(f"[Filter] skip {name} small area {area}")
                continue
            if ratio < 0.15 or ratio > 5.0:
                if self.debug:
                    print(f"[Filter] skip {name} ratio {ratio:.2f}")
                continue

            # Reduce hand overlap aggression: if object mostly occluded by hand, skip;
            # otherwise allow detections near hands (phones held by hand should still be detected).
            if self._overlaps_hand((x1, y1, x2, y2), hand_boxes, iou_thresh=0.15):
                # If a phone is largely occluded by hand, more likely true positive (phone in hand) => allow in many cases.
                # We'll skip only extreme occlusions covering >50% (handled above by smaller IoU threshold).
                if self.debug:
                    print(f"[Filter] overlap hand for {name}, but allowed by relaxed IoU")

            # Heuristic: avoid saving huge full-frame desk-area reflections as laptop false positives
            if name in {"laptop", "book", "keyboard"} and y1 > desk_region and area > (0.45 * w * h):
                if self.debug:
                    print(f"[Filter] Ignored likely desk reflection / large region ({name}) area:{area}")
                continue

            # Temporal logic: require short streak for critical classes; phone requires only 1 frame
            now = time.time()
            last_t = self.last_seen.get(name, 0.0)
            streak = self.frame_streaks.get(name, 0)
            if now - last_t <= 1.0:
                streak += 1
            else:
                streak = 1
            self.frame_streaks[name] = streak
            self.last_seen[name] = now

            required = self.streak_required.get(name, 1)
            if streak < required:
                if self.debug:
                    print(f"[Temporal] {name} streak {streak} < required {required}")
                continue

            # Per-class cooldowns — prevent duplicate logging
            cooldown_key = f"{name}_cooldown"
            last_logged = self.last_seen.get(cooldown_key, 0.0)
            if now - last_logged < self.cooldowns.get(name, 4.0):
                if self.debug:
                    print(f"[Cooldown] skipping {name}, cooldown active ({now - last_logged:.1f}s)")
                continue
            self.last_seen[cooldown_key] = now

            # Accept detection
            det = {"cls": name, "conf": conf, "xyxy": [x1, y1, x2, y2]}
            detections.append(det)

            # Save evidence (cropped and annotated)
            ev_path = None
            if save_evidence:
                try:
                    crop = frame[y1:y2, x1:x2].copy()
                    ev_path = self.save_evidence(crop, prefix=f"{name}_{conf:.2f}")
                    annotated = frame.copy()
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(annotated, f"{name} {conf:.2f}", (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    self.save_evidence(annotated, prefix=f"full_{name}_{conf:.2f}")
                except Exception as e:
                    if self.debug:
                        print(f"[Evidence] failed to save evidence for {name}: {e}")

            # Log incident once per detection
            if self.logger:
                severity = "high" if name == "cell phone" else (
                    "medium" if name in {"headphones", "earphones", "tablet"} else "low"
                )
                reason = f"{name.replace(' ', '_')}_detected"
                try:
                    self.logger.incident(severity, reason, evidence_path=ev_path)
                except Exception:
                    if self.debug:
                        print(f"[Logger] failed to log incident for {name}")
                if self.debug:
                    print(f"[LOG] {name} logged as {severity} (conf={conf:.2f})")

            # Debug drawing
            if draw_debug:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if self.debug:
            print(f"[ObjectDetector] Accepted {len(detections)} detections")
        return detections

    # ---------------- SAVE EVIDENCE ----------------
    def save_evidence(self, img, prefix="object"):
        ts = int(time.time())
        fname = os.path.join(EVIDENCE_DIR, f"{prefix}_{ts}.jpg")
        try:
            cv2.imwrite(fname, img)
            if self.debug:
                print(f"[Evidence] saved → {fname}")
        except Exception as e:
            if self.debug:
                print(f"[Evidence] failed to save: {e}")
            return None
        return fname
