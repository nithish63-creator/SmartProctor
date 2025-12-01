import os
import time
import numpy as np
import cv2
import mediapipe as mp
import torch
import threading
from tkinter import messagebox, Tk
from facenet_pytorch import InceptionResnetV1

# ---------------------------------------------------------------
# Paths setup
# ---------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
STORAGE_DIR = os.path.join(BASE_DIR, 'storage_files')
os.makedirs(STORAGE_DIR, exist_ok=True)

ENROLL_PATH = os.path.join(STORAGE_DIR, 'enroll.npy')
DESK_SNAPSHOT_PATH = os.path.join(STORAGE_DIR, 'desk_ref.jpg')
FACE_SNAPSHOT_PATH = os.path.join(STORAGE_DIR, 'face_ref.jpg')

mp_face = mp.solutions.face_detection


# ---------------------------------------------------------------
# FaceAuthenticator Class
# ---------------------------------------------------------------
class FaceAuthenticator:

    def __init__(self, min_frames=50, device='cpu'):
        self.detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        self.device = torch.device(device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.min_frames = min_frames
        self.enroll_vec = None
        if os.path.exists(ENROLL_PATH):
            try:
                self.enroll_vec = np.load(ENROLL_PATH)
            except Exception:
                self.enroll_vec = None

        # Face loss timer for alerting during exam
        self.last_face_seen = time.time()
        self.alert_active = False

    # ---------------------------------------------------------------
    # Popup helper (non-blocking)
    # ---------------------------------------------------------------
    def _popup_async(self, msg):
        def _popup():
            root = Tk()
            root.withdraw()
            messagebox.showwarning("Face Visibility Alert", msg)
            root.destroy()
        threading.Thread(target=_popup, daemon=True).start()

    # ---------------------------------------------------------------
    # Face embedding with relaxed frontal filter
    # ---------------------------------------------------------------
    def _frame_to_embedding(self, frame, require_frontal=True):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.detector.process(img_rgb)
        if not res.detections:
            return None

        # Choose the highest confidence detection
        det = max(res.detections, key=lambda d: d.score[0] if d.score else 0)
        score = det.score[0] if det.score else 0.0
        if score < 0.6:
            return None

        bbox_rel = det.location_data.relative_bounding_box
        h, w, _ = frame.shape
        x = int(bbox_rel.xmin * w)
        y = int(bbox_rel.ymin * h)
        bw = int(bbox_rel.width * w)
        bh = int(bbox_rel.height * h)

        # Skip invalid/partial faces
        if bw <= 0 or bh <= 0 or x < 0 or y < 0 or (x + bw) > w or (y + bh) > h:
            return None

        pad = int(0.25 * max(bw, bh))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)

        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            return None

        # Slightly relaxed frontal ratio
        if require_frontal:
            ratio = bw / float(bh + 1e-5)
            if ratio < 0.6 or ratio > 1.5:
                return None

        # Prepare embedding
        face_rgb = cv2.resize(face, (160, 160))
        face_rgb = cv2.cvtColor(face_rgb, cv2.COLOR_BGR2RGB)
        face_norm = (face_rgb / 255.0).astype('float32')
        tensor = torch.tensor(np.transpose(face_norm, (2, 0, 1)), dtype=torch.float32).unsqueeze(0).to(self.device)
        tensor = (tensor - 0.5) / 0.5
        with torch.no_grad():
            emb = self.resnet(tensor)
        emb = emb.cpu().numpy()[0]
        norm = np.linalg.norm(emb)
        if norm == 0:
            return None
        emb = emb / norm
        return emb, face

    # ---------------------------------------------------------------
    # Enrollment
    # ---------------------------------------------------------------
    def enroll_student(self, camera_index=0, frames=None, show_preview=True, save_desk_snapshot=True):
        frames_to_capture = frames if frames is not None else self.min_frames
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("[FaceAuthenticator] ❌ Cannot open camera.")
            return False

        collected = 0
        embeddings = []
        last_face_crop = None
        print("[Enrollment] Keep your face clearly visible. Avoid covering it.")
        start_t = time.time()

        try:
            while collected < frames_to_capture and time.time() - start_t < 120:
                ret, frame = cap.read()
                if not ret:
                    continue
                res = self._frame_to_embedding(frame)
                if res is not None:
                    emb, face_crop = res
                    embeddings.append(emb)
                    last_face_crop = face_crop.copy()
                    collected += 1
                    print(f"[Enrollment] Collected {collected}/{frames_to_capture}")
                else:
                    cv2.putText(frame, "⚠️ Face not visible — please adjust position", (30, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if show_preview:
                    cv2.imshow("Enrollment (Press Q to quit)", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("[Enrollment] Aborted by user.")
                        break
            cv2.destroyAllWindows()
        finally:
            cap.release()

        if not embeddings:
            print("[Enrollment] ❌ No valid faces collected.")
            return False

        mean_emb = np.mean(np.stack(embeddings), axis=0)
        mean_emb = mean_emb / np.linalg.norm(mean_emb)
        np.save(ENROLL_PATH, mean_emb)
        self.enroll_vec = mean_emb
        print(f"[Enrollment] ✅ Saved at {ENROLL_PATH}")

        if last_face_crop is not None:
            cv2.imwrite(FACE_SNAPSHOT_PATH, last_face_crop)

        if save_desk_snapshot:
            cap = cv2.VideoCapture(camera_index)
            time.sleep(0.2)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(DESK_SNAPSHOT_PATH, frame)
            cap.release()

        return True

    # ---------------------------------------------------------------
    # Verification (during exam)
    # ---------------------------------------------------------------
    def verify_frame(self, frame, threshold=0.78):
        """
        Returns (status, score)
        status ∈ {"verified", "mismatch", "look_away", "no_face"}
        """
        if self.enroll_vec is None and os.path.exists(ENROLL_PATH):
            try:
                self.enroll_vec = np.load(ENROLL_PATH)
            except Exception:
                self.enroll_vec = None

        if self.enroll_vec is None:
            return "no_face", 0.0

        res = self._frame_to_embedding(frame, require_frontal=False)
        if res is None:
            # If face missing for >3s, warn once
            if time.time() - self.last_face_seen > 3 and not self.alert_active:
                self.alert_active = True
                self._popup_async("Your face is not clearly visible. Please adjust your position.")
            return "look_away", 0.0

        emb, _ = res
        self.last_face_seen = time.time()
        self.alert_active = False

        score = float(np.dot(self.enroll_vec, emb))
        if score >= threshold:
            return "verified", score
        elif score >= threshold * 0.85:
            # Slight mismatch, likely due to angle — treat as look_away
            return "look_away", score
        else:
            return "mismatch", score

    # ---------------------------------------------------------------
    # Pre-exam verification
    # ---------------------------------------------------------------
    def pre_exam_verify(self, camera_index=0, threshold=0.78):
        cap = cv2.VideoCapture(camera_index)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("[PreExam] Camera error.")
            return "error", 0.0

        status, score = self.verify_frame(frame, threshold=threshold)
        print(f"[PreExam] Status={status}, Score={score:.2f}")
        return status, score

    # ---------------------------------------------------------------
    # Compatibility function for pre_exam_gui.py
    # ---------------------------------------------------------------
    def save_embedding(self, emb, path=None):
        path = path or ENROLL_PATH
        if emb is None or not isinstance(emb, np.ndarray):
            print("[FaceAuthenticator] Invalid embedding — not saved.")
            return False
        norm = np.linalg.norm(emb)
        if norm == 0:
            print("[FaceAuthenticator] Zero-norm embedding — not saved.")
            return False

        emb = emb / norm
        np.save(path, emb)
        self.enroll_vec = emb
        print(f"[FaceAuthenticator] Embedding saved → {path}")
        return True
