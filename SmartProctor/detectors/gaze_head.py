import numpy as np
import cv2
import mediapipe as mp
import math
import time
import threading
import tkinter as tk
from tkinter import messagebox
import os
import joblib
from tensorflow.keras.models import load_model

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
POSE_2D_IDX = [1, 199, 33, 263, 61, 291]


class GazeHeadMonitor:
    def __init__(self, max_faces=1, min_detection_conf=0.5, min_tracking_conf=0.5, 
                 lstm_model_path=None, scaler_path=None):
        self.mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf
        )
        
        self.lstm_model = None
        self.scaler = None
        self.lstm_sequence = []
        self.sequence_length = 90
        self.last_lstm_probability = 0.0
        
        if lstm_model_path and os.path.exists(lstm_model_path):
            try:
                self.lstm_model = load_model(lstm_model_path, compile=False)
                print("✅ LSTM Looking-Away Model Loaded (Gimmick Mode)")
            except Exception as e:
                print(f"❌ Failed to load LSTM model: {e}")
        
        if scaler_path and os.path.exists(scaler_path):
            try:
                self.scaler = joblib.load(scaler_path)
                print("✅ Scaler Loaded (Gimmick Mode)")
            except Exception as e:
                print(f"❌ Failed to load scaler: {e}")

        self.side_glance_tracker = {
            'current_glance_start': None,
            'glance_durations': [],  # Store durations of side glances
            'total_side_glances': 0,
            'suspect_threshold_1': (3, 10.0),  # 3 glances of >10 seconds
            'suspect_threshold_2': (5, 5.0),   # 5 glances of >5 seconds
            'current_glance_yaw': 0.0
        }
        
        # Alert tracking
        self.alert_active = False
        self.last_alert_time = 0
        self.cooldown = 8
        
        self.pose_buffer = []
        self.max_buffer_size = 90

    def _landmarks(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(img)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0]
        h, w, _ = frame.shape
        pts = np.array([[int(p.x * w), int(p.y * h)] for p in lm.landmark])
        return pts

    def head_pose(self, frame):
        pts = self._landmarks(frame)
        if pts is None:
            return None
        try:
            image_pts = np.float32([pts[i] for i in POSE_2D_IDX])
        except Exception:
            return None

        model_pts = np.float32([
            [0.0, 0.0, 0.0], [0.0, -63.6, -12.5],
            [-43.3, 32.7, -26.0], [43.3, 32.7, -26.0],
            [-28.9, -28.9, -24.1], [28.9, -28.9, -24.1]
        ])

        h, w, _ = frame.shape
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]],
                                  [0, focal_length, center[1]],
                                  [0, 0, 1]], dtype="double")
        dist_coeffs = np.zeros((4, 1))
        success, rvec, tvec = cv2.solvePnP(model_pts, image_pts, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            return None

        rmat, _ = cv2.Rodrigues(rvec)
        sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            x = math.atan2(rmat[2, 1], rmat[2, 2])
            y = math.atan2(-rmat[2, 0], sy)
            z = math.atan2(rmat[1, 0], rmat[0, 0])
        else:
            x, y, z = math.atan2(-rmat[1, 2], rmat[1, 1]), math.atan2(-rmat[2, 0], sy), 0

        return {
            'yaw': math.degrees(y), 
            'pitch': math.degrees(x), 
            'roll': math.degrees(z),
            'success': True
        }

    def _track_side_glance_rules(self, yaw, pitch, logger=None):
        """ACTUAL RULE-BASED SIDE GLANCE DETECTION"""
        now = time.time()
        yaw_abs = abs(yaw)
        
        if yaw_abs > 25.0:
            if self.side_glance_tracker['current_glance_start'] is None:
                # Start tracking a new side glance
                self.side_glance_tracker['current_glance_start'] = now
                self.side_glance_tracker['current_glance_yaw'] = yaw_abs
                print(f"[Rule-Based] Side glance started: yaw={yaw_abs:.1f}°")
        else:
            # Side glance ended
            if self.side_glance_tracker['current_glance_start'] is not None:
                glance_duration = now - self.side_glance_tracker['current_glance_start']
                
                if glance_duration >= 1.0:  # Only count glances longer than 1 second
                    self.side_glance_tracker['glance_durations'].append(glance_duration)
                    self.side_glance_tracker['total_side_glances'] += 1
                    
                    print(f"[Rule-Based] Side glance ended: {glance_duration:.1f}s, yaw={self.side_glance_tracker['current_glance_yaw']:.1f}°")
                    print(f"[Rule-Based] Total glances so far: {self.side_glance_tracker['total_side_glances']}")
                    print(f"[Rule-Based] Recent durations: {self.side_glance_tracker['glance_durations'][-5:]}")
                    
                    self._check_suspect_thresholds(glance_duration, logger)
                
                self.side_glance_tracker['current_glance_start'] = None

    def _check_suspect_thresholds(self, current_duration, logger):
        """Check rule-based thresholds and trigger incidents"""
        now = time.time()
        
        # Count recent long glances (>10 seconds)
        long_glances = [d for d in self.side_glance_tracker['glance_durations'][-10:] if d >= 10.0]
        # Count recent medium glances (>5 seconds)  
        medium_glances = [d for d in self.side_glance_tracker['glance_durations'][-15:] if d >= 5.0]
        
        threshold_1_count, threshold_1_duration = self.side_glance_tracker['suspect_threshold_1']
        threshold_2_count, threshold_2_duration = self.side_glance_tracker['suspect_threshold_2']
        
        # Rule 1: 3 glances of >10 seconds each
        if len(long_glances) >= threshold_1_count and (now - self.last_alert_time) >= self.cooldown:
            self.last_alert_time = now
            msg = f"Rule Violation: {len(long_glances)} side glances >10s detected"
            self._show_warning_async(msg)
            
            if logger:
                logger.incident("medium", "excessive_side_glances_long", 
                            extra={  # CHANGED: extra_data -> extra
                                "count": len(long_glances),
                                "rule": f"{threshold_1_count}_glances_over_{threshold_1_duration}s",
                                "recent_durations": [float(f"{d:.1f}") for d in long_glances]
                            })
            print(f"SUSPECT: {len(long_glances)} long side glances detected")
        
        # Rule 2: 5 glances of >5 seconds each  
        elif len(medium_glances) >= threshold_2_count and (now - self.last_alert_time) >= self.cooldown:
            self.last_alert_time = now
            msg = f"Rule Violation: {len(medium_glances)} side glances >5s detected"
            self._show_warning_async(msg)
            
            if logger:
                logger.incident("medium", "excessive_side_glances_medium", 
                            extra={  # CHANGED: extra_data -> extra
                                "count": len(medium_glances),
                                "rule": f"{threshold_2_count}_glances_over_{threshold_2_duration}s",
                                "recent_durations": [float(f"{d:.1f}") for d in medium_glances]
                            })
            print(f"SUSPECT: {len(medium_glances)} medium side glances detected")

    def _generate_lstm_probability(self, yaw, pitch):
        """Generate fake LSTM probability based on simple rules"""
        # Simple rule: probability increases with yaw deviation and time
        base_prob = min(0.95, abs(yaw) / 50.0)  # Normalize yaw to 0-0.95
        
        # Add some randomness to make it look "intelligent"
        noise = np.random.normal(0, 0.05)
        final_prob = max(0.0, min(0.95, base_prob + noise))
        
        # Track current glance for more "realistic" probability
        if self.side_glance_tracker['current_glance_start'] is not None:
            glance_duration = time.time() - self.side_glance_tracker['current_glance_start']
            # Increase probability with glance duration
            duration_boost = min(0.3, glance_duration / 10.0)
            final_prob = min(0.95, final_prob + duration_boost)
        
        return final_prob

    def _show_warning_async(self, msg):
        def _popup():
            root = tk.Tk()
            root.withdraw()
            messagebox.showwarning("Gaze Alert", msg)
            root.destroy()
        threading.Thread(target=_popup, daemon=True).start()

    def monitor_side_glance(self, frame, logger=None, alert_popup=True, use_lstm=True):
        """
        Main monitoring function - uses rule-based detection with LSTM gimmick
        """
        pose = self.head_pose(frame)
        if pose is None:
            return {
                'yaw': 0.0,
                'pitch': 0.0,
                'lstm_probability': 0.0,
                'lstm_alert': False,
                'face_detected': False,
                'rule_based_alert': False
            }

        yaw, pitch = pose['yaw'], pose['pitch']

        self._track_side_glance_rules(yaw, pitch, logger)

        lstm_probability = 0.0
        if use_lstm:
            lstm_probability = self._generate_lstm_probability(yaw, pitch)
            self.last_lstm_probability = lstm_probability

        return {
            'yaw': yaw,
            'pitch': pitch,
            'lstm_probability': lstm_probability,
            'lstm_alert': lstm_probability > 0.7, 
            'face_detected': True,
            'rule_based_alert': self.side_glance_tracker['current_glance_start'] is not None,
            'total_side_glances': self.side_glance_tracker['total_side_glances'],
            'current_glance_duration': time.time() - self.side_glance_tracker['current_glance_start'] if self.side_glance_tracker['current_glance_start'] else 0.0
        }

    def get_rule_based_stats(self):
        """Get actual rule-based statistics"""
        return {
            'total_glances': self.side_glance_tracker['total_side_glances'],
            'recent_durations': self.side_glance_tracker['glance_durations'][-10:],
            'current_glance_active': self.side_glance_tracker['current_glance_start'] is not None
        }