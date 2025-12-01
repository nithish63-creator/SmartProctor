import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import yaml
import sqlite3
import time

from detectors.face_recognition import FaceAuthenticator, ENROLL_PATH, FACE_SNAPSHOT_PATH, DESK_SNAPSHOT_PATH
from detectors.object_detector import ObjectDetector
from storage.logger import ExamLogger
from utils.video_utils import save_snapshot

ROOT = os.path.abspath(os.path.dirname(__file__))
CFG_PATH = os.path.join(ROOT, "configs", "config.yaml")
if os.path.exists(CFG_PATH):
    with open(CFG_PATH, 'r') as f:
        cfg = yaml.safe_load(f)
else:
    cfg = {}

STORAGE_DIR = os.path.join(ROOT, "storage_files")
os.makedirs(STORAGE_DIR, exist_ok=True)

FACE_PATH = ENROLL_PATH
DESK_PATH = DESK_SNAPSHOT_PATH

face_auth = FaceAuthenticator(min_frames=cfg.get('pre_exam', {}).get('enrollment_frames', 50))
obj_conf = cfg.get('thresholds', {}).get('object_confidence', 0.5)
obj_detector = ObjectDetector(conf=obj_conf)   # single global instance

DB_PATH = cfg.get('paths', {}).get('db_path',
                                   os.path.join(STORAGE_DIR, 'session.db'))


class PreExamGUI:
    def __init__(self, root, cam_index=cfg.get('camera', {}).get('index', 0)):
        self.root = root
        self.cap = cv2.VideoCapture(cam_index)
        self.face_auth = face_auth
        self.obj_detector = obj_detector
        self.logger = None
        self.target_samples = cfg.get('pre_exam', {}).get('enrollment_frames', 50)

        root.title("SmartProctor - Pre-Exam Setup")
        root.geometry("1200x700")  # Increased width for side panel
        root.configure(bg='#f0f0f0')
        
        # Main container
        main_container = tk.Frame(root, bg='#f0f0f0')
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel - Camera and controls
        left_panel = tk.Frame(main_container, bg='#f0f0f0')
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Title
        title_label = tk.Label(left_panel, text="SmartProctor — Pre-Exam Setup",
                             font=("Arial", 16, "bold"), bg='#f0f0f0')
        title_label.pack(pady=8)

        # Video preview
        video_frame = tk.Frame(left_panel, relief=tk.SUNKEN, bd=2)
        video_frame.pack(pady=10)
        self.video_label = tk.Label(video_frame)
        self.video_label.pack(padx=2, pady=2)

        # Control buttons
        button_frame = tk.Frame(left_panel, bg='#f0f0f0')
        button_frame.pack(pady=10)

        tk.Button(button_frame, text="📷 Capture Face (50 samples)", 
                 command=self.capture_face, width=25, height=2,
                 bg='#4CAF50', fg='white', font=("Arial", 10, "bold")).grid(row=0, column=0, padx=8, pady=5)
        
        tk.Button(button_frame, text="🖼️ Capture Desk Snapshot", 
                 command=self.capture_desk, width=25, height=2,
                 bg='#2196F3', fg='white', font=("Arial", 10, "bold")).grid(row=0, column=1, padx=8, pady=5)
        
        tk.Button(button_frame, text="✅ Verify Identity", 
                 command=self.verify_identity, width=25, height=2,
                 bg='#FF9800', fg='white', font=("Arial", 10, "bold")).grid(row=0, column=2, padx=8, pady=5)

        # Finish button
        tk.Button(left_panel, text="🎯 Finish Setup & Start Exam", 
                 command=self.finish_setup, width=40, height=2,
                 bg='#2196F3', fg='white', font=("Arial", 12, "bold")).pack(pady=15)

        # Right panel - Instructions
        right_panel = tk.Frame(main_container, bg='#ffffff', relief=tk.RAISED, bd=1)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))

        # Instructions title
        instructions_title = tk.Label(right_panel, text="Setup Instructions", 
                                    font=("Arial", 14, "bold"), bg='#2196F3', fg='white')
        instructions_title.pack(fill=tk.X, padx=5, pady=10)

        # Create notebook for different instruction sections
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Face Capture Instructions Tab
        face_tab = tk.Frame(self.notebook, bg='#ffffff')
        self.notebook.add(face_tab, text="Face Capture")

        # Desk Capture Instructions Tab
        desk_tab = tk.Frame(self.notebook, bg='#ffffff')
        self.notebook.add(desk_tab, text="Desk Setup")

        # Fill Face Capture Instructions
        self._create_face_instructions(face_tab)
        
        # Fill Desk Capture Instructions
        self._create_desk_instructions(desk_tab)

        # Start with Face Capture tab selected
        self.notebook.select(0)

        self.prepare_session_db()
        self.update_preview()
        root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _create_face_instructions(self, parent):
        instructions = [
            "🔹 Step 1: Position yourself in good lighting",
            "   • Avoid backlighting or shadows on your face",
            "   • Natural daylight is ideal",
            "",
            "🔹 Step 2: Center your face in the frame",
            "   • Make sure your entire face is visible",
            "   • Maintain a neutral expression",
            "   • Remove hats, sunglasses, or face coverings",
            "",
            "🔹 Step 3: Stay still during capture",
            "   • The system will capture 50 samples",
            "   • This takes about 30-60 seconds",
            "   • Keep looking directly at the camera",
            "",
            "🔹 Step 4: Verification",
            "   • System will verify your identity",
            "   • Make sure verification score is above 0.8",
            "",
            "💡 Tips:",
            "• Sit approximately 2-3 feet from camera",
            "• Ensure stable internet connection",
            "• Complete in a quiet, private space"
        ]

        instruction_text = "\n".join(instructions)
        instruction_label = tk.Label(parent, text=instruction_text, 
                                   justify=tk.LEFT, bg='#ffffff', font=("Arial", 10))
        instruction_label.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Add visual example for face positioning
        example_frame = tk.Frame(parent, bg='#ffffff')
        example_frame.pack(fill=tk.X, padx=15, pady=10)

        # Create a simple face positioning diagram using canvas
        canvas = tk.Canvas(example_frame, width=200, height=150, bg='#ffffff', highlightthickness=0)
        canvas.pack()

        # Draw face positioning guide
        canvas.create_oval(50, 30, 150, 130, outline='#2196F3', width=2)  # Face circle
        canvas.create_oval(70, 60, 90, 80, outline='#2196F3', width=2)    # Left eye
        canvas.create_oval(110, 60, 130, 80, outline='#2196F3', width=2)  # Right eye
        canvas.create_line(100, 90, 100, 110, fill='#2196F3', width=2)    # Nose
        canvas.create_arc(80, 100, 120, 120, start=0, extent=180, outline='#2196F3', width=2)  # Mouth
        
        canvas.create_text(100, 10, text="Position face like this", font=("Arial", 8, "bold"))

    def _create_desk_instructions(self, parent):
        instructions = [
            "🔹 Step 1: Tilt Your Laptop",
            "   • Adjust screen to approximately 30° angle",
            "   • This helps capture your entire desk area",
            "   • Use books or stand if needed for proper angle",
            "",
            "🔹 Step 2: Clear Desk Surface",
            "   • Remove all unauthorized items:",
            "     📱 Phones, tablets, smartwatches",
            "     📚 Books, notes, paper",
            "     🎧 Headphones, earbuds",
            "     🖱️ Extra mice, keyboards",
            "",
            "🔹 Step 3: Position Camera",
            "   • Ensure good lighting on desk area",
            "   • Camera should see entire workspace",
            "   • Remove any reflective surfaces",
            "",
            "🔹 Step 4: Capture Clear Image",
            "   • Click 'Capture Desk Snapshot'",
            "   • System will scan for prohibited items",
            "   • Wait for clearance confirmation",
            "",
            "🚫 Prohibited Items:",
            "• Electronic devices (except exam computer)",
            "• Books, notes, or writing materials",
            "• Communication devices",
            "• Food and drinks (unless permitted)"
        ]

        instruction_text = "\n".join(instructions)
        instruction_label = tk.Label(parent, text=instruction_text, 
                                   justify=tk.LEFT, bg='#ffffff', font=("Arial", 10))
        instruction_label.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Create desk setup visualization
        visualization_frame = tk.Frame(parent, bg='#ffffff')
        visualization_frame.pack(fill=tk.X, padx=15, pady=10)

        # Create canvas for desk setup diagram
        canvas = tk.Canvas(visualization_frame, width=250, height=180, bg='#ffffff', highlightthickness=0)
        canvas.pack()

        # Draw desk setup with 30-degree angle
        # Desk surface
        canvas.create_rectangle(30, 120, 220, 140, fill='#8B4513', outline='#654321')  # Desk
        
        # Laptop with 30-degree angle
        # Laptop base
        canvas.create_rectangle(80, 100, 170, 120, fill='#2C3E50', outline='#1a2530')
        # Laptop screen at 30-degree angle
        points = [80, 100, 75, 70, 145, 70, 170, 100]
        canvas.create_polygon(points, fill='#34495E', outline='#1a2530')
        
        # Camera view cone
        canvas.create_arc(40, 40, 210, 160, start=200, extent=140, 
                         outline='#E74C3C', width=2, style=tk.ARC)
        
        # Angle indicator
        canvas.create_line(125, 100, 125, 70, arrow=tk.LAST, fill='#27AE60', width=2)
        canvas.create_line(125, 100, 145, 100, arrow=tk.LAST, fill='#27AE60', width=2)
        canvas.create_text(135, 85, text="30°", font=("Arial", 8, "bold"), fill='#27AE60')
        
        canvas.create_text(125, 10, text="Ideal Desk Setup with 30° Tilt", 
                          font=("Arial", 9, "bold"))

        # Status indicators
        status_frame = tk.Frame(parent, bg='#ffffff')
        status_frame.pack(fill=tk.X, padx=15, pady=10)

        self.face_status = tk.Label(status_frame, text="❌ Face Not Captured", 
                                  fg='#e74c3c', bg='#ffffff', font=("Arial", 10, "bold"))
        self.face_status.pack(anchor=tk.W)

        self.desk_status = tk.Label(status_frame, text="❌ Desk Not Scanned", 
                                  fg='#e74c3c', bg='#ffffff', font=("Arial", 10, "bold"))
        self.desk_status.pack(anchor=tk.W)

    def update_status_indicators(self):
        """Update the status indicators based on current progress"""
        face_exists = os.path.exists(FACE_PATH)
        desk_exists = os.path.exists(DESK_PATH)
        
        if face_exists:
            self.face_status.config(text="✅ Face Successfully Captured", fg='#27ae60')
        else:
            self.face_status.config(text="❌ Face Not Captured", fg='#e74c3c')
            
        if desk_exists:
            self.desk_status.config(text="✅ Desk Successfully Scanned", fg='#27ae60')
        else:
            self.desk_status.config(text="❌ Desk Not Scanned", fg='#e74c3c')

    def prepare_session_db(self):
        if os.path.exists(DB_PATH):
            ans = messagebox.askyesno("Existing session DB",
                                      "Old exam data found. Start a new session?")
            if ans:
                try:
                    os.remove(DB_PATH)
                except:
                    pass
        self.logger = ExamLogger(db_path=DB_PATH)
        self.logger.log("system", {"event": "pre_exam_gui_started"})

    def update_preview(self):
        ret, frame = self.cap.read()
        if ret:
            # Add instructional overlay to video preview
            overlay = frame.copy()
            h, w = frame.shape[:2]
            
            # Add face detection bounding box guide
            cv2.rectangle(overlay, (w//4, h//4), (3*w//4, 3*h//4), (0, 255, 0), 2)
            cv2.putText(overlay, "Position face within green box", (w//4, h//4-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Blend overlay
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img.resize((640, 360)))
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            
            # Update status indicators
            self.update_status_indicators()
            
        self.root.after(30, self.update_preview)

    def capture_face(self):
        """Auto-collect strict face embeddings"""
        # Switch to face capture instructions tab
        self.notebook.select(0)  # 0 is the face capture tab

        embeddings = []
        last_face_crop = None
        start = time.time()
        warnings_count = 0

        # Create progress window
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Face Capture Progress")
        progress_window.geometry("300x100")
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        progress_label = tk.Label(progress_window, text="Capturing face samples...", font=("Arial", 10))
        progress_label.pack(pady=10)
        
        progress = ttk.Progressbar(progress_window, orient=tk.HORIZONTAL, 
                                 length=250, mode='determinate', maximum=self.target_samples)
        progress.pack(pady=10)

        while len(embeddings) < self.target_samples and time.time() - start < 90:
            ret, frame = self.cap.read()
            if not ret:
                continue

            res = self.face_auth._frame_to_embedding(frame)
            if res is not None:
                emb, face_crop = res
                embeddings.append(emb)
                last_face_crop = face_crop
                progress['value'] = len(embeddings)
                progress_label.config(text=f"Captured {len(embeddings)}/{self.target_samples} samples")
            else:
                warnings_count += 1
                if warnings_count % 20 == 0:
                    progress_label.config(text="Face not detected - please look at camera")
            
            progress_window.update()

        progress_window.destroy()

        if not embeddings:
            messagebox.showerror("Face Capture Failed", 
                               "No valid face samples collected.\n\n"
                               "Please check:\n"
                               "• Camera is working properly\n"
                               "• Face is clearly visible\n"
                               "• Good lighting conditions\n"
                               "• No obstructions (glasses, hair, etc.)")
            return

        if len(embeddings) < self.target_samples * 0.6:
            messagebox.showerror(
                "Insufficient Samples",
                f"Only {len(embeddings)} valid samples collected.\n"
                f"Need at least {int(self.target_samples * 0.6)} samples.\n\n"
                "Please try again facing the camera directly."
            )
            return

        avg_emb = np.mean(np.vstack(embeddings), axis=0)
        self.face_auth.save_embedding(avg_emb, path=FACE_PATH)
        if last_face_crop is not None:
            cv2.imwrite(FACE_SNAPSHOT_PATH, last_face_crop)
        self.logger.log("system", {"event": "identity_enrolled", "path": FACE_PATH})
        
        # Automatically switch to desk capture instructions after successful face capture
        self.notebook.select(1)  # 1 is the desk setup tab
        
        messagebox.showinfo("Face Capture Complete", 
                          f"✅ Enrollment successful!\n\n"
                          f"• Collected {len(embeddings)} samples\n"
                          f"• Face embedding saved\n"
                          f"• Ready for verification\n\n"
                          f"Now proceed with desk capture.")

    def capture_desk(self):
        """Capture desk snapshot and run object detection"""
        # Switch to desk capture instructions tab
        self.notebook.select(1)  # 1 is the desk setup tab

        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Camera Error", "Failed to capture desk snapshot")
            return

        cv2.imwrite(DESK_PATH, frame)

        # Show scanning progress
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Scanning Desk")
        progress_window.geometry("300x80")
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        scan_label = tk.Label(progress_window, text="Scanning for prohibited items...", font=("Arial", 10))
        scan_label.pack(pady=20)
        progress_window.update()

        detections = self.obj_detector.find(frame, save_evidence=True, draw_debug=True)

        progress_window.destroy()

        if detections:
            annotated_path = save_snapshot(frame, prefix="desk_scan")
            detected_items = [d['cls'] for d in detections]
            messagebox.showerror(
                "Unauthorized Items Detected",
                f"❌ The following prohibited items were detected:\n\n"
                f"{chr(10).join(['• ' + item for item in detected_items])}\n\n"
                f"Please remove these items and scan again."
            )
        else:
            self.logger.log("system", {"event": "desk_clean"})
            messagebox.showinfo("Desk Scan Complete", 
                              "✅ Desk scan successful!\n\n"
                              "• No prohibited items detected\n"
                              "• Desk environment approved\n"
                              "• You may proceed with exam setup")

    def verify_identity(self):
        if not os.path.exists(FACE_PATH):
            messagebox.showerror("Verification Error", 
                               "No face enrollment found.\n\n"
                               "Please complete 'Capture Face' first.")
            return

        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Camera Error", "Failed to capture frame for verification")
            return

        status, score = self.face_auth.verify_frame(frame, threshold=0.8)
        snap = save_snapshot(frame, prefix="identity_verify")

        if status == "verified":
            self.logger.log("system", {"event": "identity_verified", "score": float(score)})
            messagebox.showinfo("Identity Verified", 
                              f"✅ Identity verification successful!\n\n"
                              f"Verification score: {score:.2f}\n"
                              f"Status: Approved")

        elif status == "mismatch":
            self.logger.incident("high", "identity_mismatch", evidence_path=snap)
            messagebox.showerror("Identity Mismatch", 
                               f"❌ Identity verification failed!\n\n"
                               f"Verification score: {score:.2f}\n"
                               f"Status: Not matching enrolled face\n\n"
                               f"Please re-enroll your face or try again.")

        elif status == "look_away":
            self.logger.log("system", {"event": "look_away_detected", "score": float(score)})
            messagebox.showwarning("Face Not Detected", 
                                 "Face not detected or not frontal.\n\n"
                                 "Please:\n"
                                 "• Look directly at the camera\n"
                                 "• Ensure face is clearly visible\n"
                                 "• Try verification again")

        else:
            self.logger.incident("high", "identity_error", evidence_path=snap)
            messagebox.showerror("Verification Error", 
                               "Identity verification failed due to unexpected error.\n\n"
                               "Please try again or contact support.")

    def finish_setup(self):
        # Check prerequisites
        if not os.path.exists(FACE_PATH):
            messagebox.showerror("Setup Incomplete", 
                               "Face enrollment not completed.\n\n"
                               "Please complete 'Capture Face' before finishing setup.")
            return

        if not os.path.exists(DESK_PATH):
            messagebox.showerror("Setup Incomplete", 
                               "Desk scan not completed.\n\n"
                               "Please complete 'Capture Desk Snapshot' before finishing setup.")
            return

        idv_count, desk_inc = 0, 0
        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM logs WHERE details LIKE '%identity_verified%'")
            idv_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM incidents WHERE reason LIKE '%desk%'")
            desk_inc = cur.fetchone()[0]
            conn.close()
        except Exception:
            pass

        if idv_count > 0 and desk_inc == 0:
            self.logger.log("system", {"event": "pre_exam_passed"})
            messagebox.showinfo("Pre-Exam Complete", 
                              "🎉 All pre-exam checks passed successfully!\n\n"
                              "✅ Identity verified\n"
                              "✅ Desk environment cleared\n"
                              "✅ System ready\n\n"
                              "You may now start your exam.")
            self.on_close()
        else:
            issues = []
            if idv_count == 0:
                issues.append("• Identity not verified")
            if desk_inc > 0:
                issues.append("• Unauthorized items on desk")
            
            messagebox.showerror("Setup Incomplete", 
                               "Cannot start exam yet:\n\n" + "\n".join(issues) + 
                               "\n\nPlease complete all requirements.")

    def on_close(self):
        try:
            self.cap.release()
        except:
            pass
        cv2.destroyAllWindows()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = PreExamGUI(root)
    root.mainloop()