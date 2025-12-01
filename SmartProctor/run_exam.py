import os
import time
import yaml
import cv2
import sqlite3
import argparse
import json
import subprocess
import glob
import shutil
from storage.logger import ExamLogger
from detectors.face_recognition import FaceAuthenticator
from detectors.gaze_head import GazeHeadMonitor
from detectors.object_detector import ObjectDetector
from detectors.audio_monitor import AudioMonitor
from detectors.tab_switch import TabSwitchMonitor
from reporting.report_generator import build_report_and_sign
from utils.video_utils import draw_label, save_snapshot

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

def cleanup_old_evidence():
    """Remove old evidence files before starting new exam"""
    evidence_dirs = [
        "storage_files/evidence",
        "storage_files"
    ]
    
    for evidence_dir in evidence_dirs:
        full_path = os.path.join(ROOT_DIR, evidence_dir)
        if os.path.exists(full_path):
            # Remove all .wav, .jpg, .png files
            for file_pattern in ["*.wav", "*.jpg", "*.png", "*.jpeg"]:
                for file_path in glob.glob(os.path.join(full_path, file_pattern)):
                    try:
                        os.remove(file_path)
                        print(f"[Cleanup] Removed old evidence: {os.path.basename(file_path)}")
                    except Exception as e:
                        print(f"[Cleanup] Error removing {file_path}: {e}")
    
    # Also clean the database by removing old session data
    db_path = os.path.join(ROOT_DIR, "storage_files", "session.db")
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            # Clear previous exam data but keep schema
            cur.execute("DELETE FROM logs")
            cur.execute("DELETE FROM incidents") 
            cur.execute("DELETE FROM session_meta")
            conn.commit()
            conn.close()
            print("[Cleanup] Cleared previous exam data from database")
        except Exception as e:
            print(f"[Cleanup] Error clearing database: {e}")

def load_config(path=os.path.join(ROOT_DIR, 'configs', 'config.yaml')):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def pre_exam_passed(db_path):
    """Check if pre-exam verification was completed successfully"""
    if not os.path.exists(db_path):
        return False
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM logs WHERE event_type='system' AND details LIKE ?",
            ('%pre_exam_passed%',)
        )
        cnt = cur.fetchone()[0]
        conn.close()
        return cnt > 0
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Path to input image/video instead of webcam")
    parser.add_argument("--skip-pre-exam", action="store_true", help="Skip pre-exam validation")
    parser.add_argument("--no-report", action="store_true", help="Skip report generation (for testing)")
    args = parser.parse_args()

    # CLEANUP OLD EVIDENCE BEFORE STARTING
    print("[Main] Cleaning up old evidence files...")
    cleanup_old_evidence()

    cfg = load_config()
    db_path = cfg['paths']['db_path']

    # --- Always run pre-exam GUI unless explicitly skipped ---
    if not args.skip_pre_exam:
        print("[PreExam] Launching pre-exam verification...")
        pre_exam_script = os.path.join(ROOT_DIR, "pre_exam_gui.py")

        # Remove previous 'pre_exam_passed' logs to avoid auto-skip
        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                cur = conn.cursor()
                cur.execute("DELETE FROM logs WHERE event_type='system' AND details LIKE '%pre_exam_passed%'")
                conn.commit()
                conn.close()
                print("[PreExam] Cleared previous pre-exam records.")
            except Exception as e:
                print(f"[PreExam] Warning: Failed to clear old pre-exam data: {e}")

        # Run the GUI
        if not os.path.exists(pre_exam_script):
            print(f"[Error] Pre-exam script not found: {pre_exam_script}")
            return

        subprocess.run(["python", pre_exam_script], check=False)

        # Verify again after GUI closes
        if not pre_exam_passed(db_path):
            print("[PreExam] ❌ Pre-exam verification failed or cancelled. Aborting exam.")
            return

        print("[PreExam] ✅ Pre-exam verification passed. Starting exam...")

    logger = ExamLogger(db_path=db_path)

    # Camera / Input source setup
    cap = None
    single_frame = None
    if args.input:
        if args.input.lower().endswith(('.jpg', '.jpeg', '.png')):
            single_frame = cv2.imread(args.input)
            if single_frame is None:
                print(f"[Error] Could not read image: {args.input}")
                return
        else:
            cap = cv2.VideoCapture(args.input)
    else:
        cam_idx = cfg['camera']['index']
        cap = cv2.VideoCapture(cam_idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg['camera']['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg['camera']['height'])

    # Initialize modules with LSTM integration
    face_auth = FaceAuthenticator() if cfg['features'].get('face_recognition', True) else None
    
    # Enhanced Gaze Monitor with LSTM
    lstm_model_path = cfg['paths'].get('lstm_model_path')
    scaler_path = cfg['paths'].get('lstm_scaler_path')
    
    gaze = GazeHeadMonitor(
        lstm_model_path=lstm_model_path,
        scaler_path=scaler_path
    ) if cfg['features'].get('gaze_tracking', True) else None

    obj_detector = ObjectDetector(
        conf=cfg['thresholds']['object_confidence']
    ) if cfg['features'].get('object_detection', True) else None

    # In run_exam.py, update the audio initialization:
    audio = None
    if cfg['features'].get('audio_monitoring', True):
        try:
            print("[AudioMonitor] Initializing audio monitor...")
            audio = AudioMonitor(
                sample_rate=16000,
                chunk_duration=0.032,
                min_duration=cfg['thresholds'].get('min_speech_duration', 2.0),
                debounce_sec=cfg['thresholds'].get('speech_debounce_sec', 8.0),
                save_dir="storage_files/evidence",
                debug=True
            )
            if not audio.start():
                print("[AudioMonitor] ❌ Audio monitor failed to start")
                audio = None
            else:
                print("[AudioMonitor] ✅ Audio monitor started successfully")
        except Exception as e:
            print(f"[AudioMonitor] ❌ Critical error: {e}")
            audio = None

    tab_monitor = TabSwitchMonitor(logger=logger) if cfg['features'].get('tab_switch', False) else None

    print("[Main] ✅ SmartProctor Exam Monitoring Started with LSTM Integration.")
    logger.log("system", {"event": "exam_started"})

    duration_seconds = cfg['exam']['duration_minutes'] * 60
    start_time = time.time()
    frame_interval = 1.0 / cfg['camera'].get('fps', 10)

    # --- Event cooldowns ---
    last_obj_event = 0
    last_speech_event = 0
    last_lstm_alert = 0
    obj_cooldown = 10.0
    speech_cooldown = 5.0
    lstm_cooldown = 15.0  # Longer cooldown for LSTM alerts
    recent_obj_classes = []

    # NEW: Audio monitoring statistics tracking
    audio_stats_interval = 30.0  # Log audio stats every 30 seconds
    last_audio_stats_time = time.time()

    try:
        while True:
            ret, frame = (True, single_frame.copy()) if single_frame is not None else cap.read()
            if not ret:
                break

            ts = time.time()

            # --- Face Verification ---
            if face_auth and getattr(face_auth, 'enroll_vec', None) is not None:
                status, score = face_auth.verify_frame(frame, threshold=cfg['thresholds']['face_similarity_threshold'])

                if status == "verified":
                    logger.log("face_auth", {"status": "verified", "score": score, "ts": ts})
                elif status == "look_away":
                    logger.log("face_auth", {"status": "look_away", "score": score, "ts": ts})
                elif status == "no_face":
                    logger.log("face_auth", {"status": "no_face", "score": score, "ts": ts})
                elif status == "mismatch":
                    if score < 0.6:
                        fname = save_snapshot(frame, prefix="face_mismatch")
                        logger.incident("high", "face_mismatch", evidence_path=fname)
                        print("[FaceAuth] Possible identity mismatch logged.")

            # --- Enhanced Gaze & Head Pose with LSTM ---
            if gaze:
                # Get enhanced monitoring results with LSTM
                gaze_results = gaze.monitor_side_glance(
                    frame, 
                    logger=logger, 
                    alert_popup=True, 
                    use_lstm=True  # Enable LSTM detection
                )
                
                # Always log gaze information (even when no face detected)
                logger.log("head_pose", {
                    "yaw": float(gaze_results.get("yaw", 0.0)),
                    "pitch": float(gaze_results.get("pitch", 0.0)),
                    "lstm_probability": float(gaze_results.get("lstm_probability", 0.0)),
                    "lstm_alert": bool(gaze_results.get("lstm_alert", False)),
                    "face_detected": bool(gaze_results.get("face_detected", False)),
                    "ts": ts
                })
                
                # Log LSTM-based incidents with cooldown
                if (gaze_results.get("lstm_alert") and 
                    (ts - last_lstm_alert) >= lstm_cooldown):
                    
                    last_lstm_alert = ts
                    fname = save_snapshot(frame, prefix="lstm_looking_away")
                    logger.incident(
                        "medium", 
                        "ai_looking_away_detected", 
                        evidence_path=fname,
                        extra={ 
                            "probability": float(gaze_results.get("lstm_probability", 0.0)),
                            "yaw": float(gaze_results.get("yaw", 0.0)),
                            "pitch": float(gaze_results.get("pitch", 0.0))
                        }
                    )
                    print(f"[LSTM] Looking-away detected with probability: {gaze_results.get('lstm_probability', 0.0):.3f}")
            # --- Object Detection ---
            if obj_detector and (ts - last_obj_event) > obj_cooldown:
                dets = obj_detector.find(frame, save_evidence=True)
                if dets:
                    last_obj_event = ts
                    classes = [d["cls"] for d in dets]
                    recent_obj_classes.extend(classes)
                    fname = save_snapshot(frame, prefix="object")

                    unique = set(recent_obj_classes[-3:])
                    if "cell phone" in unique or len(unique) > 1:
                        logger.incident("high", f"object_detected_{'_'.join(unique)}", evidence_path=fname)
                        print(f"[Object] High-risk item detected: {unique}")
                    else:
                        logger.incident("low", f"object_warning_{'_'.join(unique)}", evidence_path=fname)

            # --- UPDATED: Audio Detection ---
            if audio and (ts - last_speech_event) > speech_cooldown:
                # NEW: Check if audio monitor is still running properly
                if not audio.is_running():
                    print("[AudioMonitor] ⚠️ Audio monitor stopped unexpectedly, attempting restart...")
                    try:
                        if audio.start():
                            print("[AudioMonitor] ✅ Audio monitor restarted successfully")
                        else:
                            print("[AudioMonitor] ❌ Failed to restart audio monitor")
                            audio = None
                    except Exception as e:
                        print(f"[AudioMonitor] ❌ Restart failed: {e}")
                        audio = None
                
                # Check for speech with the improved monitor
                elif audio.is_speaking():
                    last_speech_event = ts
                    # Get both image and audio evidence
                    fname = save_snapshot(frame, prefix="speech")
                    audio_path = audio._save_audio_evidence(2.0)  # Get audio evidence path
                    
                    # Log incident with both evidences
                    logger.incident("medium", "speech_detected", 
                                evidence_path=fname,  # Image evidence
                                extra={"audio_evidence": audio_path} if audio_path else None)
                    print("[Audio] Speech detected event logged with evidence.")
            # NEW: Periodic audio status logging
            if audio and (ts - last_audio_stats_time) >= audio_stats_interval:
                stats = audio.get_stats()
                logger.log("system", {
                    "event": "audio_monitor_status",
                    "stats": stats,
                    "ts": ts
                })
                if audio.debug:
                    print(f"[AudioMonitor] Status: {stats}")
                last_audio_stats_time = ts

            # --- Tab Switch ---
            if tab_monitor:
                state = tab_monitor.run_cycle()
                if state != "active":
                    fname = save_snapshot(frame, prefix="tabswitch")
                    logger.incident("low", "tab_switch", evidence_path=fname)

            # --- Display Preview with LSTM Info ---
            if single_frame is None:
                elapsed = int(time.time() - start_time)
                draw_label(frame, f"Exam Time: {elapsed}s", pos=(10, 25))
                
                # Display LSTM probability if available
                if gaze and hasattr(gaze, 'lstm_sequence') and len(gaze.lstm_sequence) > 0:
                    prob = getattr(gaze, 'last_lstm_probability', 0.0)
                    draw_label(frame, f"LSTM Looking-Away: {prob:.3f}", pos=(10, 55), color=(0, 255, 255))
                
                # NEW: Display audio status
                if audio:
                    audio_status = "SPEAKING" if audio.is_speaking() else "LISTENING"
                    audio_color = (0, 0, 255) if audio.is_speaking() else (0, 255, 0)
                    draw_label(frame, f"Audio: {audio_status}", pos=(10, 85), color=audio_color)
                
                cv2.imshow("SmartProctor – Live Monitor with LSTM (Press Q to Exit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[Main] User terminated monitoring early.")
                    break

            if single_frame is not None or (time.time() - start_time > duration_seconds):
                break

            time.sleep(frame_interval)

    except KeyboardInterrupt:
        print("[Main] Interrupted by user manually.")

    finally:
        # UPDATED: Proper audio cleanup
        if audio:
            print("[Main] Stopping audio monitor...")
            audio.stop()
            
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        
        logger.log("system", {"event": "exam_finished"})

        # --- Generate Report or Print Verdict ---
        if not args.no_report:
            out_pdf = cfg["paths"].get("output_report")
            private_key = cfg["paths"].get("private_key")
            build_report_and_sign(db_path=db_path, out_pdf=out_pdf, private_key=private_key)
            print("[Main] Exam Finished. Report Generated with LSTM Analysis.")
        else:
            from decision.fusion import decide_verdict
            verdict = decide_verdict(db_path)
            print(json.dumps(verdict, indent=2))

        logger.close()


if __name__ == "__main__":
    main()