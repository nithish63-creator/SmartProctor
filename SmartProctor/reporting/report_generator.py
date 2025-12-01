import os
import sqlite3
import glob
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from datetime import datetime
from utils.crypto_utils import sign_bytes
from decision.fusion import decide_verdict
import json


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
STORAGE_DIR = os.path.join(BASE_DIR, 'storage_files')
DB_PATH = os.path.join(STORAGE_DIR, 'session.db')
DEFAULT_PRIVATE_KEY = os.path.join(STORAGE_DIR, 'private_key.pem')
OUTPUT_PDF = os.path.join(STORAGE_DIR, 'report.pdf')

FACE_IMG = os.path.join(STORAGE_DIR, 'face_ref.jpg')
DESK_IMG = os.path.join(STORAGE_DIR, 'desk_ref.jpg')


def _read_pre_exam_results(db_path):
    results = {"identity_verified": False, "desk_clean": True}
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT details FROM logs WHERE event_type='system' AND details LIKE '%pre_exam_passed%'")
        if cur.fetchone():
            results["identity_verified"] = True
        cur.execute("SELECT COUNT(*) FROM incidents WHERE reason LIKE '%desk%' OR reason LIKE '%object%'")
        row = cur.fetchone()
        if row and row[0] > 0:
            results["desk_clean"] = False
        conn.close()
    except Exception:
        pass
    return results

def _get_exam_time_range(db_path):
    """Get the start and end time of the current exam session"""
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        
        # Get exam start time
        cur.execute("SELECT ts FROM logs WHERE event_type='system' AND details LIKE '%exam_started%' ORDER BY ts DESC LIMIT 1")
        start_row = cur.fetchone()
        
        # Get exam end time  
        cur.execute("SELECT ts FROM logs WHERE event_type='system' AND details LIKE '%exam_finished%' ORDER BY ts DESC LIMIT 1")
        end_row = cur.fetchone()
        
        conn.close()
        
        start_time = start_row[0] if start_row else None
        end_time = end_row[0] if end_row else None
        
        return start_time, end_time
    except Exception as e:
        print(f"[Report] Error getting exam time range: {e}")
        return None, None

def _get_audio_evidence_files():
    """Get all audio evidence files from storage_files/evidence"""
    audio_files = []
    evidence_dir = os.path.join(STORAGE_DIR, 'evidence')
    
    if os.path.exists(evidence_dir):
        for file in glob.glob(os.path.join(evidence_dir, "*.wav")):
            audio_files.append(file)
    
    return sorted(audio_files)

def _get_image_evidence_by_type(db_path):
    """Get image evidence files categorized by type"""
    evidence_by_type = {
        'cell_phone': [],
        'object': [],
        'face_mismatch': [],
        'speech': [],
        'side_glance': []
    }
    
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT reason, evidence_path FROM incidents WHERE evidence_path IS NOT NULL")
        
        for reason, evidence_path in cur.fetchall():
            if evidence_path and os.path.exists(evidence_path):
                reason_lower = reason.lower()
                
                if 'cell_phone' in reason_lower or 'phone' in reason_lower:
                    evidence_by_type['cell_phone'].append(evidence_path)
                elif 'object' in reason_lower:
                    evidence_by_type['object'].append(evidence_path)
                elif 'face_mismatch' in reason_lower:
                    evidence_by_type['face_mismatch'].append(evidence_path)
                elif 'speech' in reason_lower:
                    evidence_by_type['speech'].append(evidence_path)
                elif 'side_glance' in reason_lower or 'looking_away' in reason_lower:
                    evidence_by_type['side_glance'].append(evidence_path)
        
        conn.close()
    except Exception as e:
        print(f"[Report] Error getting image evidence: {e}")
    
    return evidence_by_type

def build_report_and_sign(db_path=None, out_pdf=None, private_key=None):
    db_path = db_path or DB_PATH
    out_pdf = out_pdf or OUTPUT_PDF
    private_key = private_key or DEFAULT_PRIVATE_KEY

    verdict = decide_verdict(db_path)
    pre_exam = _read_pre_exam_results(db_path)
    
    # Get exam time range for filtering evidence
    exam_start, exam_end = _get_exam_time_range(db_path)
    print(f"[Report] Exam time range: {exam_start} to {exam_end}")

    # Get evidence files
    audio_evidence = _get_audio_evidence_files()
    image_evidence = _get_image_evidence_by_type(db_path)

    c = canvas.Canvas(out_pdf, pagesize=letter)
    w, h = letter
    y = h - 60

    # ---------- HEADER ----------
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, y, "SmartProctor - Final Exam Integrity Report")
    c.setFont("Helvetica", 10)
    c.drawString(40, y - 18, f"Generated: {datetime.utcnow().isoformat()} UTC")
    y -= 40

    # ---------- PRE-EXAM RESULTS ----------
    c.setFont("Helvetica-Bold", 13)
    c.drawString(40, y, "Pre-Exam Verification")
    y -= 20
    c.setFont("Helvetica", 10)
    c.drawString(60, y, f"Identity Verification: {'PASSED' if pre_exam['identity_verified'] else 'FAILED'}")
    y -= 14
    c.drawString(60, y, f"Desk Cleanliness: {'YES' if pre_exam['desk_clean'] else 'NO (Objects Detected)'}")
    y -= 20

    # ---------- REFERENCE IMAGES ----------
    img_w, img_h = 220, 150
    if os.path.exists(FACE_IMG):
        c.setFont("Helvetica-Bold", 10)
        c.drawString(50, y, "Reference Face:")
        c.drawImage(ImageReader(FACE_IMG), 50, y - img_h - 6, width=img_w, height=img_h)
    if os.path.exists(DESK_IMG):
        c.setFont("Helvetica-Bold", 10)
        c.drawString(320, y, "Desk Snapshot:")
        c.drawImage(ImageReader(DESK_IMG), 320, y - img_h - 6, width=img_w, height=img_h)
    y -= img_h + 30

    # ---------- FINAL VERDICT ----------
    c.setFont("Helvetica-Bold", 14)
    verdict_text = f"Final Verdict: {verdict['verdict']} (Score: {verdict['score']:.2f})"
    c.drawString(40, y, verdict_text)
    y -= 20

    c.setFont("Helvetica", 10)
    c.drawString(60, y, f"Reason: {verdict.get('reason', 'N/A')}")
    y -= 14

    # REMOVED: LSTM probability display
    y -= 10

    # ---------- CHEATING INDICATORS SCORES ----------
    c.setFont("Helvetica-Bold", 13)
    c.drawString(40, y, "Cheating Indicators")
    y -= 20
    c.setFont("Helvetica", 10)
    
    # Get incident counts for scoring
    incident_counts = verdict.get('incident_counts', {})
    
    # Define indicators and their descriptions
    indicators = [
        ("Face Mismatch", incident_counts.get('face_mismatch', 0), 
         "Face verification failures during exam"),
        ("Object Detection", incident_counts.get('object_warning', 0), 
         "Suspicious objects detected on desk"),
        ("Cell Phone Detection", incident_counts.get('cell_phone', 0), 
         "Mobile phone usage detected"),
        ("Speech Recognition", incident_counts.get('speech', 0), 
         "Unauthorized speech detected"),
        ("Side Look Cheating", incident_counts.get('side_glance', 0) + incident_counts.get('ai_looking_away', 0), 
         "Suspicious gaze patterns detected")
    ]
    
    for indicator_name, count, description in indicators:
        score = min(1.0, count / 3.0)  # Normalize to 0-1 scale
        c.drawString(60, y, f"{indicator_name}: {count} incidents (score: {score:.2f})")
        y -= 14
        c.setFont("Helvetica-Oblique", 8)
        c.drawString(80, y, description)
        y -= 12
        c.setFont("Helvetica", 10)
    
    y -= 10

    # ---------- IMAGE EVIDENCE WITH DESCRIPTIONS ----------
    evidence_sections = [
        ("Cell Phone Detection", image_evidence['cell_phone'], 
         "Evidence of mobile phone usage during exam"),
        ("Object Detection", image_evidence['object'], 
         "Suspicious objects detected near candidate"),
        ("Face Mismatch", image_evidence['face_mismatch'], 
         "Face verification failures indicating potential impersonation"),
        ("Side Glance Cheating", image_evidence['side_glance'], 
         "Evidence of suspicious gaze patterns and looking away from screen"),
    ]

    for label, examples, desc in evidence_sections:
        if not examples:
            continue

        # Filter examples to only include files that exist
        valid_examples = [ex for ex in examples if os.path.exists(ex)]
        if not valid_examples:
            continue

        # Add section header
        if y < 160:
            c.showPage()
            y = h - 80
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, y, f"{label} Evidence ({len(valid_examples)} found)")
        y -= 16
        c.setFont("Helvetica-Oblique", 9)
        c.drawString(60, y, desc)
        y -= 18

        # Draw images (up to 3 per page section)
        for ex in valid_examples[:3]:
            if y < 150:
                c.showPage()
                y = h - 80
            try:
                c.drawImage(ImageReader(ex), 60, y - 120, width=180, height=120)
                c.setFont("Helvetica", 8)
                c.drawString(60, y - 130, f"Evidence: {os.path.basename(ex)}")
                y -= 140
            except Exception as e:
                print(f"[Report] Error drawing image {ex}: {e}")
                continue

        y -= 10

    # ---------- AUDIO EVIDENCE ----------
    if audio_evidence:
        if y < 100:
            c.showPage()
            y = h - 80
            
        c.setFont("Helvetica-Bold", 13)
        c.drawString(40, y, "Audio Evidence - Speech Detected")
        y -= 20
        c.setFont("Helvetica", 10)
        c.drawString(60, y, f"Total audio recordings: {len(audio_evidence)}")
        y -= 16
        
        for audio_file in audio_evidence[:5]:  # Show first 5 audio files
            if y < 80:
                c.showPage()
                y = h - 80
                
            filename = os.path.basename(audio_file)
            file_size = os.path.getsize(audio_file) / 1024  # Size in KB
            c.drawString(80, y, f"• {filename} ({file_size:.1f} KB)")
            y -= 14
            
        if len(audio_evidence) > 5:
            c.drawString(80, y, f"... and {len(audio_evidence) - 5} more files")
            y -= 14
            
        y -= 10

    # ---------- TECHNICAL DETAILS ----------
    if y < 100:
        c.showPage()
        y = h - 80
    c.setFont("Helvetica-Bold", 13)
    c.drawString(40, y, "Technical Summary")
    y -= 20
    c.setFont("Helvetica", 9)
    
    # Show total incidents by category
    c.drawString(60, y, "Incident Summary:")
    y -= 14
    
    for incident_type, count in incident_counts.items():
        if count > 0 and incident_type != 'total_incidents':
            display_name = incident_type.replace('_', ' ').title()
            c.drawString(80, y, f"{display_name}: {count}")
            y -= 12

    # ---------- INTERPRETATION ----------
    if y < 100:
        c.showPage()
        y = h - 80
    c.setFont("Helvetica-Bold", 13)
    c.drawString(40, y, "Interpretation & Notes")
    y -= 20
    c.setFont("Helvetica", 10)

    interpretation = {
        "Clean": "No significant cheating indicators detected. Candidate maintained normal behavior throughout the exam.",
        "Suspected": "Potentially suspicious behavior observed. Manual review of evidence recommended.",
        "Cheated": "Multiple decisive cheating indicators were detected based on the evidence collected."
    }
    c.drawString(60, y, interpretation.get(verdict['verdict'], "No interpretation available."))
    y -= 20

    # ---------- FOOTER / SIGNATURE ----------
    if y < 80:
        c.showPage()
        y = h - 80
    c.setFont("Helvetica", 8)
    c.drawString(40, 50, f"Digitally signed using private key: {os.path.basename(private_key)}")
    c.drawString(40, 38, f"Generated by SmartProctor | {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    
    # Add exam time range in footer if available
    if exam_start and exam_end:
        start_str = datetime.fromtimestamp(exam_start).strftime('%H:%M:%S')
        end_str = datetime.fromtimestamp(exam_end).strftime('%H:%M:%S')
        c.drawString(40, 26, f"Exam Session: {start_str} to {end_str}")
    
    # Add evidence summary in footer
    total_evidence = len(audio_evidence) + sum(len(images) for images in image_evidence.values())
    c.drawString(40, 14, f"Total Evidence Files: {total_evidence} (Images: {sum(len(images) for images in image_evidence.values())}, Audio: {len(audio_evidence)})")
    
    c.save()

    # ---------- DIGITAL SIGNATURE ----------
    sig_path = out_pdf + ".sig"
    if os.path.exists(private_key):
        try:
            with open(out_pdf, "rb") as f:
                data = f.read()
            sig_bytes = sign_bytes(data, private_key)
            with open(sig_path, "wb") as sf:
                sf.write(sig_bytes)
            print(f"[Report] Signed report generated at {sig_path}")
        except Exception as e:
            print(f"[Report] Failed to sign report: {e}")

    print(f"[Report] Detailed report generated at {out_pdf}")
    return out_pdf