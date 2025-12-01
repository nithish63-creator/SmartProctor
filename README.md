**SmartProctor** is a privacy-focused, AI-powered exam monitoring system designed for remote assessments. It runs entirely on the candidate’s machine and uses multimodal monitoring—face authentication (InceptionResnetV1), gaze and head pose tracking (MediaPipe FaceMesh + OpenCV), object detection (YOLOv8m), and audio analysis (Silero VAD)—to detect cheating behaviors in real time. A pre-exam “front-desk” scan checks the workspace for unauthorized devices or materials, while an LSTM-based fusion model combines visual and audio cues over time to estimate cheating risk. All processing is done locally, and the system generates a digitally signed, tamper-proof exam report that can be reviewed by institutions without exposing raw video/audio to external servers.

**How to Run**
**1.Clone the repository**
git clone https://github.com/your-username/smartproctor.git
cd smartproctor

**2.Install dependencies**
pip install -r requirements.txt

**3.Run the application**
python run_exam.py
