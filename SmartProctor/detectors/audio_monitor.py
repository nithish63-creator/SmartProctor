import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
import time
import os
import torch
import queue
import warnings
from typing import Optional, Dict, Any
warnings.filterwarnings('ignore')

class AudioMonitor:
    def __init__(self, sample_rate=16000, chunk_duration=0.032,
                 min_duration=1.0, debounce_sec=8.0,
                 save_dir="storage_files/evidence", debug=True):
        
        # Audio parameters
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        self.min_duration = min_duration
        self.debounce_sec = debounce_sec
        self.debug = debug
        self.save_dir = save_dir

        # State management
        self._running = False
        self._speaking = False
        self._process_thread = None
        self._speech_start = None
        self._last_event_time = 0
        
        # Audio queue for real-time processing
        self.audio_queue = queue.Queue(maxsize=100)
        
        # Silero VAD components
        self.model = None
        self.vad_iterator = None
        self.utils = None
        
        # IMPROVED: More sensitive speech detection
        self._speech_window = []  # Rolling window of recent detections
        self._window_size = int(0.5 / chunk_duration)  # REDUCED: 0.5-second window (15-16 chunks)
        self.required_speech_ratio = 0.3  # REDUCED: 30% speech in window to trigger
        self.required_silence_ratio = 0.7  # 70% silence to stop
        
        # Alternative: Simple consecutive counter (more responsive)
        self._consecutive_speech = 0
        self._consecutive_silence = 0
        self._min_consecutive_speech = 8  # ~250ms of speech
        self._min_consecutive_silence = 16  # ~500ms of silence
        
        # Audio device
        self.audio_device = None
        
        # Evidence buffer
        self._evidence_buffer = []
        self._max_buffer_duration = 10
        
        # Statistics
        self._total_speech_events = 0
        self._speech_event_count = 0
        
        print(f"[AudioMonitor] Initialized with sensitive detection")
        print(f"[AudioMonitor] Chunk size: {self.chunk_size} samples ({chunk_duration*1000}ms)")
        print(f"[AudioMonitor] Window size: {self._window_size} chunks ({0.5}s)")
        print(f"[AudioMonitor] Speech ratio threshold: {self.required_speech_ratio}")
        print(f"[AudioMonitor] Consecutive speech threshold: {self._min_consecutive_speech} chunks")

    def _get_audio_device(self):
        """Find a suitable audio device"""
        try:
            devices = sd.query_devices()
            default_input = sd.default.device[0] if sd.default.device else None
            
            if default_input is not None:
                device_info = sd.query_devices(default_input, 'input')
                if device_info['max_input_channels'] > 0:
                    if self.debug:
                        print(f"[AudioMonitor] Using input device: {device_info['name']}")
                    return default_input
            
            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    if self.debug:
                        print(f"[AudioMonitor] Using fallback device: {dev['name']}")
                    return i
                    
            print("[AudioMonitor] ⚠️ No input devices found!")
            return None
            
        except Exception as e:
            print(f"[AudioMonitor] Error querying audio devices: {e}")
            return None

    def _initialize_silero_vad(self):
        """Initialize Silero VAD model with more sensitive settings"""
        try:
            print("[AudioMonitor] Loading Silero VAD model...")
            
            self.model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                trust_repo=True,
                force_reload=False,
                verbose=False
            )
            
            (_, _, _, VADIterator, _) = utils
            
            # MORE SENSITIVE VAD SETTINGS
            self.vad_iterator = VADIterator(
                self.model,
                threshold=0.15,  # LOWER threshold for more sensitivity
                min_silence_duration_ms=200,  # Shorter silence detection
                speech_pad_ms=50  # Less padding
            )
            
            print("[AudioMonitor] ✅ Silero VAD model loaded successfully")
            return True
            
        except Exception as e:
            print(f"[AudioMonitor] ❌ Failed to load Silero VAD: {e}")
            return False

    def audio_callback(self, indata, frames, time_info, status):
        """Callback for real-time audio capture"""
        if status and self.debug:
            print(f"[AudioMonitor] Audio callback status: {status}")
        
        try:
            if indata.ndim > 1:
                audio_chunk = indata[:, 0].copy()
            else:
                audio_chunk = indata.copy()
            
            # Calculate audio level for debugging
            rms = np.sqrt(np.mean(audio_chunk**2))
            
            # Only process if there's significant audio to save CPU
            if rms > 0.001 or not self.audio_queue.full():
                self.audio_queue.put((audio_chunk, rms))
                
            self._evidence_buffer.append(audio_chunk)
            buffer_duration = len(self._evidence_buffer) * self.chunk_duration
            if buffer_duration > self._max_buffer_duration:
                self._evidence_buffer.pop(0)
                
        except Exception as e:
            if self.debug:
                print(f"[AudioMonitor] Error in audio callback: {e}")

    def process_audio_chunks(self):
        """Process audio chunks with more sensitive detection"""
        if self.debug:
            print("[AudioMonitor] Starting audio processing thread...")
        
        while self._running:
            try:
                audio_chunk, rms = self.audio_queue.get(timeout=1.0)
                
                if len(audio_chunk) < self.chunk_size:
                    padding = np.zeros(self.chunk_size - len(audio_chunk), dtype=np.float32)
                    audio_chunk = np.concatenate([audio_chunk, padding])
                
                audio_tensor = torch.from_numpy(audio_chunk).float()
                
                try:
                    # Use VAD to detect speech
                    speech_dict = self.vad_iterator(audio_tensor, return_seconds=False)
                    
                    is_speech = False
                    confidence = 0.0
                    
                    if speech_dict is not None:
                        is_speech = True
                        confidence = speech_dict.get('confidence', 0.0)
                    else:
                        # Use direct model with lower threshold
                        confidence = self.model(audio_tensor, self.sample_rate).item()
                        is_speech = confidence > 0.3  # LOWER threshold
                    
                    # Use BOTH methods for more reliable detection
                    self._update_speech_state_sensitive(is_speech, confidence, rms)
                    
                except Exception as e:
                    if self.debug:
                        print(f"[AudioMonitor] VAD processing error: {e}")
                    self.vad_iterator.reset_states()
                    
            except queue.Empty:
                continue
            except Exception as e:
                if self.debug:
                    print(f"[AudioMonitor] Error processing audio: {e}")
                continue

    def _update_speech_state_sensitive(self, is_speech: bool, confidence: float, rms: float):
        """MORE SENSITIVE: Use both window and consecutive methods"""
        current_time = time.time()
        
        # Method 1: Rolling window approach
        self._speech_window.append(is_speech)
        if len(self._speech_window) > self._window_size:
            self._speech_window.pop(0)
        
        # Method 2: Consecutive counter (more responsive)
        if is_speech:
            self._consecutive_speech += 1
            self._consecutive_silence = 0
            
            # Debug: Show when we have speech with good confidence
            if self.debug and confidence > 0.5:
                if hasattr(self, '_last_speech_debug'):
                    if current_time - self._last_speech_debug > 1.0:
                        print(f"[AudioMonitor] Speech detected (conf: {confidence:.3f}, RMS: {rms:.4f}, consecutive: {self._consecutive_speech})")
                        self._last_speech_debug = current_time
                else:
                    self._last_speech_debug = current_time
        else:
            self._consecutive_speech = 0
            self._consecutive_silence += 1
        
        # Calculate speech ratio for window method
        speech_ratio = 0.0
        if len(self._speech_window) > 0:
            speech_ratio = sum(self._speech_window) / len(self._speech_window)
        
        # TRIGGER LOGIC: Use EITHER method to start speech
        if not self._speaking:
            # Option A: Enough consecutive speech chunks
            consecutive_trigger = (self._consecutive_speech >= self._min_consecutive_speech)
            
            # Option B: Good speech ratio in window  
            window_trigger = (speech_ratio >= self.required_speech_ratio)
            
            # Option C: Single very high confidence detection
            strong_trigger = (confidence > 0.8 and self._consecutive_speech >= 3)
            
            if (consecutive_trigger or window_trigger or strong_trigger):
                if (current_time - self._last_event_time) >= self.debounce_sec:
                    self._speech_start = current_time
                    self._speaking = True
                    self._speech_event_count += 1
                    trigger_type = "consecutive" if consecutive_trigger else "window" if window_trigger else "strong"
                    print(f"[AudioMonitor] 🎤 SPEECH STARTED (#{self._speech_event_count}, {trigger_type}, conf: {confidence:.3f})")
        
        # STOP LOGIC: Use silence to end speech
        elif self._speaking:
            # Stop if we have enough consecutive silence
            if self._consecutive_silence >= self._min_consecutive_silence:
                duration = current_time - self._speech_start
                if duration >= self.min_duration:
                    self._last_event_time = current_time
                    self._total_speech_events += 1
                    print(f"[AudioMonitor] 🎤 Speech ended ({duration:.1f}s, total: {self._total_speech_events})")
                    self._save_audio_evidence(duration)
                
                # Reset state
                self._speaking = False
                self._speech_start = None
                self._consecutive_speech = 0
                # Don't clear window completely, just reset counter
                self._consecutive_silence = 0

    def _save_audio_evidence(self, duration: float):
        """Save audio evidence when speech is detected and return file path"""
        try:
            if not self._evidence_buffer:
                return None
                
            os.makedirs(self.save_dir, exist_ok=True)
            
            save_duration = min(duration + 1.0, 6.0)  # Shorter evidence
            save_chunks = min(int(save_duration / self.chunk_duration), len(self._evidence_buffer))
            
            if save_chunks == 0:
                return None
                
            audio_data = np.concatenate(self._evidence_buffer[-save_chunks:])
            
            timestamp = int(time.time())
            filename = f"speech_evidence_{timestamp}.wav"
            filepath = os.path.join(self.save_dir, filename)
            
            sf.write(filepath, audio_data, self.sample_rate)
            
            print(f"[AudioMonitor] 💾 Audio saved: {filename} ({len(audio_data)/self.sample_rate:.1f}s)")
            return filepath  # Make sure this line returns the path
            
        except Exception as e:
            if self.debug:
                print(f"[AudioMonitor] Failed to save audio: {e}")
            return None
            
        except Exception as e:
            if self.debug:
                print(f"[AudioMonitor] Failed to save audio: {e}")
            return None

    def start(self):
        """Start audio monitoring"""
        if self._running:
            print("[AudioMonitor] Audio monitor is already running")
            return True
            
        print("[AudioMonitor] Starting audio monitor...")
        
        self.audio_device = self._get_audio_device()
        if self.audio_device is None:
            print("[AudioMonitor] ❌ No audio input device available")
            return False
        
        if not self._initialize_silero_vad():
            print("[AudioMonitor] ❌ Failed to initialize Silero VAD")
            return False
        
        self._running = True
        
        try:
            self._process_thread = threading.Thread(
                target=self.process_audio_chunks, 
                daemon=True,
                name="AudioProcessor"
            )
            self._process_thread.start()
            
            print(f"[AudioMonitor] Starting audio stream at {self.sample_rate}Hz...")
            self.audio_stream = sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                device=self.audio_device,
                dtype='float32'
            )
            
            self.audio_queue.put((np.zeros(self.chunk_size, dtype=np.float32), 0.0))  # Prime the queue
            
            self.audio_stream.start()
            
            time.sleep(1.0)
            print("[AudioMonitor] ✅ Audio monitoring started successfully")
            return True
            
        except Exception as e:
            print(f"[AudioMonitor] ❌ Failed to start audio stream: {e}")
            self._running = False
            return False

    def stop(self):
        """Stop audio monitoring"""
        if not self._running:
            return
            
        print("[AudioMonitor] Stopping audio monitor...")
        self._running = False
        
        if hasattr(self, 'audio_stream') and self.audio_stream:
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
            except Exception as e:
                if self.debug:
                    print(f"[AudioMonitor] Error stopping audio stream: {e}")
        
        if self.vad_iterator:
            self.vad_iterator.reset_states()
        
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
                
        self._evidence_buffer.clear()
        
        if self._process_thread and self._process_thread.is_alive():
            self._process_thread.join(timeout=2.0)
        
        self._process_thread = None
        print("[AudioMonitor] 🔴 Audio monitoring stopped")

    def is_speaking(self):
        """Check if speech is currently detected"""
        return self._speaking

    def is_running(self):
        """Check if monitor is running"""
        return (self._running and 
                hasattr(self, 'audio_stream') and 
                self.audio_stream.active)

    def get_stats(self):
        """Get current monitoring statistics"""
        speech_ratio = 0.0
        if hasattr(self, '_speech_window') and len(self._speech_window) > 0:
            speech_ratio = sum(self._speech_window) / len(self._speech_window)
            
        return {
            "running": self.is_running(),
            "speaking": self._speaking,
            "queue_size": self.audio_queue.qsize(),
            "buffer_duration": len(self._evidence_buffer) * self.chunk_duration,
            "total_speech_events": self._total_speech_events,
            "consecutive_speech": self._consecutive_speech,
            "consecutive_silence": self._consecutive_silence,
            "speech_ratio": round(speech_ratio, 2)
        }

def test_audio_monitor():
    """Test the sensitive audio monitor"""
    monitor = AudioMonitor(
        sample_rate=16000,
        chunk_duration=0.032,
        min_duration=1.0,
        debounce_sec=5.0,
        debug=True
    )
    
    try:
        if monitor.start():
            print("Audio monitor started. Press Ctrl+C to stop...")
            print("Speak at normal conversation volume to test...")
            
            last_stats_time = time.time()
            while monitor.is_running():
                current_time = time.time()
                
                if current_time - last_stats_time >= 2.0:  # Every 2 seconds
                    stats = monitor.get_stats()
                    print(f"Stats: {stats}")
                    last_stats_time = current_time
                
                time.sleep(0.1)
        else:
            print("Failed to start audio monitor")
    except KeyboardInterrupt:
        print("\nStopping audio monitor...")
    finally:
        monitor.stop()

if __name__ == "__main__":
    test_audio_monitor()