import torch
import torchaudio
import sounddevice as sd
import numpy as np
import time
import threading
import queue

class SileroVADLiveTest:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.chunk_size = 512  # 32ms at 16kHz
        self.audio_queue = queue.Queue()
        self.is_running = False
        self.speech_count = 0
        
        # Load Silero VAD model
        print("Loading Silero VAD model...")
        try:
            self.model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                trust_repo=True,
                force_reload=False
            )
            (self.get_speech_timestamps, _, _, VADIterator, _) = utils
            
            # Initialize VAD iterator with optimized parameters
            self.vad_iterator = VADIterator(
                self.model,
                threshold=0.3,  # Lower threshold for better sensitivity
                min_silence_duration_ms=300,
                speech_pad_ms=100
            )
            print("✅ Silero VAD model loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def audio_callback(self, indata, frames, time_info, status):
        """Callback for sounddevice to capture audio"""
        if status:
            print(f"Audio callback status: {status}")
        
        # Convert to torch tensor and add to queue
        audio_data = indata[:, 0] if indata.ndim > 1 else indata
        
        # Normalize audio to proper range
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        
        audio_tensor = torch.from_numpy(audio_data.copy()).float()
        self.audio_queue.put(audio_tensor)
    
    def process_audio(self):
        """Process audio chunks for speech detection - IMPROVED VERSION"""
        print("Starting audio processing...")
        
        while self.is_running:
            try:
                # Get audio chunk from queue with timeout
                audio_chunk = self.audio_queue.get(timeout=1.0)
                
                # Ensure chunk is the right size
                if len(audio_chunk) < self.chunk_size:
                    # Pad if too short
                    padding = torch.zeros(self.chunk_size - len(audio_chunk))
                    audio_chunk = torch.cat([audio_chunk, padding])
                elif len(audio_chunk) > self.chunk_size:
                    # Truncate if too long
                    audio_chunk = audio_chunk[:self.chunk_size]
                
                # Calculate audio level for debugging
                rms_level = torch.sqrt(torch.mean(audio_chunk ** 2)).item()
                
                # Use VAD to detect speech - TRY BOTH METHODS
                try:
                    # Method 1: Use VAD iterator
                    speech_dict = self.vad_iterator(audio_chunk, return_seconds=False)
                    
                    if speech_dict is not None:
                        self.speech_count += 1
                        confidence = speech_dict.get('confidence', 0.0)
                        print(f"🎤 SPEECH DETECTED! (Count: {self.speech_count}, Confidence: {confidence:.3f}, RMS: {rms_level:.4f})")
                    else:
                        # Method 2: Fallback to direct model inference
                        confidence = self.model(audio_chunk, self.sample_rate).item()
                        if confidence > 0.5:  # Direct threshold check
                            self.speech_count += 1
                            print(f"🎤 SPEECH DETECTED (Direct)! (Count: {self.speech_count}, Confidence: {confidence:.3f}, RMS: {rms_level:.4f})")
                        else:
                            # Debug: Show audio levels occasionally
                            if hasattr(self, 'process_count'):
                                self.process_count += 1
                                if self.process_count % 100 == 0:
                                    print(f"Silent - RMS: {rms_level:.4f}, Confidence: {confidence:.3f}")
                            else:
                                self.process_count = 1
                            
                except Exception as e:
                    print(f"❌ VAD processing error: {e}")
                    # Reset VAD state on error
                    self.vad_iterator.reset_states()
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Error processing audio: {e}")
                continue
    
    def start_test(self, duration=30):
        """Start the live VAD test"""
        print(f"\n🎤 Silero VAD Live Test")
        print("=" * 50)
        print(f"Test will run for {duration} seconds")
        print("Please speak clearly into your microphone")
        print("Try speaking at normal volume for best results")
        print("Press Ctrl+C to stop early")
        print("=" * 50)
        
        self.is_running = True
        
        try:
            # Start audio processing thread
            process_thread = threading.Thread(target=self.process_audio, daemon=True)
            process_thread.start()
            
            # Start audio stream
            print(f"Starting audio stream at {self.sample_rate}Hz...")
            self.stream = sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                dtype='float32'
            )
            
            self.stream.start()
            print("✅ Audio stream started successfully!")
            print("🎤 Start speaking now...")
            print("💡 Tip: Speak clearly and at normal conversation volume")
            
            # Run for specified duration
            start_time = time.time()
            while time.time() - start_time < duration and self.is_running:
                elapsed = time.time() - start_time
                remaining = duration - elapsed
                
                # Show progress every 5 seconds
                if int(elapsed) % 5 == 0 and int(elapsed) > getattr(self, 'last_progress', 0):
                    print(f"⏱️  {int(elapsed)}s elapsed, {int(remaining)}s remaining")
                    self.last_progress = int(elapsed)
                
                time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\n⏹️  Test interrupted by user")
        except Exception as e:
            print(f"❌ Error during test: {e}")
        finally:
            self.is_running = False
            # Clean up
            if hasattr(self, 'stream'):
                self.stream.stop()
                self.stream.close()
            if hasattr(self, 'vad_iterator'):
                self.vad_iterator.reset_states()
            print(f"\n📊 Test Summary:")
            print(f"✅ Test completed")
            print(f"🎤 Total speech detections: {self.speech_count}")
            print(f"⏱️  Duration: {duration} seconds")

def main():
    # Check audio devices
    print("🔍 Checking audio devices...")
    try:
        devices = sd.query_devices()
        default_input = sd.default.device[0]
        input_device = devices[default_input]
        print(f"🎤 Using input device: {input_device['name']}")
    except Exception as e:
        print(f"⚠️  Could not query audio devices: {e}")
    
    # Run the test
    try:
        vad_test = SileroVADLiveTest(sample_rate=16000)
        vad_test.start_test(duration=30)
    except Exception as e:
        print(f"❌ Failed to run test: {e}")

if __name__ == "__main__":
    main()