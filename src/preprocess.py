import librosa
import noisereduce as nr
from faster_whisper import WhisperModel
import os
import soundfile as sf

def preprocess_audio(audio_path, output_dir):
    """Preprocess audio: normalize, denoise, and transcribe."""
    try:
        # Load and normalize
        audio, sr = librosa.load(audio_path, sr=16000)
        audio = nr.reduce_noise(y=audio, sr=sr)
        
        # Save processed audio
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(audio_path))
        sf.write(output_path, audio, sr)
        
        # Transcribe using faster-whisper
        model = WhisperModel("base", device="cpu")
        segments, _ = model.transcribe(audio_path, beam_size=5)
        transcript = " ".join(segment.text for segment in segments)
        
        # Save transcript
        transcript_path = output_path.replace(".wav", ".txt")
        with open(transcript_path, "w") as f:
            f.write(transcript)
        
        return audio, sr, transcript
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None, None, None