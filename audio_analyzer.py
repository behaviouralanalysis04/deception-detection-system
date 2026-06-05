import librosa
import speech_recognition as sr
import numpy as np
import tempfile
import subprocess
import os

class AudioAnalyzer:
    def __init__(self, audio_path=None):
        self.audio_path = audio_path
        self.y = None
        self.sr = None
        self.duration = 0
    
    def load_audio(self, file_path=None):
        """Load audio file"""
        path = file_path or self.audio_path
        if not path or not os.path.exists(path):
            return None, None
        
        try:
            self.y, self.sr = librosa.load(path, sr=16000, duration=60)
            self.duration = len(self.y) / self.sr if self.sr else 0
            return self.y, self.sr
        except Exception as e:
            print(f"Error loading audio: {e}")
            return None, None
    
    def extract_audio_from_video(self, video_path):
        """Extract audio from video file"""
        audio_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
        try:
            subprocess.run([
                'ffmpeg', '-i', video_path, '-acodec', 'pcm_s16le',
                '-ar', '16000', '-ac', '1', audio_path, '-y',
                '-loglevel', 'quiet'
            ], check=True, timeout=60)
            return audio_path
        except Exception as e:
            print(f"FFmpeg error: {e}")
            return None
    
    def extract_all_features(self, y=None, sr=None):
        """Extract comprehensive audio features"""
        y = y or self.y
        sr = sr or self.sr
        
        if y is None or sr is None:
            return self._get_default_features()
        
        features = {}
        
        try:
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features['speech_tempo'] = float(tempo) if isinstance(tempo, (int, float)) else 120.0
            
            # Spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['avg_spectral_centroid'] = float(np.mean(spectral_centroids))
            features['std_spectral_centroid'] = float(np.std(spectral_centroids))
            
            # RMS energy
            rms = librosa.feature.rms(y=y)[0]
            features['avg_energy'] = float(np.mean(rms))
            features['std_energy'] = float(np.std(rms))
            features['energy_range'] = float(np.max(rms) - np.min(rms))
            
            # Pitch
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = []
            for i in range(pitches.shape[1]):
                index = magnitudes[:, i].argmax()
                pitch = pitches[index, i]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                features['avg_pitch'] = float(np.mean(pitch_values))
                features['std_pitch'] = float(np.std(pitch_values))
                features['pitch_range'] = float(np.max(pitch_values) - np.min(pitch_values))
            else:
                features['avg_pitch'] = 150.0
                features['std_pitch'] = 25.0
                features['pitch_range'] = 100.0
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features['avg_zcr'] = float(np.mean(zcr))
            features['speech_activity'] = float(np.mean(zcr > 0.01))
            
            # MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(min(13, mfccs.shape[0])):
                features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
                features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))
            
            # Spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features['avg_rolloff'] = float(np.mean(rolloff))
            
            # Spectral bandwidth
            bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            features['avg_bandwidth'] = float(np.mean(bandwidth))
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return self._get_default_features()
        
        return features
    
    def _get_default_features(self):
        """Return default audio features"""
        return {
            'speech_tempo': 120.0,
            'avg_spectral_centroid': 2000.0,
            'std_spectral_centroid': 500.0,
            'avg_energy': 0.05,
            'std_energy': 0.02,
            'energy_range': 0.1,
            'avg_pitch': 150.0,
            'std_pitch': 25.0,
            'pitch_range': 100.0,
            'avg_zcr': 0.05,
            'speech_activity': 0.5,
            'avg_rolloff': 3000.0,
            'avg_bandwidth': 2000.0
        }
    
    def transcribe_audio(self, file_path=None):
        """Transcribe speech to text"""
        path = file_path or self.audio_path
        if not path or not os.path.exists(path):
            return ""
        
        try:
            recognizer = sr.Recognizer()
            with sr.AudioFile(path) as source:
                recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = recognizer.record(source)
                text = recognizer.recognize_google(audio)
                return text
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""
    
    def calculate_deception_score(self, features):
        """Calculate deception probability from audio features"""
        score = 0
        
        if features.get('std_pitch', 0) > 30:
            score += 20
        if features.get('std_energy', 0) > 0.05:
            score += 15
        tempo = features.get('speech_tempo', 120)
        if tempo > 160 or tempo < 100:
            score += 15
        if features.get('energy_range', 0) > 0.15:
            score += 15
        if features.get('speech_activity', 0) < 0.4:
            score += 10
        
        return min(100, score)