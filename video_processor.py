import cv2
import numpy as np
import time
from collections import deque
import utils

class VideoAnalysisProcessor:
    def __init__(self):
        self.frame_count = 0
        self.blink_counter = 0
        self.eye_closed_start = None
        self.gaze_durations = {'left': 0, 'right': 0, 'center': 0}
        
        # Trackers for temporal features
        self.mar_values = deque(maxlen=300)
        self.lip_comp_values = deque(maxlen=300)
        self.asymmetry_values = deque(maxlen=300)
        self.head_pitch_values = deque(maxlen=300)
        self.head_yaw_values = deque(maxlen=300)
        self.head_roll_values = deque(maxlen=300)
        
        self.prev_head_pitch = None
        self.prev_head_yaw = None
        self.prev_head_roll = None
        self.prev_landmarks = None
        self.micro_expression_frames = 0
        self.nod_count = 0
        self.shake_count = 0
        self.tilt_count = 0
        self.processing_times = deque(maxlen=30)
    
    def process_frame(self, frame, frame_idx, fps=30):
        """Process a single frame and extract features"""
        if frame is None:
            return None
        
        start_time = time.time()
        h, w = frame.shape[:2]
        
        # Get landmarks
        landmarks = utils.get_landmarks(frame)
        
        if landmarks is None:
            return None
        
        # Eye blink detection
        left_ear = utils.eye_aspect_ratio(landmarks, list(range(36, 42)))
        right_ear = utils.eye_aspect_ratio(landmarks, list(range(42, 48)))
        ear = (left_ear + right_ear) / 2.0
        
        # Blink threshold
        if ear < 0.2:
            if self.eye_closed_start is None:
                self.eye_closed_start = frame_idx
        else:
            if self.eye_closed_start is not None:
                self.blink_counter += 1
                self.eye_closed_start = None
        
        # Gaze direction
        gaze = utils.gaze_direction(landmarks, w)
        self.gaze_durations[gaze] += 1
        
        # Mouth features
        mar = utils.mouth_aspect_ratio(landmarks)
        self.mar_values.append(mar)
        
        lip_comp = utils.lip_compression(landmarks)
        self.lip_comp_values.append(lip_comp)
        
        asymmetry = utils.facial_asymmetry(landmarks)
        self.asymmetry_values.append(asymmetry)
        
        # Head pose
        pitch, yaw, roll = utils.head_pose(landmarks, w, h)
        self.head_pitch_values.append(pitch)
        self.head_yaw_values.append(yaw)
        self.head_roll_values.append(roll)
        
        # Head movement detection
        if self.prev_head_pitch is not None:
            if abs(pitch - self.prev_head_pitch) > 5:
                self.nod_count += 1
            if self.prev_head_yaw is not None and abs(yaw - self.prev_head_yaw) > 10:
                self.shake_count += 1
            if self.prev_head_roll is not None and abs(roll - self.prev_head_roll) > 8:
                self.tilt_count += 1
        
        self.prev_head_pitch, self.prev_head_yaw, self.prev_head_roll = pitch, yaw, roll
        
        # Micro-expressions
        if self.prev_landmarks is not None:
            movement = utils.micro_expression_magnitude(self.prev_landmarks, landmarks)
            if movement > 5.0:
                self.micro_expression_frames += 1
        
        self.prev_landmarks = landmarks
        self.frame_count = frame_idx
        
        # Track processing performance
        self.processing_times.append(time.time() - start_time)
        
        return landmarks
    
    def get_features(self, duration_sec, fps=30):
        """Extract aggregated features after processing"""
        total_frames = max(1, self.frame_count)
        
        features = {
            'blink_rate': (self.blink_counter / total_frames) * fps * 60 if total_frames > 0 else 0,
            'avg_blink_duration': 0.2,
            'gaze_left_ratio': self.gaze_durations['left'] / total_frames,
            'gaze_right_ratio': self.gaze_durations['right'] / total_frames,
            'gaze_center_ratio': self.gaze_durations['center'] / total_frames,
            'avg_mouth_open_ratio': float(np.mean(self.mar_values)) if self.mar_values else 0.0,
            'std_mouth_open_ratio': float(np.std(self.mar_values)) if self.mar_values else 0.0,
            'avg_facial_asymmetry': float(np.mean(self.asymmetry_values)) if self.asymmetry_values else 0.0,
            'std_facial_asymmetry': float(np.std(self.asymmetry_values)) if self.asymmetry_values else 0.0,
            'avg_lip_compression': float(np.mean(self.lip_comp_values)) if self.lip_comp_values else 0.0,
            'micro_expression_frequency': self.micro_expression_frames / max(duration_sec, 0.1),
            'avg_head_pitch': float(np.mean(self.head_pitch_values)) if self.head_pitch_values else 0.0,
            'std_head_pitch': float(np.std(self.head_pitch_values)) if self.head_pitch_values else 0.0,
            'avg_head_yaw': float(np.mean(self.head_yaw_values)) if self.head_yaw_values else 0.0,
            'std_head_yaw': float(np.std(self.head_yaw_values)) if self.head_yaw_values else 0.0,
            'avg_head_roll': float(np.mean(self.head_roll_values)) if self.head_roll_values else 0.0,
            'std_head_roll': float(np.std(self.head_roll_values)) if self.head_roll_values else 0.0,
            'head_nod_frequency': self.nod_count / max(duration_sec, 0.1),
            'head_shake_frequency': self.shake_count / max(duration_sec, 0.1),
            'head_tilt_frequency': self.tilt_count / max(duration_sec, 0.1),
            'duration_seconds': duration_sec
        }
        
        # Cap values at reasonable limits
        features['blink_rate'] = min(features['blink_rate'], 45.0)
        
        return features
    
    def generate_summary(self, features, deception_score):
        """Generate analysis summary"""
        indicators = []
        
        if features['blink_rate'] > 30:
            indicators.append(('Elevated blink rate', 'high'))
        elif features['blink_rate'] < 10:
            indicators.append(('Reduced blink rate', 'medium'))
        
        gaze_aversion = features['gaze_left_ratio'] + features['gaze_right_ratio']
        if gaze_aversion > 0.6:
            indicators.append(('Frequent gaze aversion', 'high'))
        elif gaze_aversion > 0.4:
            indicators.append(('Occasional gaze aversion', 'medium'))
        
        if features['avg_lip_compression'] > 15:
            indicators.append(('Lip compression detected', 'high'))
        
        if features['avg_facial_asymmetry'] > 15:
            indicators.append(('Facial asymmetry detected', 'medium'))
        
        if features['micro_expression_frequency'] > 2:
            indicators.append(('Frequent micro-expressions', 'high'))
        elif features['micro_expression_frequency'] > 1:
            indicators.append(('Occasional micro-expressions', 'medium'))
        
        if features.get('head_nod_frequency', 0) > 1.5:
            indicators.append(('Excessive head nodding', 'medium'))
        
        classification = (
            'HIGH PROBABILITY OF DECEPTION' if deception_score >= 60 else
            'POSSIBLE DECEPTION' if deception_score >= 40 else
            'LOW PROBABILITY OF DECEPTION'
        )
        
        return {
            'deception_score': deception_score,
            'indicators': indicators,
            'classification': classification,
            'color': '#dc3545' if deception_score >= 60 else '#ffc107' if deception_score >= 40 else '#28a745',
            'icon': '🔴' if deception_score >= 60 else '🟡' if deception_score >= 40 else '🟢',
            'confidence': 'High' if len(indicators) >= 4 else 'Medium' if len(indicators) >= 2 else 'Low'
        }
    
    def get_performance_stats(self):
        """Get processing performance statistics"""
        if self.processing_times:
            return {
                'avg_processing_time_ms': np.mean(self.processing_times) * 1000,
                'fps': 1.0 / np.mean(self.processing_times) if self.processing_times else 0
            }
        return {'avg_processing_time_ms': 0, 'fps': 0}