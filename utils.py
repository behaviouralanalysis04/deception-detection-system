import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe for face detection and landmarks (lightweight)
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

face_mesh = None
face_detection = None

def init_face_detection():
    global face_mesh, face_detection
    try:
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        face_detection = mp_face_detection.FaceDetection(
            model_selection=1, 
            min_detection_confidence=0.5
        )
        return True
    except:
        return False

# Initialize on import
init_face_detection()

# Face landmark indices (MediaPipe has 468 points, we map to ~68 key points)
# Key facial landmarks mapping
FACE_INDICES = {
    'left_eye': [33, 133, 157, 158, 159, 160, 161, 173],
    'right_eye': [362, 263, 387, 386, 385, 384, 398, 466],
    'left_eyebrow': [46, 53, 52, 65, 55],
    'right_eyebrow': [276, 283, 282, 295, 285],
    'nose': [1, 2, 4, 5, 6, 19, 94, 195],
    'mouth': [61, 78, 81, 13, 311, 308, 324, 318, 402, 317, 14, 87],
    'jaw': [10, 338, 297, 332, 284, 251, 389]
}

def get_landmarks_mediapipe(frame):
    """Extract facial landmarks using MediaPipe"""
    if face_mesh is None:
        return None
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if not results.multi_face_landmarks:
        return None
    
    h, w = frame.shape[:2]
    landmarks = []
    face_landmarks = results.multi_face_landmarks[0]
    
    # Extract key landmarks (simplified to 68 points approximation)
    for idx in range(468):
        x = int(face_landmarks.landmark[idx].x * w)
        y = int(face_landmarks.landmark[idx].y * h)
        landmarks.append((x, y))
    
    return landmarks

def get_landmarks_haar(frame):
    """Fallback: Simple face detection using Haar Cascade"""
    cascade_path = "haarcascade_frontalface_default.xml"
    try:
        cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5)
        
        if len(faces) == 0:
            return None
        
        # Return approximate landmarks based on face bounding box
        x, y, w, h = faces[0]
        landmarks = []
        
        # Generate approximate facial points
        for i in range(68):
            if i < 36:  # Face contour
                px = x + int(w * (i / 67))
                py = y + int(h * 0.3)
            elif i < 48:  # Eyes
                if i < 42:  # Left eye
                    px = x + int(w * 0.35 + (i-36) * (w * 0.03))
                else:  # Right eye
                    px = x + int(w * 0.65 + (i-42) * (w * 0.03))
                py = y + int(h * 0.35)
            else:  # Mouth
                px = x + int(w * 0.5 + (i-48) * (w * 0.02))
                py = y + int(h * 0.7)
            landmarks.append((px, py))
        
        return landmarks
    except:
        return None

def get_landmarks(frame):
    """Main function to get landmarks with fallback"""
    landmarks = get_landmarks_mediapipe(frame)
    if landmarks is None:
        landmarks = get_landmarks_haar(frame)
    return landmarks

def eye_aspect_ratio(landmarks, eye_indices):
    """Calculate eye aspect ratio"""
    if len(landmarks) <= max(eye_indices):
        return 0.25  # Default value
    
    pts = np.array([landmarks[i] for i in eye_indices])
    if len(pts) < 6:
        return 0.25
    
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C) if C > 0 else 0.25

def mouth_aspect_ratio(landmarks):
    """Calculate mouth aspect ratio"""
    if len(landmarks) < 68:
        return 0.3
    
    try:
        outer_indices = list(range(48, 60))
        outer = np.array([landmarks[i] for i in outer_indices if i < len(landmarks)])
        if len(outer) > 9:
            A = np.linalg.norm(outer[3] - outer[9])
            B = np.linalg.norm(outer[0] - outer[6])
            return A / B if B > 0 else 0.3
    except:
        pass
    return 0.3

def lip_compression(landmarks):
    """Measure lip compression"""
    if len(landmarks) < 68:
        return 10.0
    
    try:
        upper = np.array(landmarks[51])
        lower = np.array(landmarks[57])
        return np.linalg.norm(upper - lower)
    except:
        return 10.0

def facial_asymmetry(landmarks):
    """Calculate facial asymmetry score"""
    if len(landmarks) < 68:
        return 10.0
    
    try:
        left_indices = [36, 37, 38, 48, 49, 50]
        right_indices = [45, 46, 47, 54, 53, 52]
        
        left_pts = np.array([landmarks[i] for i in left_indices if i < len(landmarks)])
        right_pts = np.array([landmarks[i] for i in right_indices if i < len(landmarks)])
        
        if len(left_pts) > 0 and len(right_pts) > 0:
            return np.linalg.norm(np.mean(left_pts, axis=0) - np.mean(right_pts, axis=0))
    except:
        pass
    return 10.0

def gaze_direction(landmarks, frame_width):
    """Determine gaze direction"""
    if len(landmarks) < 68:
        return "center"
    
    try:
        left_eye_center = np.mean([landmarks[i] for i in range(36, 42)], axis=0)
        right_eye_center = np.mean([landmarks[i] for i in range(42, 48)], axis=0)
        eye_center = (left_eye_center + right_eye_center) / 2
        nose = np.array(landmarks[30])
        
        offset = nose[0] - eye_center[0]
        threshold = frame_width * 0.05
        
        if offset > threshold:
            return "right"
        elif offset < -threshold:
            return "left"
    except:
        pass
    return "center"

def head_pose(landmarks, w, h):
    """Estimate head pose (simplified)"""
    if len(landmarks) < 68:
        return 0, 0, 0
    
    try:
        # Simplified head pose estimation based on eye and nose positions
        left_eye = np.mean([landmarks[i] for i in range(36, 42)], axis=0)
        right_eye = np.mean([landmarks[i] for i in range(42, 48)], axis=0)
        nose = np.array(landmarks[30])
        chin = np.array(landmarks[8])
        
        # Pitch (up/down)
        eye_y = (left_eye[1] + right_eye[1]) / 2
        pitch = (nose[1] - eye_y) / h * 30
        
        # Yaw (left/right)
        eye_center_x = (left_eye[0] + right_eye[0]) / 2
        yaw = (nose[0] - eye_center_x) / w * 40
        
        # Roll (head tilt)
        roll = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]) * 180 / np.pi
        
        return pitch, yaw, roll
    except:
        return 0, 0, 0

def micro_expression_magnitude(prev_landmarks, curr_landmarks):
    """Calculate micro-expression magnitude"""
    if prev_landmarks is None or curr_landmarks is None:
        return 0
    
    n = min(len(prev_landmarks), len(curr_landmarks), 68)
    if n == 0:
        return 0
    
    movements = []
    for i in range(n):
        movement = np.linalg.norm(np.array(prev_landmarks[i]) - np.array(curr_landmarks[i]))
        movements.append(movement)
    
    return np.mean(movements)