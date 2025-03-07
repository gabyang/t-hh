import cv2
import numpy as np
from typing import Optional, Tuple

class FramePreprocessor:
    def __init__(self, face_cascade_path: str = "haarcascade_frontalface_default.xml"):
        """
        Initialize frame preprocessor.
        
        Args:
            face_cascade_path (str): Path to the face cascade classifier XML file
        """
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        if self.face_cascade.empty():
            raise ValueError(f"Failed to load face cascade classifier from {face_cascade_path}")

    def detect_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect face in the frame.
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            Optional[Tuple[int, int, int, int]]: Face coordinates (x, y, w, h) or None if no face detected
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return None
            
        # Get the largest face
        return max(faces, key=lambda x: x[2] * x[3])

    def extract_face(self, frame: np.ndarray, face_coords: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract and process face region from frame.
        
        Args:
            frame (np.ndarray): Input frame
            face_coords (Tuple[int, int, int, int]): Face coordinates (x, y, w, h)
            
        Returns:
            np.ndarray: Processed face image
        """
        x, y, w, h = face_coords
        face_img = frame[y:y+h, x:x+w]
        
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Resize to 16x16
        face_resized = cv2.resize(face_rgb, (16, 16))
        
        # Normalize to [-1, 1] range
        face_normalized = (face_resized.astype('float32') / 127.5) - 1.0
        
        # Transpose from (H,W,C) to (C,H,W)
        return np.transpose(face_normalized, (2, 0, 1))

    def preprocess_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Preprocess a single frame for TSCAN model.
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            Optional[np.ndarray]: Preprocessed frame or None if face detection fails
        """
        if frame is None:
            raise ValueError("Input frame is None")
            
        face_coords = self.detect_face(frame)
        if face_coords is None:
            print("Warning: No face detected in frame")
            return None
            
        return self.extract_face(frame, face_coords)

def crop_face(image, face_cascade_path="haarcascade_frontalface_default.xml", scale_factor=1.5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return image

    x, y, w, h = faces[0]
    cx, cy = x + w // 2, y + h // 2
    max_side = int(max(w, h) * scale_factor)
    x1, y1 = max(cx - max_side // 2, 0), max(cy - max_side // 2, 0)
    x2, y2 = min(cx + max_side // 2, image.shape[1]), min(cy + max_side // 2, image.shape[0])
    
    return image[y1:y2, x1:x2]

def preprocess_frame(frame):
    """
    Preprocess a single frame for TSCAN model with face detection verification.
    """
    # Face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        print("Warning: No face detected in frame")
        return None
    
    # Get the largest face
    largest_face = max(faces, key=lambda x: x[2] * x[3])
    x, y, w, h = largest_face
    
    # Extract and process face region
    face_img = frame[y:y+h, x:x+w]
    
    # Convert BGR to RGB
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    
    # Resize to 16x16
    face_resized = cv2.resize(face_rgb, (16, 16))
    
    # Normalize to [-1, 1] range
    face_normalized = (face_resized.astype('float32') / 127.5) - 1.0
    
    # Transpose from (H,W,C) to (C,H,W)
    face_chw = np.transpose(face_normalized, (2, 0, 1))
    
    return face_chw
