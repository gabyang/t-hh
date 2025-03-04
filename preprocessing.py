import cv2
import numpy as np

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
    Preprocess a single frame for TSCAN model.
    
    Args:
        frame: BGR image from cv2.imread
    Returns:
        Preprocessed frame tensor of shape (C, H, W)
    """
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize to 16x16
    frame_resized = cv2.resize(frame_rgb, (16, 16))
    
    # Normalize to [0,1] range
    frame_normalized = frame_resized.astype('float32') / 255.0
    
    # Transpose from (H,W,C) to (C,H,W)
    frame_chw = np.transpose(frame_normalized, (2, 0, 1))
    
    return frame_chw
