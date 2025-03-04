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

def preprocess_frame(image, resize_dim=(72, 72)):
    face_img = crop_face(image)
    face_img = cv2.resize(face_img, resize_dim)
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    mean = np.mean(face_img, axis=(0, 1), keepdims=True)
    std = np.std(face_img, axis=(0, 1), keepdims=True) + 1e-6
    face_img = (face_img - mean) / std

    return np.transpose(face_img, (2, 0, 1))  # Convert to (C, H, W) for PyTorch
