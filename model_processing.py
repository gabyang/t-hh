import os
import cv2
import torch
import numpy as np
from pathlib import Path
import logging
from typing import Tuple
from models.tscan_model import TSCAN
from preprocessing import preprocess_frame

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TScanProcessor:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.frame_depth = 10  # From config
        self.model = self._load_model()
        
    def _load_model(self):
        model_path = Path("models/PURE_TSCAN.pth")
        logger.info(f"Loading model from {model_path}")
        
        # Initialize model
        model = TSCAN(frame_depth=self.frame_depth, drop_rate=0.2)
        
        try:
            # Load state dictionary
            checkpoint = torch.load(model_path, map_location=self.device)
            logger.info(f"Checkpoint loaded successfully")
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
                
            # Remove 'module.' prefix if it exists
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            # Load state dict
            model.load_state_dict(state_dict, strict=False)
            logger.info("Model state dictionary loaded successfully")
            
            model = model.to(self.device)
            model.eval()
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def _extract_frames(self, video_path: str) -> list:
        """Extract frames from video file."""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            
        cap.release()
        return frames

    def _calculate_heart_rate(self, signal: np.ndarray, fps: int = 30) -> float:
        """Calculate heart rate from the predicted signal using FFT."""
        # Remove DC component and normalize
        signal = signal - np.mean(signal)
        signal = signal / (np.std(signal) + 1e-6)
        
        # Apply FFT
        fft = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(len(signal), d=1.0/fps)
        
        # Find peak in physiologically possible heart rate range (40-180 BPM)
        mask = (freqs >= 40/60) & (freqs <= 180/60)
        peaks = np.abs(fft)[mask]
        peak_freq = freqs[mask][np.argmax(peaks)]
        
        # Convert to BPM
        heart_rate = peak_freq * 60
        return float(heart_rate)

    async def process_file(self, file_path: str) -> Tuple[float, float]:
        """Process either image or video file for rPPG analysis."""
        try:
            # Try to open as video first
            cap = cv2.VideoCapture(file_path)
            if cap.isOpened():
                logger.info("Processing video file...")
                frames = self._extract_frames(file_path)
                cap.release()
            else:
                # If not video, try as image
                logger.info("Processing image file...")
                frame = cv2.imread(file_path)
                if frame is None:
                    raise ValueError("Could not read file as image or video")
                # For single images, duplicate the frame
                frames = [frame] * self.frame_depth
            
            # Preprocess frames
            processed_frames = []
            for frame in frames:
                try:
                    processed = preprocess_frame(frame)
                    processed_frames.append(processed)
                except Exception as e:
                    logger.warning(f"Error processing frame: {str(e)}")
                    continue
            
            if not processed_frames:
                raise ValueError("No valid frames after preprocessing")
            
            # Stack frames and prepare for model
            processed_frames = np.stack(processed_frames, axis=0)  # (N, C, H, W)
            
            # Create clips
            if len(processed_frames) < self.frame_depth:
                raise ValueError(f"Need at least {self.frame_depth} valid frames")
            
            # Take the first frame_depth frames and reshape for 3D convolution
            clip = processed_frames[:self.frame_depth]  # (T, C, H, W)
            clip = np.transpose(clip, (1, 0, 2, 3))    # (C, T, H, W)
            clip = torch.from_numpy(clip).float().unsqueeze(0)  # (1, C, T, H, W)
            clip = clip.to(self.device)
            
            logger.info(f"Input shape: {clip.shape}")
            
            # Get predictions
            with torch.no_grad():
                predictions = self.model(clip)
                logger.info(f"Output shape: {predictions.shape}")
                
            # Convert predictions to numpy
            signal = predictions.cpu().numpy().squeeze()
            
            # Calculate heart rate
            heart_rate = self._calculate_heart_rate(signal)
            
            # SpO2 is a placeholder as it requires specific hardware
            spo2 = 98.0
            
            logger.info(f"Calculated HR: {heart_rate:.1f} bpm, SpO2: {spo2:.1f}%")
            return heart_rate, spo2
            
        except Exception as e:
            logger.error(f"Error in process_file: {str(e)}")
            raise

# Create global processor instance
processor = None

async def process_rppg(file_path: str) -> Tuple[float, float]:
    """Process file using TScan model."""
    global processor
    
    # Initialize processor if not already done
    if processor is None:
        processor = TScanProcessor()
    
    return await processor.process_file(file_path) 