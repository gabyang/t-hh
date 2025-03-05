import os
import glob
from typing import Optional
import torch
import numpy as np
import cv2
from models.tscan_model import TSCAN
from preprocessing import FramePreprocessor
from signal_processing import HeartRateCalculator
from visualization import SignalVisualizer

class TSCANInference:
    def __init__(self, checkpoint_path: str, device: str = 'cuda:0', 
                 frame_depth: int = 10, drop_rate: float = 0.2,
                 fps: int = 30):
        """
        Initialize TSCAN inference pipeline.
        
        Args:
            checkpoint_path (str): Path to model checkpoint
            device (str): Device to run inference on
            frame_depth (int): Number of frames per clip
            drop_rate (float): Dropout rate for model
            fps (int): Frames per second of input video
        """
        self.device = device
        self.frame_depth = frame_depth
        self.fps = fps
        
        # Initialize components
        self.model = self._load_model(checkpoint_path, device, frame_depth, drop_rate)
        self.preprocessor = FramePreprocessor()
        self.hr_calculator = HeartRateCalculator(fps=fps)
        self.visualizer = SignalVisualizer(fps=fps)

    def _load_model(self, checkpoint_path: str, device: str, 
                   frame_depth: int, drop_rate: float) -> TSCAN:
        """Load TSCAN model from checkpoint."""
        model = TSCAN(frame_depth=frame_depth, drop_rate=drop_rate)
        
        # Load the state dict and remove the 'module.' prefix if it exists
        state_dict = torch.load(checkpoint_path, map_location=device)
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(new_state_dict)
        model.eval()
        model.to(device)
        return model

    def make_clips(self, frames: np.ndarray, clip_length: int, stride: int = 1) -> np.ndarray:
        """
        Group frames into consecutive clips with overlap.
        
        Args:
            frames (np.ndarray): Array of shape (N, C, H, W)
            clip_length (int): Length of each clip
            stride (int): Number of frames to shift window
            
        Returns:
            np.ndarray: Array of clips
        """
        clips = []
        for i in range(0, len(frames) - clip_length + 1, stride):
            clip = frames[i : i + clip_length]
            clips.append(clip)
        return np.array(clips)

    def run_inference(self, folder_path: str, batch_size: int = 4, 
                     stride: int = 1) -> Optional[np.ndarray]:
        """
        Run inference on a folder of images.
        
        Args:
            folder_path (str): Path to folder containing images
            batch_size (int): Batch size for inference
            stride (int): Stride for clip creation
            
        Returns:
            Optional[np.ndarray]: Predicted PPG signals or None if inference fails
        """
        # 1) Gather all image paths
        image_paths = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
        print(f"Found {len(image_paths)} frames")

        # 2) Preprocess frames
        preprocessed_frames = []
        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None:
                continue
            frame_tensor = self.preprocessor.preprocess_frame(img)
            if frame_tensor is not None:
                preprocessed_frames.append(frame_tensor)

        if len(preprocessed_frames) < self.frame_depth:
            print("Not enough frames to form a single clip.")
            return None

        # 3) Convert list to numpy array
        preprocessed_frames = np.stack(preprocessed_frames, axis=0)

        # 4) Split into overlapping clips
        clips = self.make_clips(preprocessed_frames, self.frame_depth, stride=stride)
        print(f"Number of clips (with overlap): {clips.shape[0]}")

        # 5) Inference
        all_ppg_signals = []
        with torch.no_grad():
            for start_idx in range(0, len(clips), batch_size):
                batch_clips = clips[start_idx : start_idx + batch_size]
                
                # Reorder dimensions and ensure correct size
                batch_clips = np.transpose(batch_clips, (0, 2, 1, 3, 4))
                batch_clips_torch = torch.from_numpy(batch_clips).float().to(self.device)
                
                # Forward pass
                ppg_pred = self.model(batch_clips_torch)
                ppg_pred_np = ppg_pred.cpu().numpy()
                all_ppg_signals.append(ppg_pred_np)

        # Concatenate predictions across all batches
        if all_ppg_signals:
            return np.concatenate(all_ppg_signals, axis=0)
        return None

    def _print_summary(self, hr_bpm: float, bvp_analysis: dict) -> None:
        """
        Print a comprehensive summary of the analysis results.
        
        Args:
            hr_bpm (float): Estimated heart rate in BPM
            bvp_analysis (dict): BVP analysis results
        """
        print("\n" + "="*50)
        print("ANALYSIS SUMMARY")
        print("="*50)
        
        # Heart Rate Summary
        print("\nHEART RATE:")
        print(f"Estimated BPM: {hr_bpm:.1f}")
        status = "Valid" if hr_bpm > 0 else "Invalid"
        print(f"Status: {status}")
        
        # BVP Summary
        print("\nBLOOD VOLUME PULSE (BVP):")
        if bvp_analysis is not None:
            print(f"Status: {bvp_analysis['message']}")
            print(f"Amplitude: {bvp_analysis['amplitude']:.3f}")
            print(f"Peak Quality: {bvp_analysis['peak_quality']:.2f}")
            print(f"Mean Interval: {bvp_analysis['mean_interval']:.1f} ms")
            print(f"Variability (CV): {bvp_analysis['interval_cv']:.2f}")
        else:
            print("Status: No valid BVP analysis available")
        
        print("\n" + "="*50)

    def process_video(self, folder_path, batch_size=32, stride=10):
        """
        Process a folder of images to estimate heart rate.
        
        Args:
            folder_path (str): Path to folder containing images
            batch_size (int): Batch size for model inference
            stride (int): Stride for overlapping clips
        """
        # Run inference to get predicted signals
        predicted_signals = self.run_inference(folder_path, batch_size, stride)
        
        # Calculate heart rate and BVP analysis
        hr_bpm, fft_vals, freqs, stats, bvp_analysis = self.hr_calculator.calculate_heart_rate(predicted_signals)
        
        # Visualize results
        self.visualizer.plot_results(predicted_signals, fft_vals, freqs, hr_bpm, stats, bvp_analysis)
        
        # Print summary
        self._print_summary(hr_bpm, bvp_analysis)
        
        return hr_bpm, bvp_analysis

if __name__ == "__main__":
    # 1) Setup
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "checkpoints/PURE_TSCAN.pth"
    frame_depth = 10
    batch_size = 4
    fps = 30
    stride = 1

    # 2) Initialize inference pipeline
    inference = TSCANInference(
        checkpoint_path=checkpoint_path,
        device=device,
        frame_depth=frame_depth,
        fps=fps
    )

    # 3) Process video
    folder_path = "data"
    inference.process_video(folder_path, batch_size, stride)

