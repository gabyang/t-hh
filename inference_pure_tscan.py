import os
import torch
import numpy as np
import cv2
from typing import Optional, Tuple, Dict, List
from models.pure_tscan_model import PURE_TSCAN
from preprocessing import FramePreprocessor
from signal_processing import HeartRateCalculator
from visualization import SignalVisualizer

class PURE_TSCANInference:
    """
    Inference class for PURE TSCAN model.
    """
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the inference pipeline.
        
        Args:
            device: Device to run inference on ('cuda', 'mps', or 'cpu'). If None, will auto-detect.
        """
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        self.device = device
        print(f"Using device: {self.device}")
        
        self.model = self._load_model()
        self.frame_preprocessor = FramePreprocessor()
        self.heart_rate_calculator = HeartRateCalculator()
        self.signal_visualizer = SignalVisualizer()
        
    def _load_model(self) -> torch.nn.Module:
        """
        Load the PURE TSCAN model.
        
        Returns:
            Loaded PyTorch model
        """
        model = PURE_TSCAN()
        checkpoint_path = "checkpoints/PURE_TSCAN.pth"
            
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        # Load checkpoint with appropriate device mapping
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Remove 'module.' prefix if it exists
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model
        
    def _make_clips(self, frames: List[np.ndarray], clip_length: int = 10) -> torch.Tensor:
        """
        Create overlapping clips from frames.
        
        Args:
            frames: List of preprocessed frames
            clip_length: Number of frames per clip
            
        Returns:
            Tensor of shape (num_clips, channels, frames, height, width)
        """
        if len(frames) < clip_length:
            raise ValueError(f"Insufficient frames: {len(frames)} < {clip_length}")
            
        # Stack frames into a single tensor
        frames_tensor = np.stack(frames)
        
        # Create clips
        clips = []
        for i in range(len(frames) - clip_length + 1):
            clip = frames_tensor[i:i + clip_length]
            # Take only the first 3 channels and keep the time dimension
            clip = clip[:, :3]  # Shape: (clip_length, 3, 16, 16)
            # Transpose to match model's expected shape (B, C, T, H, W)
            clip = np.transpose(clip, (1, 0, 2, 3))  # Shape: (3, clip_length, 16, 16)
            clips.append(clip)
            
        # Stack clips and add batch dimension
        clips = np.stack(clips)  # Shape: (num_clips, 3, clip_length, 16, 16)
        return torch.FloatTensor(clips).to(self.device)
        
    def run_inference(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Run inference on a list of frames.
        
        Args:
            frames: List of preprocessed frames
            
        Returns:
            Predicted PPG signals
        """
        with torch.no_grad():
            clips = self._make_clips(frames)
            predictions = self.model(clips)
            return predictions.cpu().numpy()
            
    def process_video(self, frames: List[np.ndarray]) -> Tuple[float, Dict]:
        """
        Process a video sequence to estimate heart rate and analyze BVP.
        
        Args:
            frames: List of preprocessed frames
            
        Returns:
            Tuple of (estimated_hr_bpm, bvp_analysis)
        """
        # Run inference to get PPG signals
        ppg_signals = self.run_inference(frames)
        
        # Calculate heart rate and analyze BVP
        hr_bpm, fft_vals, freqs, stats, bvp_analysis = self.heart_rate_calculator.calculate_heart_rate(ppg_signals)
        
        # Visualize results
        self.signal_visualizer.plot_results(
            ppg_signals=ppg_signals,
            fft_vals=fft_vals,
            freqs=freqs,
            hr_bpm=hr_bpm,
            stats=stats,
            bvp_analysis=bvp_analysis
        )
        
        return hr_bpm, bvp_analysis
        
    def _print_summary(self, hr_bpm: float, bvp_analysis: Dict):
        """
        Print a summary of the analysis results.
        
        Args:
            hr_bpm: Estimated heart rate in BPM
            bvp_analysis: Dictionary containing BVP analysis results or None if analysis failed
        """
        print("\nAnalysis Summary:")
        print(f"Model Type: PURE")
        print(f"Estimated Heart Rate: {hr_bpm:.1f} BPM")
        print("\nBVP Analysis:")
        
        if bvp_analysis is None:
            print("Status: Analysis failed - No valid peaks detected in physiological range")
            print("Amplitude: N/A")
            print("Peak Quality: N/A")
            print("Mean Interval: N/A")
            print("Variability (CV): N/A")
        else:
            print(f"Status: {bvp_analysis['message']}")
            if bvp_analysis['valid']:
                print(f"Amplitude: {bvp_analysis['amplitude']:.3f}")
                print(f"Peak Quality: {bvp_analysis['peak_quality']:.3f}")
                print(f"Mean Interval: {bvp_analysis['mean_interval']:.1f} ms")
                print(f"Variability (CV): {bvp_analysis['interval_cv']:.2%}")

def main():
    # Initialize inference
    inference = PURE_TSCANInference()
    
    # Load and preprocess frames
    frames = []
    frame_dir = "data"  # Directory containing the frames
    
    if not os.path.exists(frame_dir):
        print(f"Error: Directory '{frame_dir}' not found!")
        return
        
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
    
    if not frame_files:
        print(f"Error: No JPG files found in '{frame_dir}'!")
        return
    
    print(f"Loading {len(frame_files)} frames...")
    for i, frame_file in enumerate(frame_files, 1):
        frame_path = os.path.join(frame_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is not None:
            # Preprocess the frame
            processed_frame = inference.frame_preprocessor.preprocess_frame(frame)
            if processed_frame is not None:
                frames.append(processed_frame)
        if i % 10 == 0:  # Progress update every 10 frames
            print(f"Processed {i}/{len(frame_files)} frames...")
    
    if not frames:
        print("No valid frames found!")
        return
        
    print(f"Successfully loaded {len(frames)} frames")
    
    # Process video and get results
    hr_bpm, bvp_analysis = inference.process_video(frames)
    
    # Print summary
    inference._print_summary(hr_bpm, bvp_analysis)

if __name__ == "__main__":
    main() 