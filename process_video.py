import os
import cv2
import shutil
import argparse
from typing import List, Optional, Tuple, Dict

from inference_service.inference_bp4d_tscan import BP4D_TSCANInference
from inference_service.inference_pure_tscan import PURE_TSCANInference
from inference_service.inference_bp4d_physnet import BP4D_PhysNetInference

class VideoProcessor:
    """
    A unified interface for processing videos using different inference services.
    Supports BP4D TSCAN, PURE TSCAN, and BP4D PhysNet models.
    """
    
    def __init__(self, model_type: str, device: Optional[str] = None):
        """
        Initialize the video processor with specified model type.
        
        Args:
            model_type: Type of model to use ('bp4d_tscan', 'pure_tscan', or 'bp4d_physnet')
            device: Device to run inference on ('cuda', 'mps', or 'cpu'). If None, will auto-detect.
        """
        self.model_type = model_type.lower()
        self.data_dir = "data"  # Default directory for frame storage
        
        # Initialize the appropriate inference service
        if self.model_type == 'bp4d_tscan':
            self.inference = BP4D_TSCANInference(device=device)
        elif self.model_type == 'pure_tscan':
            self.inference = PURE_TSCANInference(device=device)
        elif self.model_type == 'bp4d_physnet':
            self.inference = BP4D_PhysNetInference(device=device)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
    def extract_frames(self, video_path: str, fps: int = 30) -> None:
        """
        Extract frames from a video file and save them to the data directory.
        
        Args:
            video_path: Path to the video file
            fps: Target frames per second to extract
        """
        # Clear existing frames if any
        if os.path.exists(self.data_dir):
            shutil.rmtree(self.data_dir)
        os.makedirs(self.data_dir)
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")
        
        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, int(video_fps / fps))
        
        print(f"Video FPS: {video_fps}")
        print(f"Total frames in video: {total_frames}")
        print(f"Extracting every {frame_interval}th frame to achieve approximately {video_fps/frame_interval:.1f} FPS")
        
        frame_count = 0
        saved_count = 0
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save frame at specified intervals
            if frame_count % frame_interval == 0:
                # Detect face in the frame
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                # Only save if a face is detected
                if len(faces) > 0:
                    frame_path = os.path.join(self.data_dir, f'frame_{saved_count:03d}.jpg')
                    cv2.imwrite(frame_path, frame)
                    saved_count += 1
                    
                    if saved_count % 10 == 0:
                        print(f"Saved {saved_count} frames...")
            
            frame_count += 1
            
            # Break if we have enough frames
            if saved_count >= 100:
                break
        
        cap.release()
        print(f"\nExtraction complete! Saved {saved_count} frames to {self.data_dir}")
            
    def process_video_file(self, video_path: str, extract_frames: bool = True) -> Tuple[float, Dict]:
        """
        Process a video file and estimate heart rate.
        
        Args:
            video_path: Path to the video file
            extract_frames: Whether to extract frames from video (if False, uses existing frames)
            
        Returns:
            Tuple of (estimated_hr_bpm, bvp_analysis)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Extract frames if requested
        if extract_frames:
            print(f"Extracting frames from video: {video_path}")
            self.extract_frames(video_path)
            
        # Process the frames from data directory
        return self.process_frame_directory(self.data_dir)
        
    def process_frame_directory(self, frame_dir: str) -> Tuple[float, Dict]:
        """
        Process a directory of frame images.
        
        Args:
            frame_dir: Path to directory containing frame images
            
        Returns:
            Tuple of (estimated_hr_bpm, bvp_analysis)
        """
        if not os.path.exists(frame_dir):
            raise FileNotFoundError(f"Frame directory not found: {frame_dir}")
            
        # Get sorted list of frame files
        frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(('.jpg', '.png'))])
        if not frame_files:
            raise RuntimeError(f"No image files found in directory: {frame_dir}")
            
        frames = []
        print(f"Processing frames from directory: {frame_dir}")
        print(f"Using model: {self.model_type}")
        
        for i, frame_file in enumerate(frame_files, 1):
            frame_path = os.path.join(frame_dir, frame_file)
            frame = cv2.imread(frame_path)
            
            if frame is not None:
                processed_frame = self.inference.frame_preprocessor.preprocess_frame(frame)
                if processed_frame is not None:
                    frames.append(processed_frame)
                    
            if i % 10 == 0:
                print(f"Processed {i}/{len(frame_files)} frames...")
                
        if not frames:
            raise RuntimeError("No valid frames found in directory")
            
        print(f"Successfully processed {len(frames)} frames")
        
        # Run inference and get results
        hr_bpm, bvp_analysis = self.inference.process_video(frames)
        
        # Print summary
        self._print_summary(hr_bpm, bvp_analysis)
        
        return hr_bpm, bvp_analysis
        
    def _print_summary(self, hr_bpm: float, bvp_analysis: Dict):
        """
        Print a summary of the analysis results.
        
        Args:
            hr_bpm: Estimated heart rate in BPM
            bvp_analysis: Dictionary containing BVP analysis results
        """
        print("\nAnalysis Summary:")
        print(f"Model Type: {self.model_type}")
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
    parser = argparse.ArgumentParser(description='Process video for heart rate estimation')
    parser.add_argument('input_path', help='Path to video file or directory of frames')
    parser.add_argument('--model', choices=['bp4d_tscan', 'pure_tscan', 'bp4d_physnet'],
                      default='bp4d_physnet', help='Model type to use for inference')
    parser.add_argument('--device', choices=['cuda', 'mps', 'cpu'],
                      help='Device to run inference on (default: auto-detect)')
    parser.add_argument('--no-extract', action='store_true',
                      help='Skip frame extraction and use existing frames in data directory')
    parser.add_argument('--fps', type=int, default=30,
                      help='Target FPS for frame extraction (default: 30)')
    
    args = parser.parse_args()
    
    try:
        processor = VideoProcessor(model_type=args.model, device=args.device)
        
        if os.path.isfile(args.input_path):
            processor.process_video_file(args.input_path, extract_frames=not args.no_extract)
        elif os.path.isdir(args.input_path):
            processor.process_frame_directory(args.input_path)
        else:
            print(f"Error: Input path does not exist: {args.input_path}")
            
    except Exception as e:
        print(f"Error processing video: {str(e)}")

if __name__ == "__main__":
    main() 