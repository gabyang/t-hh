import cv2
import os
import argparse

def extract_frames(video_path, output_dir, fps=30):
    """
    Extract frames from a video file.
    
    Args:
        video_path (str): Path to the video file
        output_dir (str): Directory to save the extracted frames
        fps (int): Number of frames per second to extract (default: 30)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame interval (ensure it's at least 1)
    frame_interval = max(1, int(video_fps / fps))
    
    print(f"Video FPS: {video_fps}")
    print(f"Total frames in video: {total_frames}")
    print(f"Extracting every {frame_interval}th frame to achieve approximately {video_fps/frame_interval:.1f} FPS")
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Save frame at specified intervals
        if frame_count % frame_interval == 0:
            # Detect face in the frame
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Only save if a face is detected
            if len(faces) > 0:
                frame_path = os.path.join(output_dir, f'frame_{saved_count:03d}.jpg')
                cv2.imwrite(frame_path, frame)
                saved_count += 1
                
                # Print progress every 10 frames
                if saved_count % 10 == 0:
                    print(f"Saved {saved_count} frames...")
        
        frame_count += 1
        
        # Break if we have enough frames for TSCAN
        if saved_count >= 100:  # Save more than needed to ensure we have enough good quality frames
            break
    
    cap.release()
    print(f"\nExtraction complete! Saved {saved_count} frames to {output_dir}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract frames from a video file')
    parser.add_argument('video_path', type=str, help='Path to the video file')
    parser.add_argument('--output_dir', type=str, default='data', help='Directory to save frames (default: data)')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second to extract (default: 30)')
    
    args = parser.parse_args()
    extract_frames(args.video_path, args.output_dir, args.fps) 