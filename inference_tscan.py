import os
import glob
import torch
import numpy as np
import cv2
from models.tscan_model import TSCAN
from preprocessing import preprocess_frame  # Or copy the function here

def load_model(checkpoint_path, device='cuda:0', frame_depth=10, drop_rate=0.2):
    """
    Load a TSCAN model from a .pth file.
    """
    model = TSCAN(frame_depth=frame_depth, drop_rate=drop_rate)
    
    # Load the state dict and remove the 'module.' prefix if it exists
    state_dict = torch.load(checkpoint_path, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '')  # Remove 'module.' prefix
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model.eval()
    model.to(device)
    return model

def make_clips(frames, clip_length):
    """
    Group frames into consecutive clips of size `clip_length`.
    frames: List/array of shape (N, C, H, W)
    """
    clips = []
    for i in range(0, len(frames) - clip_length + 1, clip_length):
        clip = frames[i : i + clip_length]
        clips.append(clip)
    return np.array(clips)  # shape: (num_clips, clip_length, C, H, W)

def run_inference_on_folder(
    model, 
    folder_path="data", 
    frame_depth=10, 
    batch_size=4, 
    device='cuda:0'
):
    """
    Example function to run TSCAN inference on all images in a folder. 
    Returns a list of predicted rPPG signals (one per clip).
    """
    # 1) Gather all image paths
    image_paths = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))

    # 2) Preprocess frames
    preprocessed_frames = []
    for img_path in image_paths:
        img = cv2.imread(img_path)  # BGR
        if img is None:
            continue
        frame_tensor = preprocess_frame(img)  # (C,H,W)
        preprocessed_frames.append(frame_tensor)

    if len(preprocessed_frames) < frame_depth:
        print("Not enough frames to form a single clip.")
        return None

    # 3) Convert list to a numpy array (N, C, H, W)
    preprocessed_frames = np.stack(preprocessed_frames, axis=0)

    # 4) Split into clips of length frame_depth
    clips = make_clips(preprocessed_frames, frame_depth)
    print("Number of clips:", clips.shape[0])

    # 5) Inference
    all_ppg_signals = []
    with torch.no_grad():
        for start_idx in range(0, len(clips), batch_size):
            batch_clips = clips[start_idx : start_idx + batch_size]
            
            # Debug print to check dimensions
            print(f"Initial batch shape: {batch_clips.shape}")
            
            # Reorder dimensions and ensure correct size
            batch_clips = np.transpose(batch_clips, (0, 2, 1, 3, 4))
            batch_clips_torch = torch.from_numpy(batch_clips).float().to(device)
            
            # Debug print after transformation
            print(f"Transformed batch shape: {batch_clips_torch.shape}")
            
            # Forward pass
            ppg_pred = model(batch_clips_torch)
            print(f"Output shape: {ppg_pred.shape}")
            
            ppg_pred_np = ppg_pred.cpu().numpy()
            all_ppg_signals.append(ppg_pred_np)

    # Concatenate predictions across all batches => shape: (num_clips, frame_depth)
    if len(all_ppg_signals) > 0:
        all_ppg_signals = np.concatenate(all_ppg_signals, axis=0)
    else:
        all_ppg_signals = None

    return all_ppg_signals

if __name__ == "__main__":
    # 1) Setup
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "checkpoints/PURE_TSCAN.pth"  # Update if needed
    frame_depth = 10
    batch_size = 4

    # 2) Load Model
    model = load_model(checkpoint_path, device=device, frame_depth=frame_depth)

    # 3) Run inference on a folder of frames
    folder_path = "data"  # A folder containing .jpg frames
    predicted_signals = run_inference_on_folder(model, folder_path, frame_depth, batch_size, device)

    if predicted_signals is not None:
        print("Predicted rPPG signals shape:", predicted_signals.shape)
        # predicted_signals[i] is the rPPG signal for clip i
        # Each row in predicted_signals is a 1D signal across frame_depth frames.
    else:
        print("No predictions were made (not enough frames).")

    # 4) (Optional) Calculate heart rate using FFT or peak detection on the predicted signal.
    # For demonstration, let's show a simple average over frames:
    # The final "ppg" for each clip is an array of length = frame_depth
    # You might do something like:
    #   - Concatenate all clips or do a window-based approach
    #   - Perform FFT to estimate frequency peak
    #   - Convert that frequency to BPM

    # Example pseudo-code for a single clip's signal:
    # ppg_clip = predicted_signals[0]  # shape: (frame_depth,)
    # # do a 1D FFT to find main frequency
    # freqs = np.fft.rfftfreq(frame_depth, d=1.0/30)  # sampling rate = 30
    # fft_vals = np.abs(np.fft.rfft(ppg_clip))
    # peak_idx = np.argmax(fft_vals)
    # peak_freq = freqs[peak_idx]  # in Hz
    # hr_bpm = peak_freq * 60
    # print("Estimated HR (BPM) for first clip: ", hr_bpm)

