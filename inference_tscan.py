import os
import glob
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from models.tscan_model import TSCAN
from preprocessing import preprocess_frame  # Or copy the function here
from scipy import signal

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

def make_clips(frames, clip_length, stride=1):
    """
    Group frames into consecutive clips with overlap
    frames: List/array of shape (N, C, H, W)
    stride: Number of frames to shift window (1 = maximum overlap)
    """
    clips = []
    for i in range(0, len(frames) - clip_length + 1, stride):
        clip = frames[i : i + clip_length]
        clips.append(clip)
    return np.array(clips)

def calculate_heart_rate(ppg_signals, fps=30):
    """
    Calculate heart rate from PPG signals with robust normalization.
    """
    # Normalize the signal robustly
    ppg_signals = ppg_signals.flatten()
    
    # Z-score normalization
    ppg_normalized = (ppg_signals - np.mean(ppg_signals)) / (np.std(ppg_signals) + 1e-6)
    
    print("\nNormalized Signal Statistics:")
    print(f"Mean: {np.mean(ppg_normalized):.6f}")
    print(f"Std: {np.std(ppg_normalized):.6f}")
    print(f"Min: {np.min(ppg_normalized):.6f}")
    print(f"Max: {np.max(ppg_normalized):.6f}")
    
    # Apply bandpass filter
    from scipy import signal
    nyquist = fps / 2
    low = 0.75 / nyquist   # 45 BPM
    high = 3.0 / nyquist   # 180 BPM
    b, a = signal.butter(3, [low, high], btype='band')
    ppg_filtered = signal.filtfilt(b, a, ppg_normalized)
    
    # Apply Hanning window
    window = np.hanning(len(ppg_filtered))
    ppg_windowed = ppg_filtered * window
    
    # Compute FFT
    fft_vals = np.abs(np.fft.rfft(ppg_windowed))
    freqs = np.fft.rfftfreq(len(ppg_windowed), d=1.0/fps)
    
    # Normalize FFT values
    fft_vals = fft_vals / np.max(fft_vals)
    
    # Find peaks with lower threshold and minimum distance
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(fft_vals, 
                                 height=0.2,          # Minimum height threshold
                                 distance=fps//2)     # Minimum samples between peaks
    
    print("\nAll detected peaks:")
    for peak_idx in peaks:
        freq = freqs[peak_idx]
        bpm = freq * 60
        magnitude = fft_vals[peak_idx]
        print(f"Frequency: {freq:.2f} Hz, BPM: {bpm:.1f}, Magnitude: {magnitude:.2f}")
    
    # Filter peaks to physiological range
    valid_peaks = []
    for peak in peaks:
        bpm = freqs[peak] * 60
        if 45 <= bpm <= 180:
            valid_peaks.append(peak)
    
    if len(valid_peaks) == 0:
        print("\nWarning: No peaks found in physiological range (45-180 BPM)")
        return 0, fft_vals, freqs
    
    # Get the strongest valid peak
    peak_idx = valid_peaks[np.argmax(fft_vals[valid_peaks])]
    peak_freq = freqs[peak_idx]
    hr_bpm = peak_freq * 60
    
    return hr_bpm, fft_vals, freqs

def plot_results(ppg_signals, fft_vals, freqs, hr_bpm):
    """
    Enhanced visualization of the signals and spectrum.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    
    # Plot raw PPG signals
    time = np.arange(len(ppg_signals)) / 30.0
    ax1.plot(time, ppg_signals.flatten(), label='Raw Signal')
    ax1.set_title('Raw rPPG Signals')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)
    
    # Plot filtered signal
    nyquist = 30 / 2
    low = 0.75 / nyquist
    high = 3.0 / nyquist
    b, a = signal.butter(3, [low, high], btype='band')
    ppg_filtered = signal.filtfilt(b, a, ppg_signals.flatten())
    ax2.plot(time, ppg_filtered, label='Filtered Signal', color='orange')
    ax2.set_title('Filtered rPPG Signal (0.75-3 Hz)')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True)
    
    # Plot frequency spectrum
    freqs_bpm = freqs * 60
    ax3.plot(freqs_bpm, fft_vals)
    if hr_bpm > 0:
        ax3.axvline(x=hr_bpm, color='r', linestyle='--', 
                    label=f'Peak: {hr_bpm:.1f} BPM')
    
    # Add physiological range markers
    ax3.axvspan(45, 180, color='g', alpha=0.1, label='Normal HR Range')
    ax3.set_title('Frequency Spectrum')
    ax3.set_xlabel('Heart Rate (BPM)')
    ax3.set_ylabel('Magnitude')
    ax3.set_xlim(0, 200)
    ax3.grid(True)
    ax3.legend()
    
    plt.tight_layout()
    plt.show()

def run_inference_on_folder(
    model, 
    folder_path="data", 
    frame_depth=10, 
    batch_size=4, 
    device='cuda:0',
    stride=1  # Add stride parameter
):
    """
    Run TSCAN inference on all images in a folder with overlapping windows.
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
        frame_tensor = preprocess_frame(img)
        preprocessed_frames.append(frame_tensor)

    if len(preprocessed_frames) < frame_depth:
        print("Not enough frames to form a single clip.")
        return None

    # 3) Convert list to numpy array
    preprocessed_frames = np.stack(preprocessed_frames, axis=0)

    # 4) Split into overlapping clips
    clips = make_clips(preprocessed_frames, frame_depth, stride=stride)
    print(f"Number of clips (with overlap): {clips.shape[0]}")

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
    checkpoint_path = "checkpoints/PURE_TSCAN.pth"
    frame_depth = 10
    batch_size = 4
    fps = 30
    stride = 1  # Process with maximum overlap

    # 2) Load Model
    model = load_model(checkpoint_path, device=device, frame_depth=frame_depth)

    # 3) Run inference with overlapping windows
    folder_path = "data"
    predicted_signals = run_inference_on_folder(
        model, folder_path, frame_depth, batch_size, device, stride=stride
    )

    if predicted_signals is not None:
        print("Predicted rPPG signals shape:", predicted_signals.shape)
        
        # 4) Calculate heart rate
        hr_bpm, fft_vals, freqs = calculate_heart_rate(predicted_signals, fps=fps)
        print(f"\nEstimated Heart Rate: {hr_bpm:.1f} BPM")
        
        # 5) Plot results
        plot_results(predicted_signals, fft_vals, freqs, hr_bpm)
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

