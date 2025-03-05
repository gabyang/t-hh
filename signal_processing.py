import numpy as np
from scipy import signal as scipy_signal

class HeartRateCalculator:
    def __init__(self, fps=30, min_bpm=45, max_bpm=180, peak_height_threshold=0.2):
        """
        Initialize heart rate calculator with configurable parameters.
        
        Args:
            fps (int): Frames per second of the video
            min_bpm (int): Minimum valid heart rate in BPM
            max_bpm (int): Maximum valid heart rate in BPM
            peak_height_threshold (float): Minimum height threshold for peak detection
        """
        self.fps = fps
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.peak_height_threshold = peak_height_threshold
        self.nyquist = fps / 2
        
        # Calculate frequency bounds for bandpass filter
        self.low_freq = min_bpm / 60 / self.nyquist
        self.high_freq = max_bpm / 60 / self.nyquist

    def normalize_signal(self, ppg_signals):
        """
        Normalize the PPG signal using z-score normalization.
        
        Args:
            ppg_signals (np.ndarray): Raw PPG signals
            
        Returns:
            tuple: (normalized_signal, statistics_dict)
        """
        ppg_signals = ppg_signals.flatten()
        ppg_normalized = (ppg_signals - np.mean(ppg_signals)) / (np.std(ppg_signals) + 1e-6)
        
        stats = {
            'mean': np.mean(ppg_normalized),
            'std': np.std(ppg_normalized),
            'min': np.min(ppg_normalized),
            'max': np.max(ppg_normalized)
        }
        
        return ppg_normalized, stats

    def apply_bandpass_filter(self, signal):
        """
        Apply bandpass filter to the signal.
        
        Args:
            signal (np.ndarray): Input signal
            
        Returns:
            np.ndarray: Filtered signal
        """
        b, a = scipy_signal.butter(3, [self.low_freq, self.high_freq], btype='band')
        return scipy_signal.filtfilt(b, a, signal)

    def apply_window(self, signal):
        """
        Apply Hanning window to the signal.
        
        Args:
            signal (np.ndarray): Input signal
            
        Returns:
            np.ndarray: Windowed signal
        """
        window = np.hanning(len(signal))
        return signal * window

    def compute_fft(self, signal):
        """
        Compute FFT of the signal.
        
        Args:
            signal (np.ndarray): Input signal
            
        Returns:
            tuple: (fft_values, frequencies)
        """
        fft_vals = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(len(signal), d=1.0/self.fps)
        return fft_vals, freqs

    def find_peaks(self, fft_vals, freqs):
        """
        Find peaks in the frequency spectrum.
        
        Args:
            fft_vals (np.ndarray): FFT values
            freqs (np.ndarray): Frequencies
            
        Returns:
            list: List of valid peaks in BPM
        """
        # Normalize FFT values
        fft_vals = fft_vals / np.max(fft_vals)
        
        # Find peaks
        peaks, _ = scipy_signal.find_peaks(
            fft_vals,
            height=self.peak_height_threshold,
            distance=self.fps//2
        )
        
        # Convert peaks to BPM and filter by physiological range
        valid_peaks = []
        for peak in peaks:
            bpm = freqs[peak] * 60
            if self.min_bpm <= bpm <= self.max_bpm:
                valid_peaks.append((peak, bpm, fft_vals[peak]))
        
        return valid_peaks

    def calculate_heart_rate(self, ppg_signals):
        """
        Calculate heart rate from PPG signals.
        
        Args:
            ppg_signals (np.ndarray): Raw PPG signals
            
        Returns:
            tuple: (heart_rate_bpm, fft_values, frequencies, statistics)
        """
        # Normalize signal
        ppg_normalized, stats = self.normalize_signal(ppg_signals)
        
        # Print normalized signal statistics
        print("\nNormalized Signal Statistics:")
        print(f"Mean: {stats['mean']:.6f}")
        print(f"Std: {stats['std']:.6f}")
        print(f"Min: {stats['min']:.6f}")
        print(f"Max: {stats['max']:.6f}")
        
        # Apply signal processing pipeline
        ppg_filtered = self.apply_bandpass_filter(ppg_normalized)
        ppg_windowed = self.apply_window(ppg_filtered)
        fft_vals, freqs = self.compute_fft(ppg_windowed)
        
        # Find valid peaks
        valid_peaks = self.find_peaks(fft_vals, freqs)
        
        if not valid_peaks:
            print(f"\nWarning: No peaks found in physiological range ({self.min_bpm}-{self.max_bpm} BPM)")
            return 0, fft_vals, freqs, stats
        
        # Get the strongest valid peak
        peak_idx, hr_bpm, magnitude = max(valid_peaks, key=lambda x: x[2])
        
        print("\nAll detected peaks:")
        for peak_idx, bpm, magnitude in valid_peaks:
            print(f"Frequency: {bpm/60:.2f} Hz, BPM: {bpm:.1f}, Magnitude: {magnitude:.2f}")
        
        return hr_bpm, fft_vals, freqs, stats 