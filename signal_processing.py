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
        
        # BVP analysis parameters
        self.bvp_min_amplitude = 0.1  # Minimum acceptable BVP amplitude
        self.bvp_max_amplitude = 2.0  # Maximum acceptable BVP amplitude
        self.bvp_min_std = 0.05      # Minimum acceptable BVP standard deviation

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

    def analyze_bvp(self, ppg_signals):
        """
        Analyze Blood Volume Pulse (BVP) characteristics.
        
        Args:
            ppg_signals (np.ndarray): Raw PPG signals
            
        Returns:
            dict: Dictionary containing BVP analysis results
        """
        # Normalize and filter the signal
        ppg_normalized, _ = self.normalize_signal(ppg_signals)
        ppg_filtered = self.apply_bandpass_filter(ppg_normalized)
        
        # Print debugging information
        print("\nBVP Analysis Debug:")
        print(f"Signal length: {len(ppg_filtered)}")
        print(f"Signal range: [{np.min(ppg_filtered):.3f}, {np.max(ppg_filtered):.3f}]")
        
        # Find peaks in the filtered signal with more lenient parameters
        peaks, peak_properties = scipy_signal.find_peaks(
            ppg_filtered,
            height=0.1,  # Lower height threshold
            distance=self.fps//4,  # Allow closer peaks
            prominence=0.05  # Lower prominence threshold
        )
        
        print(f"Number of peaks detected: {len(peaks)}")
        
        if len(peaks) < 2:
            return {
                'valid': False,
                'message': f'Insufficient peaks detected for BVP analysis (found {len(peaks)} peaks)',
                'amplitude': 0,
                'std': 0,
                'intervals': [],
                'mean_interval': 0,
                'prominence': 0,
                'peak_quality': 0
            }
        
        # Calculate BVP characteristics
        peak_values = ppg_filtered[peaks]
        peak_intervals = np.diff(peaks) / self.fps * 1000  # Convert to milliseconds
        
        print(f"Peak values range: [{np.min(peak_values):.3f}, {np.max(peak_values):.3f}]")
        print(f"Interval range: [{np.min(peak_intervals):.1f}, {np.max(peak_intervals):.1f}] ms")
        
        # Calculate BVP amplitude (peak-to-peak)
        bvp_amplitude = np.mean(np.abs(np.diff(peak_values)))
        
        # Calculate BVP standard deviation
        bvp_std = np.std(peak_values)
        
        # Calculate mean interval between peaks
        mean_interval = np.mean(peak_intervals)
        
        # Calculate peak prominence (quality measure)
        peak_prominence = np.mean(peak_properties['prominences'])
        
        # Calculate peak quality score (0-1)
        peak_quality = min(1.0, peak_prominence / 0.1)  # Normalize by height threshold
        
        # Calculate interval variability (HRV-like measure)
        interval_std = np.std(peak_intervals)
        interval_cv = interval_std / mean_interval if mean_interval > 0 else 0
        
        # More lenient validation criteria
        is_valid = (
            bvp_amplitude >= 0.05 and  # Lower minimum amplitude
            bvp_std >= 0.02 and        # Lower minimum std
            peak_quality >= 0.3 and    # Lower quality threshold
            mean_interval >= 200 and   # Minimum interval (300 BPM max)
            mean_interval <= 2000      # Maximum interval (30 BPM min)
        )
        
        # Generate detailed status message
        if is_valid:
            status = "BVP characteristics within normal range"
        else:
            issues = []
            if bvp_amplitude < 0.05:
                issues.append("amplitude too low")
            if bvp_std < 0.02:
                issues.append("variability too low")
            if peak_quality < 0.3:
                issues.append("poor peak quality")
            if mean_interval < 200:
                issues.append("intervals too short")
            if mean_interval > 2000:
                issues.append("intervals too long")
            status = f"BVP characteristics outside normal range: {', '.join(issues)}"
        
        return {
            'valid': is_valid,
            'message': status,
            'amplitude': bvp_amplitude,
            'std': bvp_std,
            'intervals': peak_intervals,
            'mean_interval': mean_interval,
            'peak_values': peak_values,
            'peak_indices': peaks,
            'prominence': peak_prominence,
            'peak_quality': peak_quality,
            'interval_std': interval_std,
            'interval_cv': interval_cv
        }

    def calculate_heart_rate(self, ppg_signals):
        """
        Calculate heart rate from PPG signals.
        
        Args:
            ppg_signals (np.ndarray): Raw PPG signals
            
        Returns:
            tuple: (heart_rate_bpm, fft_values, frequencies, statistics, bvp_analysis)
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
            return 0, fft_vals, freqs, stats, None
        
        # Get the strongest valid peak
        peak_idx, hr_bpm, magnitude = max(valid_peaks, key=lambda x: x[2])
        
        print("\nAll detected peaks:")
        for peak_idx, bpm, magnitude in valid_peaks:
            print(f"Frequency: {bpm/60:.2f} Hz, BPM: {bpm:.1f}, Magnitude: {magnitude:.2f}")
        
        # Perform BVP analysis
        bvp_analysis = self.analyze_bvp(ppg_signals)
        
        # Print BVP analysis results
        print("\nBVP Analysis:")
        print(f"Status: {bvp_analysis['message']}")
        print(f"Amplitude: {bvp_analysis['amplitude']:.3f}")
        print(f"Standard Deviation: {bvp_analysis['std']:.3f}")
        print(f"Mean Interval: {bvp_analysis['mean_interval']:.1f} ms")
        
        return hr_bpm, fft_vals, freqs, stats, bvp_analysis 