import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
from scipy.sparse import spdiags

class SignalVisualizer:
    def __init__(self, fps=30):
        """
        Initialize signal visualizer.
        
        Args:
            fps (int): Frames per second of the video
        """
        self.fps = fps
        self.nyquist = fps / 2

    def _detrend(self, input_signal, lambda_value=100):
        """
        Detrend the input signal using a smoothing filter.
        
        Args:
            input_signal (np.ndarray): Input signal
            lambda_value (float): Smoothing parameter
            
        Returns:
            np.ndarray: Detrended signal
        """
        signal_length = input_signal.shape[0]
        H = np.identity(signal_length)
        ones = np.ones(signal_length)
        minus_twos = -2 * np.ones(signal_length)
        diags_data = np.array([ones, minus_twos, ones])
        diags_index = np.array([0, 1, 2])
        D = spdiags(diags_data, diags_index, signal_length - 2, signal_length).toarray()

        detrended_signal = np.dot(
            (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal
        )
        return detrended_signal

    def _next_power_of_2(self, x):
        """Calculate the nearest power of 2."""
        return 1 if x == 0 else 2 ** (x - 1).bit_length()

    def plot_results(self, ppg_signals, fft_vals, freqs, hr_bpm, stats, bvp_analysis=None):
        """
        Create enhanced visualization of the signals and spectrum.
        
        Args:
            ppg_signals (np.ndarray): Raw PPG signals
            fft_vals (np.ndarray): FFT values
            freqs (np.ndarray): Frequencies
            hr_bpm (float): Estimated heart rate in BPM
            stats (dict): Signal statistics
            bvp_analysis (dict, optional): BVP analysis results
        """
        if bvp_analysis is None:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        else:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))
        
        # Plot raw PPG signals
        time = np.arange(len(ppg_signals)) / self.fps
        ppg_signals_flat = ppg_signals.flatten()
        
        # Detrend the signal
        detrended_signal = self._detrend(ppg_signals_flat)
        ax1.plot(time, detrended_signal, label='Detrended Signal', color='blue')
        ax1.plot(time, ppg_signals_flat, label='Raw Signal', color='gray', alpha=0.5)
        ax1.set_title('Raw and Detrended rPPG Signals')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True)
        ax1.legend()
        
        # Plot filtered signal
        low = 0.75 / self.nyquist
        high = 3.0 / self.nyquist
        b, a = scipy_signal.butter(3, [low, high], btype='band')
        ppg_filtered = scipy_signal.filtfilt(b, a, detrended_signal)
        ax2.plot(time, ppg_filtered, label='Filtered Signal', color='orange')
        ax2.set_title('Filtered rPPG Signal (0.75-3 Hz)')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Amplitude')
        ax2.grid(True)
        ax2.legend()
        
        # Plot frequency spectrum with improved resolution
        N = self._next_power_of_2(len(ppg_signals_flat))
        f, pxx = scipy_signal.periodogram(ppg_filtered, fs=self.fps, nfft=N, detrend=False)
        freqs_bpm = f * 60
        
        ax3.plot(freqs_bpm, pxx / pxx.max(), label='Power Spectrum')
        if hr_bpm > 0:
            ax3.axvline(x=hr_bpm, color='r', linestyle='--', 
                        label=f'Peak: {hr_bpm:.1f} BPM')
        
        # Add physiological range markers
        ax3.axvspan(45, 180, color='g', alpha=0.1, label='Normal HR Range')
        ax3.set_title('Frequency Spectrum')
        ax3.set_xlabel('Heart Rate (BPM)')
        ax3.set_ylabel('Normalized Power')
        ax3.set_xlim(0, 200)
        ax3.grid(True)
        ax3.legend()
        
        # Plot BVP analysis if available
        if bvp_analysis is not None and bvp_analysis['valid']:
            peak_times = bvp_analysis['peak_indices'] / self.fps
            peak_values = bvp_analysis['peak_values']
            
            # Plot BVP peaks and intervals
            ax4.plot(time, ppg_filtered, label='Filtered Signal', color='orange', alpha=0.5)
            ax4.scatter(peak_times, peak_values, color='red', label='Detected Peaks')
            
            # Add vertical lines for peak intervals
            for peak_time in peak_times:
                ax4.axvline(x=peak_time, color='r', alpha=0.2, linestyle='--')
            
            # Add BVP statistics
            stats_text = (
                f"BVP Analysis:\n"
                f"Amplitude: {bvp_analysis['amplitude']:.3f}\n"
                f"Std Dev: {bvp_analysis['std']:.3f}\n"
                f"Mean Interval: {bvp_analysis['mean_interval']:.1f} ms\n"
                f"Peak Quality: {bvp_analysis['peak_quality']:.2f}\n"
                f"Interval CV: {bvp_analysis['interval_cv']:.2f}"
            )
            ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes,
                    verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
            
            # Add quality indicator
            quality_color = 'green' if bvp_analysis['valid'] else 'red'
            ax4.text(0.98, 0.98, f"Quality: {'Good' if bvp_analysis['valid'] else 'Poor'}",
                    transform=ax4.transAxes, horizontalalignment='right',
                    verticalalignment='top', color=quality_color, weight='bold')
            
            ax4.set_title('BVP Analysis')
            ax4.set_xlabel('Time (seconds)')
            ax4.set_ylabel('Amplitude')
            ax4.grid(True)
            ax4.legend()
        
        # Add signal statistics
        stats_text = (
            f"Signal Statistics:\n"
            f"Mean: {stats['mean']:.6f}\n"
            f"Std: {stats['std']:.6f}\n"
            f"Min: {stats['min']:.6f}\n"
            f"Max: {stats['max']:.6f}"
        )
        fig.text(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show() 