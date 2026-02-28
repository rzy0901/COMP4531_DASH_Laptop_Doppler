#!/usr/bin/env python3
"""
Signal Processing Functions for Doppler Velocity Measurement
=============================================================

This file contains the core signal processing algorithms for Doppler velocity
measurement. Students can modify these functions to experiment with different
signal processing approaches.

Functions:
    - analyze_single_tone: Process single-tone (CW) Doppler signals
"""

import numpy as np
from scipy.fft import fft, fftfreq
from dataclasses import dataclass
from typing import Optional, Tuple
import time


# ============== Physical Constants ==============
SPEED_OF_SOUND = 343.0  # m/s at 20°C


@dataclass
class DopplerResult:
    """Result of Doppler analysis."""
    frequency_shift: float  # Hz
    velocity: float  # m/s
    signal_strength: float  # dB
    timestamp: float


def analyze_single_tone(
    audio_data: np.ndarray,
    sample_rate: int,
    center_freq: float
) -> Tuple[Optional[DopplerResult], Optional[Tuple[np.ndarray, np.ndarray]]]:
    """
    Analyze single-tone (CW) signal for Doppler shift.

    Method: FFT around the expected frequency, find peak offset.
    The Doppler shift is calculated from the frequency difference between
    the detected peak and the expected center frequency.

    Args:
        audio_data: Input audio samples (1D numpy array)
        sample_rate: Audio sample rate in Hz
        center_freq: Expected center frequency in Hz

    Returns:
        Tuple of (DopplerResult or None, spectrum_data or None)
        spectrum_data is (frequencies, magnitudes) for display
    """
    n = len(audio_data)

    # Apply Hanning window to reduce spectral leakage
    window = np.hanning(n)
    windowed = audio_data * window

    # Compute FFT
    spectrum = fft(windowed)
    freqs = fftfreq(n, 1 / sample_rate)

    # Only consider positive frequencies
    positive_mask = freqs >= 0
    freqs = freqs[positive_mask]
    spectrum = np.abs(spectrum[positive_mask])

    # Store spectrum data for display
    spectrum_data = (freqs, spectrum)

    # Search around expected frequency (±500 Hz window)
    search_window = 500  # Hz
    freq_mask = (freqs >= center_freq - search_window) & \
               (freqs <= center_freq + search_window)

    if not np.any(freq_mask):
        return None, spectrum_data

    search_freqs = freqs[freq_mask]
    search_spectrum = spectrum[freq_mask]

    # Find the peak in the search window
    peak_idx = np.argmax(search_spectrum)
    peak_freq = search_freqs[peak_idx]
    peak_power = search_spectrum[peak_idx]

    # Parabolic interpolation for sub-bin frequency accuracy
    if 0 < peak_idx < len(search_spectrum) - 1:
        alpha = search_spectrum[peak_idx - 1]
        beta = search_spectrum[peak_idx]
        gamma = search_spectrum[peak_idx + 1]
        if 2 * beta - alpha - gamma != 0:
            p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
            freq_resolution = search_freqs[1] - search_freqs[0]
            peak_freq = peak_freq + p * freq_resolution

    # Calculate Doppler shift and velocity
    freq_shift = peak_freq - center_freq

    # Doppler formula for sound waves:
    # f_received = f_source * (c + v_receiver) / (c - v_source)
    # For small velocities (v << c): delta_f / f ≈ v / c
    # Therefore: v = delta_f * c / f
    velocity = freq_shift * SPEED_OF_SOUND / center_freq

    # Signal strength in dB
    if peak_power > 0:
        signal_strength = 20 * np.log10(peak_power + 1e-10)
    else:
        signal_strength = -100

    result = DopplerResult(
        frequency_shift=freq_shift,
        velocity=velocity,
        signal_strength=signal_strength,
        timestamp=time.time()
    )

    return result, spectrum_data
