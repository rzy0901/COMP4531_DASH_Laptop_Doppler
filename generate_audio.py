#!/usr/bin/env python3
"""
Doppler Experiment Audio Generator
==================================
Generates audio signals for Doppler velocity measurement experiments.
Uses Single-tone (CW) signals.

Usage:
    python generate_audio.py

This will launch a simple GUI to configure and generate audio files.
"""

import numpy as np
from scipy.io import wavfile
import os
import sys

# Default parameters
DEFAULT_SAMPLE_RATE = 44100  # Hz
DEFAULT_DURATION = 30  # seconds
DEFAULT_SINGLE_TONE_FREQ = 18000  # Hz (ultrasonic, less audible)


def generate_single_tone(
    frequency: float = DEFAULT_SINGLE_TONE_FREQ,
    duration: float = DEFAULT_DURATION,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    amplitude: float = 0.8
) -> np.ndarray:
    """
    Generate a single-tone (continuous wave) signal.

    Args:
        frequency: Signal frequency in Hz
        duration: Signal duration in seconds
        sample_rate: Sample rate in Hz
        amplitude: Signal amplitude (0-1)

    Returns:
        numpy array of audio samples
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = amplitude * np.sin(2 * np.pi * frequency * t)
    return signal


def save_wav(signal: np.ndarray, filename: str, sample_rate: int = DEFAULT_SAMPLE_RATE):
    """Save signal to WAV file."""
    # Normalize and convert to 16-bit PCM
    signal_normalized = signal / np.max(np.abs(signal)) * 0.9
    signal_16bit = (signal_normalized * 32767).astype(np.int16)
    wavfile.write(filename, sample_rate, signal_16bit)
    print(f"Saved: {filename}")


def run_gui():
    """Run the GUI for audio generation."""
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox, filedialog
    except ImportError:
        print("Tkinter not available. Running in command-line mode.")
        run_cli()
        return

    class AudioGeneratorApp:
        def __init__(self, root):
            self.root = root
            self.root.title("Doppler Experiment - Audio Generator")
            self.root.geometry("500x450")
            self.root.resizable(False, False)

            # Style
            style = ttk.Style()
            style.configure("Title.TLabel", font=("Helvetica", 16, "bold"))
            style.configure("Header.TLabel", font=("Helvetica", 12, "bold"))

            self.create_widgets()

        def create_widgets(self):
            # Title
            title = ttk.Label(self.root, text="Doppler Audio Generator", style="Title.TLabel")
            title.pack(pady=20)

            # Main frame
            main_frame = ttk.Frame(self.root, padding="20")
            main_frame.pack(fill="both", expand=True)

            # Common parameters
            common_frame = ttk.LabelFrame(main_frame, text="Parameters", padding="10")
            common_frame.pack(fill="x", pady=10)

            ttk.Label(common_frame, text="Duration (seconds):").grid(row=0, column=0, sticky="w", pady=5)
            self.duration_var = tk.StringVar(value=str(DEFAULT_DURATION))
            ttk.Entry(common_frame, textvariable=self.duration_var, width=15).grid(row=0, column=1, pady=5)

            ttk.Label(common_frame, text="Sample Rate (Hz):").grid(row=1, column=0, sticky="w", pady=5)
            self.sample_rate_var = tk.StringVar(value=str(DEFAULT_SAMPLE_RATE))
            ttk.Entry(common_frame, textvariable=self.sample_rate_var, width=15).grid(row=1, column=1, pady=5)

            # Single-tone parameters
            self.single_frame = ttk.LabelFrame(main_frame, text="Signal Parameters", padding="10")
            self.single_frame.pack(fill="x", pady=10)

            ttk.Label(self.single_frame, text="Frequency (Hz):").grid(row=0, column=0, sticky="w", pady=5)
            self.single_freq_var = tk.StringVar(value=str(DEFAULT_SINGLE_TONE_FREQ))
            ttk.Entry(self.single_frame, textvariable=self.single_freq_var, width=15).grid(row=0, column=1, pady=5)

            ttk.Label(self.single_frame, text="Recommended: 17000-19000 Hz",
                     font=("Helvetica", 9), foreground="gray").grid(row=1, column=0, columnspan=2, sticky="w")

            # Generate button
            btn_frame = ttk.Frame(main_frame)
            btn_frame.pack(fill="x", pady=20)

            generate_btn = ttk.Button(btn_frame, text="Generate Audio File",
                                      command=self.generate)
            generate_btn.pack(pady=10)

            # Status
            self.status_var = tk.StringVar(value="Ready")
            status_label = ttk.Label(main_frame, textvariable=self.status_var,
                                    foreground="green")
            status_label.pack(pady=10)

        def generate(self):
            """Generate and save the audio file."""
            try:
                duration = float(self.duration_var.get())
                sample_rate = int(self.sample_rate_var.get())

                freq = float(self.single_freq_var.get())
                signal = generate_single_tone(freq, duration, sample_rate)
                default_name = f"single_tone_{int(freq)}Hz.wav"

                # Ask for save location
                filename = filedialog.asksaveasfilename(
                    defaultextension=".wav",
                    filetypes=[("WAV files", "*.wav")],
                    initialfile=default_name
                )

                if filename:
                    save_wav(signal, filename, sample_rate)
                    self.status_var.set(f"Generated: {os.path.basename(filename)}")
                    messagebox.showinfo("Success", f"Audio file saved:\n{filename}")

            except ValueError as e:
                messagebox.showerror("Error", f"Invalid parameter: {e}")
            except Exception as e:
                messagebox.showerror("Error", f"Generation failed: {e}")

    root = tk.Tk()
    AudioGeneratorApp(root)
    root.mainloop()


def run_cli():
    """Run in command-line mode."""
    print("=" * 50)
    print("Doppler Experiment - Audio Generator")
    print("=" * 50)

    duration = float(input(f"Duration in seconds [{DEFAULT_DURATION}]: ") or DEFAULT_DURATION)
    sample_rate = int(input(f"Sample rate [{DEFAULT_SAMPLE_RATE}]: ") or DEFAULT_SAMPLE_RATE)
    freq = float(input(f"Frequency in Hz [{DEFAULT_SINGLE_TONE_FREQ}]: ") or DEFAULT_SINGLE_TONE_FREQ)

    signal = generate_single_tone(freq, duration, sample_rate)
    filename = f"single_tone_{int(freq)}Hz.wav"

    save_wav(signal, filename, sample_rate)
    print(f"\nDone! Transfer '{filename}' to your phone and play it.")


if __name__ == "__main__":
    # Check if running in a GUI environment
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        run_cli()
    else:
        run_gui()
