# Doppler Velocity Analyzer

Real-time velocity measurement using the acoustic Doppler effect. A single-tone (continuous wave) audio signal is played from a mobile device, and the frequency shift caused by movement is detected by a computer's microphone to estimate velocity.

## Project Structure

```
doppler_for_student/
├── doppler_analyzer.py     # Main application — real-time analyzer with web UI
├── signal_processing.py    # Core DSP algorithms (student-editable)
├── generate_audio.py       # Audio file generator (GUI / CLI)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Prerequisites

- **Python** 3.8+
- A working **microphone** connected to the computer
- A **phone** (or any device) capable of playing WAV files

## Installation

```bash
pip install -r requirements.txt
```

Dependencies installed:

| Package | Purpose |
|---------|---------|
| numpy | Numerical computing |
| scipy | FFT, WAV file I/O |
| sounddevice | Real-time microphone input |
| dash | Web-based UI framework |
| plotly | Interactive charts |

## Quick Start

### 1. Generate the Audio Signal

```bash
python generate_audio.py
```

This opens a Tkinter GUI where you can configure:

| Parameter | Default | Description |
|-----------|---------|-------------|
| Duration | 30 s | Length of the generated tone |
| Sample Rate | 44100 Hz | Audio sample rate |
| Frequency | 18000 Hz | Carrier frequency (recommended 17000–19000 Hz) |

Click **Generate Audio File** to save a `.wav` file, then transfer it to your phone.

> **CLI mode:** run `python generate_audio.py --cli` if no GUI environment is available.

### 2. Launch the Analyzer

```bash
python doppler_analyzer.py
```

Open **http://localhost:8050** in your browser. The web UI provides:

- **Center Frequency** input — must match the frequency used in step 1.
- **Start / Stop** buttons to control the microphone stream.
- **Current Velocity** readout in m/s.
- **Frequency Shift** and **Signal Strength** indicators.
- **Velocity vs. Time** plot updated in real time.
- **Frequency Spectrum** plot showing the region around the carrier.
- **Recording controls** to capture raw audio to a WAV file for later analysis.

### 3. Run the Experiment

1. Set the **Center Frequency** in the web UI to match the generated audio file.
2. Click **Start**.
3. Play the audio on your phone.
4. Move the phone **towards** the computer (positive velocity) or **away** (negative velocity).
5. Observe the velocity and spectrum plots updating in real time.
6. Click **Stop** when finished.

#### Recording Audio

While the analyzer is running:

1. Click **Start Recording**.
2. Perform your movement.
3. Click **Stop & Save** — the recording is saved as `recorded_doppler_<timestamp>.wav` in the working directory.

## Module Reference

### `signal_processing.py`

Contains the DSP algorithms that students can modify and extend.

#### Constants

- `SPEED_OF_SOUND` — 343.0 m/s (at 20 °C)

#### `DopplerResult` (dataclass)

| Field | Type | Description |
|-------|------|-------------|
| `frequency_shift` | `float` | Detected Doppler shift in Hz |
| `velocity` | `float` | Estimated velocity in m/s |
| `signal_strength` | `float` | Peak power in dB |
| `timestamp` | `float` | Unix timestamp of the measurement |

#### `analyze_single_tone(audio_data, sample_rate, center_freq)`

Performs FFT-based Doppler analysis on a single audio block.

**Algorithm outline:**

1. Apply a Hanning window to reduce spectral leakage.
2. Compute the FFT and retain positive frequencies.
3. Search for the spectral peak within ±500 Hz of the center frequency.
4. Refine the peak location using parabolic interpolation for sub-bin accuracy.
5. Compute the Doppler shift: `Δf = f_peak − f_center`.
6. Convert to velocity using the approximation `v = Δf · c / f_center` (valid when `v ≪ c`).

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `audio_data` | `np.ndarray` | 1-D array of audio samples |
| `sample_rate` | `int` | Sample rate in Hz |
| `center_freq` | `float` | Expected carrier frequency in Hz |

**Returns:** `(DopplerResult | None, spectrum_data | None)` where `spectrum_data` is a `(frequencies, magnitudes)` tuple for plotting.

---

### `doppler_analyzer.py`

Main application file. Contains the `DopplerAnalyzer` class and the Dash web UI.

#### `DopplerAnalyzer`

| Method | Description |
|--------|-------------|
| `__init__(center_freq, sample_rate, block_size)` | Create an analyzer. Defaults: 18000 Hz, 44100 Hz, 4096 samples. |
| `start()` | Open the microphone stream and begin processing. |
| `stop()` | Close the stream and stop processing. |
| `start_recording()` | Begin buffering raw audio for file export. |
| `stop_recording(filename=None)` | Save buffered audio to a WAV file and return the filename. |
| `get_latest_result()` | Return the most recent `DopplerResult` (non-blocking). |

The analyzer uses a producer–consumer pattern: the `sounddevice` callback enqueues audio blocks, and a background thread dequeues and processes them via `analyze_single_tone`.

#### Default Parameters

| Constant | Value | Description |
|----------|-------|-------------|
| `DEFAULT_SAMPLE_RATE` | 44100 | Audio sample rate (Hz) |
| `DEFAULT_BLOCK_SIZE` | 4096 | Samples per FFT frame |
| `DEFAULT_SINGLE_TONE_FREQ` | 18000 | Carrier frequency (Hz) |

---

### `generate_audio.py`

Utility for generating single-tone WAV files.

#### `generate_single_tone(frequency, duration, sample_rate, amplitude)`

Returns a NumPy array containing a pure sine wave at the specified frequency.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `frequency` | 18000 Hz | Tone frequency |
| `duration` | 30 s | Signal length |
| `sample_rate` | 44100 Hz | Sample rate |
| `amplitude` | 0.8 | Peak amplitude (0–1) |

#### `save_wav(signal, filename, sample_rate)`

Normalizes the signal to 90% full-scale, converts to 16-bit PCM, and writes a WAV file.

## Background Theory

The Doppler effect causes a frequency shift when a sound source moves relative to a receiver:

```
f_received = f_source × (c + v_receiver) / (c − v_source)
```

For small velocities (`v ≪ c`), this simplifies to:

```
Δf / f ≈ v / c   →   v = Δf × c / f
```

where:
- `Δf` is the observed frequency shift (Hz)
- `f` is the carrier (center) frequency (Hz)
- `c` is the speed of sound (≈ 343 m/s)
- `v` is the relative velocity (m/s), positive for approach

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No microphone input | Check OS audio settings; ensure `sounddevice` can see your device (`python -m sounddevice`) |
| Velocity reads ~0 with movement | Verify the center frequency in the UI matches the generated audio file |
| Very noisy readings | Move closer to the microphone; ensure a quiet environment |
| Port 8050 in use | Stop other Dash apps or change the port in `doppler_analyzer.py` |
| Tkinter not available | Use `python generate_audio.py --cli` for command-line mode |
