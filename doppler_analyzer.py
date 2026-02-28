#!/usr/bin/env python3
"""
Doppler Velocity Analyzer
=========================
Real-time Doppler velocity measurement using microphone input.
Uses Single-tone signal analysis.

Usage:
    python doppler_analyzer.py

This will launch a web-based UI at http://localhost:8050

Note: Signal processing functions are in signal_processing.py
      Students can modify that file to experiment with different algorithms.
"""

import numpy as np
import threading
import queue
from collections import deque
from typing import Optional
from datetime import datetime

# Import signal processing functions from separate module
from signal_processing import (
    DopplerResult,
    analyze_single_tone,
)

# Audio processing
try:
    import sounddevice as sd
except ImportError:
    print("Please install sounddevice: pip install sounddevice")
    raise

# For saving audio files
from scipy.io import wavfile

# Web UI
try:
    from dash import Dash, html, dcc, callback, Output, Input, State
    import plotly.graph_objs as go
except ImportError:
    print("Please install dash and plotly: pip install dash plotly")
    raise


# ============== Default Parameters ==============
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_BLOCK_SIZE = 4096
DEFAULT_SINGLE_TONE_FREQ = 18000  # Hz


class DopplerAnalyzer:
    """Real-time Doppler velocity analyzer."""

    def __init__(
        self,
        center_freq: float = DEFAULT_SINGLE_TONE_FREQ,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        block_size: int = DEFAULT_BLOCK_SIZE
    ):
        self.center_freq = center_freq
        self.sample_rate = sample_rate
        self.block_size = block_size

        # Processing state
        self.running = False
        self.stream: Optional[sd.InputStream] = None
        self.audio_queue = queue.Queue(maxsize=100)
        self.result_queue = queue.Queue(maxsize=100)

        # History for display
        self.velocity_history = deque(maxlen=200)
        self.freq_shift_history = deque(maxlen=200)
        self.time_history = deque(maxlen=200)
        self.spectrum_data = None

        # Recording buffer (stores raw audio for saving)
        self.recording_buffer = []
        self.is_recording = False

        # Processing thread
        self.process_thread: Optional[threading.Thread] = None

    def start(self):
        """Start audio capture and analysis."""
        if self.running:
            return

        self.running = True

        # Clear history
        self.velocity_history.clear()
        self.freq_shift_history.clear()
        self.time_history.clear()

        # Start audio stream
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            blocksize=self.block_size,
            callback=self._audio_callback
        )
        self.stream.start()

        # Start processing thread
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()

        print("Started Doppler analyzer")

    def stop(self):
        """Stop audio capture and analysis."""
        self.running = False

        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        print("Stopped Doppler analyzer")

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream."""
        if status:
            print(f"Audio status: {status}")
        try:
            audio_data = indata.copy().flatten()
            self.audio_queue.put_nowait(audio_data)
            # Store audio if recording
            if self.is_recording:
                self.recording_buffer.append(audio_data.copy())
        except queue.Full:
            pass

    def start_recording(self):
        """Start recording audio to buffer."""
        self.recording_buffer = []
        self.is_recording = True
        print("Recording started...")

    def stop_recording(self, filename: str = None) -> str:
        """Stop recording and save to WAV file."""
        self.is_recording = False
        
        if not self.recording_buffer:
            print("No audio recorded")
            return None
        
        # Concatenate all audio chunks
        audio_data = np.concatenate(self.recording_buffer)
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recorded_doppler_{timestamp}.wav"
        
        # Normalize and save
        audio_normalized = audio_data / np.max(np.abs(audio_data) + 1e-10) * 0.9
        audio_16bit = (audio_normalized * 32767).astype(np.int16)
        wavfile.write(filename, self.sample_rate, audio_16bit)
        
        duration = len(audio_data) / self.sample_rate
        print(f"Saved recording: {filename} ({duration:.1f} seconds)")
        
        self.recording_buffer = []
        return filename

    def _process_loop(self):
        """Main processing loop running in separate thread."""
        while self.running:
            try:
                audio_data = self.audio_queue.get(timeout=0.1)
                result = self._analyze(audio_data)
                if result:
                    self.velocity_history.append(result.velocity)
                    self.freq_shift_history.append(result.frequency_shift)
                    self.time_history.append(result.timestamp)
                    try:
                        self.result_queue.put_nowait(result)
                    except queue.Full:
                        pass
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")

    def _analyze(self, audio_data: np.ndarray) -> Optional[DopplerResult]:
        """Analyze audio data for Doppler shift using functions from signal_processing.py."""
        result, spectrum_data = analyze_single_tone(
            audio_data, self.sample_rate, self.center_freq
        )

        # Store spectrum data for display
        if spectrum_data is not None:
            self.spectrum_data = spectrum_data

        return result

    def get_latest_result(self) -> Optional[DopplerResult]:
        """Get the most recent analysis result."""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None


# ============== Web UI ==============

# Global analyzer instance
analyzer = DopplerAnalyzer()

# Create Dash app
app = Dash(__name__)
app.title = "Doppler Velocity Analyzer"

app.layout = html.Div([
    html.Div([
        html.H1("Doppler Velocity Analyzer",
               style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
        html.P("Real-time velocity measurement using audio Doppler effect",
              style={'textAlign': 'center', 'color': '#7f8c8d', 'marginTop': '0'})
    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'}),

    # Control Panel
    html.Div([
        html.Div([
            html.H3("Signal Configuration", style={'color': '#2c3e50', 'marginTop': '0'}),

            # Single-tone parameters
            html.Div(id='single-params', children=[
                html.Label("Center Frequency (Hz):"),
                dcc.Input(id='center-freq', type='number', value=DEFAULT_SINGLE_TONE_FREQ,
                         style={'width': '100%', 'marginBottom': '10px', 'padding': '8px'})
            ]),

            html.Hr(),

            # Control buttons
            html.Button('Start', id='start-btn', n_clicks=0,
                       style={'backgroundColor': '#27ae60', 'color': 'white', 'border': 'none',
                             'padding': '15px 30px', 'fontSize': '16px', 'cursor': 'pointer',
                             'borderRadius': '5px', 'marginRight': '10px', 'width': '45%'}),
            html.Button('Stop', id='stop-btn', n_clicks=0,
                       style={'backgroundColor': '#e74c3c', 'color': 'white', 'border': 'none',
                             'padding': '15px 30px', 'fontSize': '16px', 'cursor': 'pointer',
                             'borderRadius': '5px', 'width': '45%'}),

            html.Div(id='status', children='Status: Stopped',
                    style={'marginTop': '15px', 'padding': '10px', 'backgroundColor': '#f8f9fa',
                          'borderRadius': '5px', 'textAlign': 'center'}),

            html.Hr(),

            # Recording controls
            html.H4("Recording (for Assignment)", style={'color': '#2c3e50', 'marginTop': '10px'}),
            html.Button('Start Recording', id='record-btn', n_clicks=0,
                       style={'backgroundColor': '#9b59b6', 'color': 'white', 'border': 'none',
                             'padding': '10px 20px', 'fontSize': '14px', 'cursor': 'pointer',
                             'borderRadius': '5px', 'width': '100%', 'marginBottom': '10px'}),
            html.Button('Stop & Save', id='save-btn', n_clicks=0,
                       style={'backgroundColor': '#8e44ad', 'color': 'white', 'border': 'none',
                             'padding': '10px 20px', 'fontSize': '14px', 'cursor': 'pointer',
                             'borderRadius': '5px', 'width': '100%'}),
            html.Div(id='record-status', children='Recording: Off',
                    style={'marginTop': '10px', 'padding': '10px', 'backgroundColor': '#f8f9fa',
                          'borderRadius': '5px', 'textAlign': 'center', 'fontSize': '12px'})
        ], style={'width': '25%', 'padding': '20px', 'backgroundColor': 'white',
                 'borderRadius': '10px', 'boxShadow': '0 2px 5px rgba(0,0,0,0.1)'}),

        # Main display
        html.Div([
            # Real-time velocity display
            html.Div([
                html.Div([
                    html.H2("Current Velocity", style={'color': '#7f8c8d', 'marginBottom': '5px', 'fontSize': '16px'}),
                    html.Div(id='velocity-display', children='0.00',
                            style={'fontSize': '72px', 'fontWeight': 'bold', 'color': '#2c3e50'}),
                    html.Span("m/s", style={'fontSize': '24px', 'color': '#7f8c8d'})
                ], style={'textAlign': 'center', 'padding': '20px'}),

                html.Div([
                    html.Div([
                        html.Span("Frequency Shift: ", style={'color': '#7f8c8d'}),
                        html.Span(id='freq-shift-display', children='0.0 Hz',
                                 style={'fontWeight': 'bold', 'color': '#3498db'})
                    ], style={'display': 'inline-block', 'marginRight': '30px'}),
                    html.Div([
                        html.Span("Signal Strength: ", style={'color': '#7f8c8d'}),
                        html.Span(id='signal-strength-display', children='-100 dB',
                                 style={'fontWeight': 'bold', 'color': '#e74c3c'})
                    ], style={'display': 'inline-block'})
                ], style={'textAlign': 'center', 'padding': '10px', 'borderTop': '1px solid #ecf0f1'})
            ], style={'backgroundColor': 'white', 'borderRadius': '10px',
                     'boxShadow': '0 2px 5px rgba(0,0,0,0.1)', 'marginBottom': '20px'}),

            # Graphs
            dcc.Graph(id='velocity-graph', style={'height': '300px'}),
            dcc.Graph(id='spectrum-graph', style={'height': '250px'})
        ], style={'width': '72%', 'marginLeft': '3%'})
    ], style={'display': 'flex'}),

    # Update interval
    dcc.Interval(id='update-interval', interval=100, n_intervals=0),

    # Instructions
    html.Div([
        html.H3("Instructions", style={'color': '#2c3e50'}),
        html.Ol([
            html.Li("Use generate_audio.py to create an audio file with matching parameters"),
            html.Li("Transfer the audio file to your phone"),
            html.Li("Configure the same frequency settings above"),
            html.Li("Click 'Start' to begin recording"),
            html.Li("Play the audio on your phone and move it towards/away from the computer"),
            html.Li("Positive velocity = approaching, Negative velocity = receding")
        ], style={'color': '#7f8c8d'})
    ], style={'marginTop': '30px', 'padding': '20px', 'backgroundColor': 'white',
             'borderRadius': '10px', 'boxShadow': '0 2px 5px rgba(0,0,0,0.1)'})

], style={'padding': '20px', 'backgroundColor': '#f5f6fa', 'minHeight': '100vh', 'fontFamily': 'Arial, sans-serif'})


@callback(
    Output('status', 'children'),
    [Input('start-btn', 'n_clicks'),
     Input('stop-btn', 'n_clicks')],
    State('center-freq', 'value')
)
def control_analyzer(start_clicks, stop_clicks, center_freq):
    global analyzer

    from dash import ctx
    triggered_id = ctx.triggered_id

    if triggered_id == 'start-btn' and start_clicks > 0:
        analyzer.stop()
        analyzer = DopplerAnalyzer(
            center_freq=center_freq or DEFAULT_SINGLE_TONE_FREQ
        )
        analyzer.start()
        return "Status: Running"

    elif triggered_id == 'stop-btn' and stop_clicks > 0:
        analyzer.stop()
        return "Status: Stopped"

    return "Status: Stopped"


@callback(
    Output('record-status', 'children'),
    [Input('record-btn', 'n_clicks'),
     Input('save-btn', 'n_clicks')]
)
def control_recording(record_clicks, save_clicks):
    global analyzer

    from dash import ctx
    triggered_id = ctx.triggered_id

    if triggered_id == 'record-btn' and record_clicks > 0:
        if analyzer.running:
            analyzer.start_recording()
            return "Recording: ON (move your phone!)"
        else:
            return "Error: Start analyzer first!"

    elif triggered_id == 'save-btn' and save_clicks > 0:
        if analyzer.is_recording:
            filename = analyzer.stop_recording()
            if filename:
                return f"Saved: {filename}"
            return "No audio recorded"
        return "Recording: Off"

    return "Recording: Off"


@callback(
    [Output('velocity-display', 'children'),
     Output('freq-shift-display', 'children'),
     Output('signal-strength-display', 'children'),
     Output('velocity-graph', 'figure'),
     Output('spectrum-graph', 'figure')],
    Input('update-interval', 'n_intervals')
)
def update_display(n):
    # Get data from analyzer
    velocity_data = list(analyzer.velocity_history)
    freq_shift_data = list(analyzer.freq_shift_history)
    time_data = list(analyzer.time_history)

    # Current values
    if velocity_data:
        current_velocity = velocity_data[-1]
        current_freq_shift = freq_shift_data[-1]
        velocity_text = f"{current_velocity:.2f}"
        freq_text = f"{current_freq_shift:.1f} Hz"
    else:
        velocity_text = "0.00"
        freq_text = "0.0 Hz"
        current_freq_shift = 0

    # Get latest result for signal strength
    result = analyzer.get_latest_result()
    if result:
        strength_text = f"{result.signal_strength:.1f} dB"
    else:
        strength_text = "-100 dB"

    # Velocity graph
    if time_data:
        # Convert to relative time
        t0 = time_data[0]
        rel_time = [t - t0 for t in time_data]

        velocity_fig = go.Figure()
        velocity_fig.add_trace(go.Scatter(
            x=rel_time,
            y=velocity_data,
            mode='lines',
            name='Velocity',
            line=dict(color='#3498db', width=2)
        ))
        velocity_fig.add_hline(y=0, line_dash="dash", line_color="gray")
        velocity_fig.update_layout(
            title="Velocity over Time",
            xaxis_title="Time (s)",
            yaxis_title="Velocity (m/s)",
            margin=dict(l=50, r=20, t=40, b=40),
            paper_bgcolor='white',
            plot_bgcolor='#f8f9fa',
            yaxis=dict(range=[-5, 5])  # Reasonable range for walking speed
        )
    else:
        velocity_fig = go.Figure()
        velocity_fig.update_layout(
            title="Velocity over Time",
            xaxis_title="Time (s)",
            yaxis_title="Velocity (m/s)",
            margin=dict(l=50, r=20, t=40, b=40),
            paper_bgcolor='white',
            plot_bgcolor='#f8f9fa'
        )

    # Spectrum graph
    if analyzer.spectrum_data:
        freqs, spectrum = analyzer.spectrum_data

        # Limit display range around center frequency
        display_mask = (freqs >= analyzer.center_freq - 500) & (freqs <= analyzer.center_freq + 500)

        display_freqs = freqs[display_mask]
        display_spectrum = spectrum[display_mask]

        # Convert to dB
        display_spectrum_db = 20 * np.log10(display_spectrum + 1e-10)

        spectrum_fig = go.Figure()
        spectrum_fig.add_trace(go.Scatter(
            x=display_freqs,
            y=display_spectrum_db,
            mode='lines',
            name='Spectrum',
            line=dict(color='#e74c3c', width=1),
            fill='tozeroy',
            fillcolor='rgba(231, 76, 60, 0.3)'
        ))

        # Mark center frequency
        spectrum_fig.add_vline(x=analyzer.center_freq, line_dash="dash",
                              line_color="green", annotation_text="Expected")

        spectrum_fig.update_layout(
            title="Frequency Spectrum",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Magnitude (dB)",
            margin=dict(l=50, r=20, t=40, b=40),
            paper_bgcolor='white',
            plot_bgcolor='#f8f9fa'
        )
    else:
        spectrum_fig = go.Figure()
        spectrum_fig.update_layout(
            title="Frequency Spectrum",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Magnitude (dB)",
            margin=dict(l=50, r=20, t=40, b=40),
            paper_bgcolor='white',
            plot_bgcolor='#f8f9fa'
        )

    return velocity_text, freq_text, strength_text, velocity_fig, spectrum_fig


if __name__ == "__main__":
    print("=" * 50)
    print("Doppler Velocity Analyzer")
    print("=" * 50)
    print("\nStarting web server...")
    print("Open http://localhost:8050 in your browser")
    print("\nPress Ctrl+C to stop\n")

    app.run(debug=False, host='0.0.0.0', port=8050)
