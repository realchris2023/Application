import sys
import os
import numpy as np
import soundfile as sf
import sounddevice as sd
from tkinter import Tk, Frame, Scale, HORIZONTAL, StringVar, OptionMenu, Button, Entry, Label
from components.play_button import PlayButton
from audio.vbap2d import calculate_gains as calculate_gains_2d

from gui.plot import plot_audio_channels, plot_speaker_and_source_positions

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class AudioPanningApp:
    def __init__(self, master):
        self.pan_range_scale_limit = 10
        
        self.master = master  # Set the master widget
        self.frame = Frame(master)  # Create a new frame widget
        self.frame.pack()  # Pack the frame widget
        self.default_speaker_x_distance = 250.0  # Default speaker distance in cm (horizontal span)
        self.default_speaker_y_distance = 216.5  # Default speaker distance in cm (forward)
        self.default_speaker_z_distance = 0.0    # Default height in cm

        # Speaker management
        self.speakers = []
        self.active_speaker_indices = []
        self.channel_gains = np.zeros(0, dtype='float32')
        self.set_speaker_positions(
            [
                (-self.default_speaker_x_distance / 2, self.default_speaker_y_distance, self.default_speaker_z_distance),
                (self.default_speaker_x_distance / 2, self.default_speaker_y_distance, self.default_speaker_z_distance),
            ],
            update_pan_widget=False,
        )

        # Virtual source starts centered in front of the listener
        self.virtual_source_position = self._ensure_vector3(
            (0.0, self.default_speaker_y_distance, self.default_speaker_z_distance)
        )

        self.play_button = PlayButton(self.frame, self.play_audio)  # Create a play button and assign the play_audio method
        self.play_button.pack(side='left', padx=5, pady=5)  # Pack the play button

        self.stop_button = Button(self.frame, text="Stop", command=self.stop_audio)  # Create a stop button and assign the stop_audio method
        self.stop_button.pack(side='left', padx=5, pady=5)  # Pack the stop button

        pan_min, pan_max = self._calculate_pan_limits()
        self.pan_knob = Scale(
            self.frame,
            from_=pan_min,
            to=pan_max,
            orient=HORIZONTAL,
            resolution=25,
            command=self.update_pan,
            length=400,
            sliderlength=30,
        )
        
        self.pan_knob.set(0.0)  # Initialize at center
        self.pan_knob.pack() # Pack the pan-slider widget

        self.audio_directory = os.path.join(os.path.dirname(__file__), "..", "audio/audio_files") # Define the audio directory
        self._check_audio_directory() # Check if the audio directory exists
        self.audio_files = self._get_audio_files() # Get audio files in the directory
        self.selected_audio = StringVar() # Variable to store selected audio file
        # Only try to populate and load an audio file if we actually found any
        if self.audio_files:
            self.selected_audio.set(self.audio_files[0])
            self.audio_menu = OptionMenu(self.frame, self.selected_audio, *self.audio_files, command=self.load_audio_file)
            self.audio_file = os.path.join(self.audio_directory, self.selected_audio.get())
            try:
                self.load_audio_file(self.audio_file)
            except Exception as e:
                print(f"Failed to load audio {self.audio_file}: {e}")
                # fallback to a tiny silent buffer so UI still works
                self.audio_samples = np.zeros(1, dtype='float32')
                self.sample_rate = 44100
            self.audio_menu.pack()
        else:
            # No audio files found — disable the menu and provide a silent buffer
            self.selected_audio.set("")
            self.audio_menu = OptionMenu(self.frame, self.selected_audio, "")
            try:
                self.audio_menu.configure(state='disabled')
            except Exception:
                pass
            self.audio_samples = np.zeros(1, dtype='float32')
            self.sample_rate = 44100
            self.audio_menu.pack()

        self.create_speaker_input_fields() # Create input fields for speaker positions

        self.stream = None # Initialize audio stream
        self.update_pan(self.pan_knob.get()) # Update pan position from slider value
        self.playback_index = 0 # Initialize playback index
        
        # Experiment pan positions Menu
        self.experiment_pan_positions = self.calculate_experiment_pan_positions()
        self.selected_experiment_pan_position = StringVar()
        self.selected_experiment_pan_position.set(self.experiment_pan_positions[8])
        self.experiment_pan_positions_menu = OptionMenu(self.frame, self.selected_experiment_pan_position, *self.experiment_pan_positions)
        self.experiment_pan_positions_menu.pack()

        self.update_experiment_pan_button = Button(self.frame, text="Update Pan Position", command=self.update_experiment_pan_positions)
        self.update_experiment_pan_button.pack()
        
        # Plot waveforms Button
        self.plot_button = Button(self.frame, text="Plot Waveform", command=self.plot_current_audio) # Create a plot button
        self.plot_button.pack() # Pack the plot button
        
        # Scatter plot of speaker and source positions Button
        self.scatter_plot_button = Button(self.frame, text="Plot Speaker and Source Positions", command=self.plot_scatter_positions) # Create a scatter plot button
        self.scatter_plot_button.pack() # Pack the scatter plot button

        # Save audio file with postfix
        Label(self.frame, text="Save Audio - Postfix:").pack() # Create a label widget
        self.save_postfix_entry = Entry(self.frame) # Create an entry widget for the postfix
        self.save_postfix_entry.pack() # Pack the entry widget
        self.save_button = Button(self.frame, text="Save Audio", command=self.save_audio) # Create a save button
        self.save_button.pack() # Pack the save button

    def create_speaker_input_fields(self):
        """Create input fields for user to enter speaker coordinates."""
        Label(self.frame, text="Horizontal distance between speakers(cm):").pack() # Create a label widget 
        self.speakers_x_entry = Entry(self.frame) # Create an entry widget for speakers summed x-coordinate
        self.speakers_x_entry.pack() # Pack the entry widget
        self.speakers_x_entry.insert(0, "250")

        Label(self.frame, text="Distance to wall (cm):").pack() # Create a label widget for speaker y-coordinate
        self.speakers_y_entry = Entry(self.frame) # Create an entry widget for speaker y-coordinate
        self.speakers_y_entry.pack() # Pack the entry widget
        self.speakers_y_entry.insert(0, "216.5") # Set default value

        update_button = Button(self.frame, text="Update Speaker Positions", command=self.update_speaker_positions) # Create a button to update speaker positions
        update_button.pack() # Pack the button

    def update_speaker_positions(self):
        """Update speaker positions based on user input."""
        try:
            speakers_x = float(self.speakers_x_entry.get())  # Convert input to float
            left_x = float(np.negative(speakers_x / 2))  # Negative value for left
            right_x = speakers_x / 2  # Positive value for right
            y = float(self.speakers_y_entry.get())  # Y-coordinate is the same for both speakers

            new_positions = [
                (left_x, y, self.default_speaker_z_distance),  # Left speaker
                (right_x, y, self.default_speaker_z_distance),  # Right speaker
            ]
            self.set_speaker_positions(new_positions)
            self.virtual_source_position = self._ensure_vector3((0.0, y, self.virtual_source_position[2]))

            print(f"Updated speaker positions: {self._get_active_speakers()}")  # Debug print
            print(f"Updated Virtual source position: {self.virtual_source_position}")  # Debug print

            # Update experiment pan positions
            self.experiment_pan_positions = self.calculate_experiment_pan_positions()
            self.selected_experiment_pan_position.set(self.experiment_pan_positions[0])
            self.experiment_pan_positions_menu['menu'].delete(0, 'end')
            for position in self.experiment_pan_positions:
                self.experiment_pan_positions_menu['menu'].add_command(label=position, command=lambda value=position: self.selected_experiment_pan_position.set(value))

        except ValueError:
            print("Invalid input for speaker coordinates. Please enter numeric values.")  # Error message

    def calculate_experiment_pan_positions(self):
        """Calculate experiment pan positions based on the distance between speakers."""
        active = self._get_active_speakers(dims=2)
        if len(active) < 2:
            return [0.0]
        distance = abs(active[-1][0] - active[0][0])
        return [
            -1.5 * distance,
            -1.4 * distance,
            -1.3 * distance,
            -1.2 * distance,
            -1.1 * distance,
            -distance,
            -0.9 * distance,
            -0.8 * distance,
            -0.7 * distance,
            -0.6 * distance,
            -0.5 * distance, # LEFT
            -0.4 * distance,
            -0.3 * distance,
            -0.2 * distance,
            -0.1 * distance,
            0,                  #Center
            0.1 * distance,  
            0.2 * distance,  
            0.3 * distance,  
            0.4 * distance,  
            0.5 * distance, # Right
            0.6 * distance,
            0.7 * distance,
            0.8 * distance,
            0.9 * distance,
            distance,
            1.1 * distance,
            1.2 * distance,
            1.3 * distance,
            1.4 * distance,
            1.5 * distance,
        ]

    def update_experiment_pan_positions(self):
        """Update the pan position based on the selected experiment pan position."""
        pan_value = float(self.selected_experiment_pan_position.get())
        self.pan_knob.set(pan_value)
        self.update_pan(pan_value)

    def _check_audio_directory(self):
        if not os.path.exists(self.audio_directory): # Check if the audio directory exists
            print(f"Audio directory not found: {self.audio_directory}") # Print error message
        else:
            print(f"Audio directory found: {self.audio_directory}") # Print success message

    def _get_audio_files(self): 
        return [f for f in os.listdir(self.audio_directory) if f.endswith(".wav")] # Get audio files in the directory

    def load_audio_file(self, filename):
        # Accept either a filename or a full path
        if os.path.isabs(filename):
            path = filename
        else:
            path = os.path.join(self.audio_directory, filename)

        self.audio_file = path
        try:
            self.audio_samples, self.sample_rate = sf.read(self.audio_file, dtype='float32') # Read the audio file
        except Exception as e:
            # don't crash the app if a single file is broken; fall back to silence
            print(f"Error reading audio file '{self.audio_file}': {e}")
            self.audio_samples = np.zeros(1, dtype='float32')
            self.sample_rate = 44100
            return

        if hasattr(self.audio_samples, 'ndim') and self.audio_samples.ndim == 2:  # Check if 2 dimensional audio
            self.audio_samples = np.mean(self.audio_samples, axis=1)  # Average the stereo channels to convert to mono

    def update_pan(self, value): 
        pan_value = float(value)
        # Create vector from pan value and y-coordinate
        active_speakers = self._get_active_speakers()
        if not active_speakers:
            print("No active speakers configured.")
            return
        y_reference = active_speakers[0][1]
        self.virtual_source_position = self.get_virtual_source_position(pan_value, y_reference)
        if len(active_speakers) < 2:
            print("Need at least two speakers to calculate gains.")
            self.channel_gains = self._match_gain_vector(self.channel_count)
            self.channel_gains.fill(0.0)
            self.left_gain = 0.0
            self.right_gain = 0.0
            return
        raw_gains = self._solve_channel_gains(self.virtual_source_position, active_speakers)
        gain_vector = self._match_gain_vector(self.channel_count)
        gain_vector.fill(0.0)
        limit = min(len(raw_gains), gain_vector.shape[0])
        if limit:
            gain_vector[:limit] = raw_gains[:limit]
        self.channel_gains = gain_vector
        self.left_gain = gain_vector[0] if gain_vector.shape[0] > 0 else 0.0  # Left speaker gain
        self.right_gain = gain_vector[1] if gain_vector.shape[0] > 1 else 0.0  # Right speaker gain
        gain_norm = float(np.dot(self.channel_gains, self.channel_gains))
        print(f"Pan: {pan_value:.2f}, Position: {self.virtual_source_position}, Gains L/R: {self.left_gain:.2f}/{self.right_gain:.2f}")
        print(f"Sum of squared gain factors: {gain_norm:.4f}") # Debug print proving that gain factors are constant
    
    def get_virtual_source_position(self, pan_value, y_distance, z_distance=None):
        """Get the virtual source position (as a vector) from the pan value."""
        if z_distance is None:
            z_distance = self.virtual_source_position[2] if hasattr(self, "virtual_source_position") else 0.0
        return np.array([pan_value, y_distance, z_distance], dtype=float)

    def play_audio(self):
        """Start the audio playback stream."""
        # close any existing stream cleanly
        self._safe_close_stream()

        # ensure we have audio loaded
        if not hasattr(self, 'sample_rate') or not hasattr(self, 'audio_samples'):
            print("No audio loaded to play.")
            return
        channels = getattr(self, "channel_count", 0)
        if channels < 1:
            print("No speakers configured for playback.")
            return
        self._match_gain_vector(channels)
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=channels,
            dtype='float32',
            callback=self.audio_callback
        )
        self.stream.start()

    def save_audio(self):
        """Save the audio to a file."""
        postfix = int(self.virtual_source_position[0])  # Get the pan value as the postfix and cast to int
        if postfix or postfix == 0:
            save_filename = os.path.splitext(self.audio_file)[0] + f"_{postfix}.wav"
            gains = self._match_gain_vector(self.channel_count)
            if gains.size == 0:
                print("No speaker gains available; cannot save multichannel audio.")
                return
            multichannel_audio = np.multiply.outer(self.audio_samples, gains).astype('float32', copy=False)
            sf.write(save_filename, multichannel_audio, self.sample_rate)
            print(f"Audio saved as {save_filename}")
        else:
            print("Please enter a postfix to save the audio file.")
    
    def stop_audio(self):
        """Stop the audio playback stream and reset the playback index."""
        self._safe_close_stream()
        self.playback_index = 0  # Reset playback index
        
    def audio_callback(self, outdata, frames, time, status):
        """Process audio with VBAP gains."""
        if status:
            print(f"Stream error: {status}")
            return
        if self.playback_index >= len(self.audio_samples): # Reset playback if end is reached
            self.playback_index = 0 
            outdata.fill(0)
            return
        # Get audio chunk
        end_idx = min(self.playback_index + frames, len(self.audio_samples))
        chunk = self.audio_samples[self.playback_index:end_idx]
        # Pad if needed
        if len(chunk) < frames:
            chunk = np.pad(chunk, (0, frames - len(chunk)), 'constant')
        # Apply VBAP gains
        gains = self._match_gain_vector(outdata.shape[1])
        if gains.size == 0:
            outdata.fill(0)
            return
        outdata[:] = chunk[:, np.newaxis] * gains[np.newaxis, :]
        self.playback_index += frames

    def _safe_close_stream(self):
        """Try to stop and close the output stream without raising exceptions."""
        if getattr(self, 'stream', None) is not None:
            try:
                self.stream.stop()
            except Exception as e:
                print("Warning: error stopping stream:", e)
            try:
                self.stream.close()
            except Exception as e:
                print("Warning: error closing stream:", e)
            finally:
                self.stream = None
        
    def plot_current_audio(self):
        """Capture and plot current audio state."""
        # Get current chunk of audio
        start_idx = self.playback_index
        chunk_size = min(1000, len(self.audio_samples) - start_idx)
        audio_chunk = self.audio_samples[start_idx:start_idx + chunk_size]
        # Plot using the visualization function
        plot_audio_channels(
            audio_chunk,
            self.left_gain,
            self.right_gain,
            float(self.pan_knob.get())
        )
        
    def plot_scatter_positions(self):
        """Plot the speaker positions, listener position, and virtual source position."""
        plot_speaker_and_source_positions(
            self._get_active_speakers(dims=2),
            self.virtual_source_position[:2],
            self.pan_range_scale_limit,
        )

    def set_speaker_positions(self, positions, update_pan_widget=True):
        """Register loudspeaker positions in 3D coordinates."""
        processed = [self._ensure_vector3(pos) for pos in positions]
        self.speakers = processed
        if not self.active_speaker_indices:
            self.active_speaker_indices = list(range(len(self.speakers)))
        else:
            self.active_speaker_indices = [idx for idx in self.active_speaker_indices if idx < len(self.speakers)]
            if not self.active_speaker_indices:
                self.active_speaker_indices = list(range(len(self.speakers)))
        self.channel_count = len(self.active_speaker_indices)
        self.channel_gains = self._match_gain_vector(self.channel_count)
        if update_pan_widget:
            self._update_pan_widget_limits()

    def _ensure_vector3(self, vector):
        """Guarantee a speaker vector is 3D, padding z=0 if needed."""
        arr = np.asarray(vector, dtype=float)
        if arr.shape[0] == 2:
            arr = np.append(arr, 0.0)
        elif arr.shape[0] != 3:
            raise ValueError("Speaker positions must have 2 or 3 components.")
        return arr

    def _get_active_speakers(self, dims=3):
        """Return currently active speaker vectors."""
        active = [self.speakers[idx] for idx in self.active_speaker_indices]
        if dims is None or dims >= 3:
            return active
        return [speaker[:dims] for speaker in active]

    def _calculate_pan_limits(self):
        """Determine slider limits from active speaker spread."""
        active = self._get_active_speakers()
        if len(active) < 2:
            return (-self.pan_range_scale_limit, self.pan_range_scale_limit)
        xs = [speaker[0] for speaker in active]
        return (
            self.pan_range_scale_limit * min(xs),
            self.pan_range_scale_limit * max(xs),
        )

    def _update_pan_widget_limits(self):
        """Sync the pan slider range with current speakers."""
        if getattr(self, "pan_knob", None) is None:
            return
        pan_min, pan_max = self._calculate_pan_limits()
        self.pan_knob.config(from_=pan_min, to=pan_max)

    def _match_gain_vector(self, channel_count):
        """Ensure channel gains align with the expected channel count."""
        current = getattr(self, "channel_gains", np.zeros(0, dtype='float32'))
        if current.shape[0] == channel_count:
            if current.dtype != np.float32:
                current = current.astype('float32')
                self.channel_gains = current
            return current
        updated = np.zeros(channel_count, dtype='float32')
        length = min(current.shape[0], channel_count)
        if length:
            updated[:length] = current[:length]
        self.channel_gains = updated
        return self.channel_gains

    def _solve_channel_gains(self, virtual_source, active_speakers=None):
        """Compute channel gains using the current panning solver."""
        if active_speakers is None:
            active_speakers = self._get_active_speakers()
        active_count = len(active_speakers)
        if active_count < 2:
            return np.zeros(active_count, dtype='float32')
        if active_count == 2:
            pair_2d = [speaker[:2] for speaker in active_speakers]
            gains = calculate_gains_2d(pair_2d[0], pair_2d[1], virtual_source[:2])
            return gains.astype('float32')
        print("3D gain calculation not implemented yet; returning zeros.")
        return np.zeros(active_count, dtype='float32')
