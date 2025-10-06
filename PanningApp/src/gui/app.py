import sys
import os
import numpy as np
import soundfile as sf
import sounddevice as sd
from tkinter import Tk, Frame, Scale, HORIZONTAL, StringVar, OptionMenu, Button, Entry, Label
from components.play_button import PlayButton
from audio.vbap import calculate_gains

from gui.plot import plot_audio_channels, plot_speaker_and_source_positions

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class AudioPanningApp:
    def __init__(self, master):
        self.pan_range_scale_limit = 10
        
        self.master = master  # Set the master widget
        self.frame = Frame(master)  # Create a new frame widget
        self.frame.pack()  # Pack the frame widget
        self.default_speaker_x_distance = 250  # Default speaker distance in cm (Horizontal)
        self.default_speaker_y_distance = 216.5  # Default speaker distance in cm (forward)
        # Define default speaker positions in 2D space (x, y) in centimeters - further assignable in application
        self.speaker_positions = [
            np.array([-125, 216.5]),  # Left speaker
            np.array([125, 216.5])    # Right speaker
        ]
        self.virtual_source_position = np.array([0, self.speaker_positions[0][1]])  # Default virtual source position

        self.play_button = PlayButton(self.frame, self.play_audio)  # Create a play button and assign the play_audio method
        self.play_button.pack(side='left', padx=5, pady=5)  # Pack the play button

        self.stop_button = Button(self.frame, text="Stop", command=self.stop_audio)  # Create a stop button and assign the stop_audio method
        self.stop_button.pack(side='left', padx=5, pady=5)  # Pack the stop button

        self.pan_knob = Scale(self.frame, from_= self.pan_range_scale_limit*(self.speaker_positions[0][0]), to= self.pan_range_scale_limit*self.speaker_positions[1][0], orient=HORIZONTAL, resolution=25, command=self.update_pan, length=400, sliderlength=30) # Create a larger pan-slider widget
        
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

            self.speaker_positions = [
                np.array([left_x, y]),  # Left speaker
                np.array([right_x, y])  # Right speaker
            ]
            self.pan_knob.config(from_= self.pan_range_scale_limit * (self.speaker_positions[0][0]), to=self.pan_range_scale_limit * (self.speaker_positions[1][0])) # Update pan range
            self.virtual_source_position = np.array([0, y])  # Update virtual source position

            print(f"Updated speaker positions: {self.speaker_positions}")  # Debug print
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
        distance = self.speaker_positions[1][0]*2
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
        self.virtual_source_position = self.get_virtual_source_position(pan_value, self.speaker_positions[0][1])
        # Calculate gains
        gains = calculate_gains(
            self.speaker_positions[0],
            self.speaker_positions[1],
            self.virtual_source_position
        )
        self.left_gain = gains[0] # Left speaker gain
        self.right_gain = gains[1] # Right speaker gain
        print(f"Pan: {pan_value:.2f}, Position: {self.virtual_source_position}, Gains L/R: {self.left_gain:.2f}/{self.right_gain:.2f}")
        print(f"Sum of squared gain factors: {np.pow(self.left_gain,2.) + np.pow(self.right_gain,2.):.4f}") # Debug print proving that gain factors are constant
    
    def get_virtual_source_position(self, pan_value, y_distance):
        """Get the virtual source position (as a vector) from the pan value."""
        x = pan_value # X-coordinate is the pan value
        return np.array([x, y_distance]) # Return the virtual source position as a vector

    def play_audio(self):
        """Start the audio playback stream."""
        # close any existing stream cleanly
        self._safe_close_stream()

        # ensure we have audio loaded
        if not hasattr(self, 'sample_rate') or not hasattr(self, 'audio_samples'):
            print("No audio loaded to play.")
            return
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=2, dtype='float32',
            callback=self.audio_callback
        )
        self.stream.start()

    def save_audio(self):
        """Save the audio to a file."""
        postfix = int(self.virtual_source_position[0])  # Get the pan value as the postfix and cast to int
        if postfix or postfix == 0:
            save_filename = os.path.splitext(self.audio_file)[0] + f"_{postfix}.wav"
            # Apply VBAP gains to the audio samples
            left_channel = self.audio_samples * self.left_gain
            right_channel = self.audio_samples * self.right_gain
            stereo_audio = np.column_stack((left_channel, right_channel))
            sf.write(save_filename, stereo_audio, self.sample_rate)
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
        outdata[:, 0] = chunk * self.left_gain   # Left channel
        outdata[:, 1] = chunk * self.right_gain  # Right channel
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
        plot_speaker_and_source_positions(self.speaker_positions, self.virtual_source_position, self.pan_range_scale_limit)

