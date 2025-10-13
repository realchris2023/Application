import sys
import os
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import soundfile as sf
import sounddevice as sd
from tkinter import Tk, Frame, StringVar, OptionMenu, Button, Entry, Label
from components.play_button import PlayButton
from audio.vbap2d import calculate_gains as calculate_gains_2d

from gui.plot import plot_audio_channels, plot_speaker_and_source_positions
from gui.speakers import SpeakerLayout
from gui.ui.pan_canvas import PanCanvas
from gui.ui.speaker_editor import SpeakerEditor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class AudioPanningApp:
    def __init__(self, master):
        self.master = master
        self.frame = Frame(master)
        self.frame.pack()

        # Default geometry references used across the app
        self.default_speaker_x_distance = 250.0
        self.default_speaker_y_distance = 216.5
        self.default_speaker_z_distance = 0.0

        # Speaker layout and virtual source state
        self.layout = SpeakerLayout()
        initial_positions = [
            (-self.default_speaker_x_distance / 2, self.default_speaker_y_distance, self.default_speaker_z_distance),
            (self.default_speaker_x_distance / 2, self.default_speaker_y_distance, self.default_speaker_z_distance),
        ]
        self.layout.set_positions(initial_positions)
        self.channel_count = len(self.layout.positions)
        self.channel_gains = np.zeros(self.channel_count, dtype="float32")
        self.virtual_source_position = np.array(
            [0.0, self.default_speaker_y_distance, self.default_speaker_z_distance],
            dtype=float,
        )

        # Transport controls
        controls_bar = Frame(self.frame)
        controls_bar.pack(anchor="w", pady=4)
        self.play_button = PlayButton(controls_bar, self.play_audio)
        self.play_button.pack(side="left", padx=5, pady=5)
        Button(controls_bar, text="Stop", command=self.stop_audio).pack(side="left", padx=5, pady=5)

        # Virtual source canvas (top-down view with height)
        self.pan_canvas = PanCanvas(
            self.frame,
            move_callback=self.set_virtual_position,
            initial_position=self.virtual_source_position,
        )
        self.pan_canvas.pack(anchor="w", pady=8, padx=5)

        # Speaker configuration editor
        self.speaker_editor = SpeakerEditor(
            self.frame,
            layout=self.layout,
            apply_callback=self._handle_speaker_positions,
            default_distances=(
                self.default_speaker_x_distance,
                self.default_speaker_y_distance,
                max(self.default_speaker_z_distance, 200.0),
            ),
        )
        self.speaker_editor.pack(anchor="w", pady=10)
        self._refresh_pan_canvas()

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

        self.stream = None # Initialize audio stream
        self.set_virtual_position()  # Initialize VBAP gains based on default virtual source position
        self.playback_index = 0 # Initialize playback index
        
        # Experiment pan positions Menu
        self.experiment_pan_positions = self.calculate_experiment_pan_positions()
        self.selected_experiment_pan_position = StringVar()
        default_index = 8 if len(self.experiment_pan_positions) > 8 else 0
        self.selected_experiment_pan_position.set(self.experiment_pan_positions[default_index])
        self.experiment_pan_positions_menu = OptionMenu(self.frame, self.selected_experiment_pan_position, *self.experiment_pan_positions)
        self.experiment_pan_positions_menu.pack()
        self._refresh_experiment_menu()

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

    # ------------------------------------------------------------------
    # Speaker layout management
    # ------------------------------------------------------------------
    def _handle_speaker_positions(self, positions: Iterable[Sequence[float]]):
        """Callback used by the speaker editor whenever positions are applied."""
        self.layout.set_positions(positions)
        self.channel_count = len(self.layout.positions)
        self.channel_gains = self._match_gain_vector(self.channel_count)
        print(f"[SpeakerConfig] Updated speaker positions: {self.layout.positions}")

        self._refresh_pan_canvas()
        self._clamp_virtual_source_to_bounds()
        self.set_virtual_position()
        self._refresh_experiment_menu()

    def _refresh_pan_canvas(self):
        defaults = (
            self.default_speaker_x_distance,
            self.default_speaker_y_distance,
            max(self.default_speaker_z_distance, 200.0),
        )
        self.pan_canvas.configure_scene(self.layout.active_speakers(), defaults)

    def _refresh_experiment_menu(self):
        self.experiment_pan_positions = self.calculate_experiment_pan_positions()
        if not self.experiment_pan_positions:
            self.experiment_pan_positions = [0.0]
        self.selected_experiment_pan_position.set(self.experiment_pan_positions[0])
        menu_widget = getattr(self, "experiment_pan_positions_menu", None)
        if menu_widget is None:
            return
        menu_widget["menu"].delete(0, "end")
        for value in self.experiment_pan_positions:
            menu_widget["menu"].add_command(
                label=value,
                command=lambda v=value: self.selected_experiment_pan_position.set(v),
            )

    def _active_speakers(self, dims: int | None = 3) -> List[np.ndarray]:
        active = self.layout.active_speakers()
        if dims is None or dims >= 3:
            return active
        return [speaker[:dims] for speaker in active]

    def _clamp_virtual_source_to_bounds(self):
        bounds = getattr(self.pan_canvas, "bounds", None)
        if bounds is None:
            return
        x = float(np.clip(self.virtual_source_position[0], bounds.x[0], bounds.x[1]))
        y = float(np.clip(self.virtual_source_position[1], bounds.y[0], bounds.y[1]))
        z = float(np.clip(self.virtual_source_position[2], bounds.z[0], bounds.z[1]))
        self.virtual_source_position = np.array([x, y, z], dtype=float)


    def calculate_experiment_pan_positions(self):
        """Calculate experiment pan positions based on the distance between speakers."""
        active = self._active_speakers(dims=2)
        if len(active) < 2:
            return [0.0]
        xs = [speaker[0] for speaker in active]
        distance = abs(max(xs) - min(xs))
        if distance == 0:
            return [0.0]
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
        self.set_virtual_position(x=pan_value)

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

    def set_virtual_position(self, x=None, y=None, z=None):
        """Update the virtual source position and recompute VBAP gains."""
        position = np.array(self.virtual_source_position, dtype=float)
        if x is not None:
            position[0] = float(x)
        if y is not None:
            position[1] = float(y)
        if z is not None:
            position[2] = float(z)
        self.virtual_source_position = position
        self._clamp_virtual_source_to_bounds()
        print(f"[VirtualSource] Requested update -> clamped position {self.virtual_source_position}")

        active_speakers = self._active_speakers()
        if not active_speakers:
            print("No active speakers configured.")
            self.channel_gains = np.zeros(0, dtype='float32')
            self.left_gain = 0.0
            self.right_gain = 0.0
            self.pan_canvas.update_virtual_position(self.virtual_source_position)
            return

        raw_gains = self._solve_channel_gains(self.virtual_source_position, active_speakers)
        gain_vector = self._match_gain_vector(self.channel_count)
        gain_vector.fill(0.0)
        limit = min(len(raw_gains), gain_vector.shape[0])
        if limit:
            gain_vector[:limit] = raw_gains[:limit]
        self.channel_gains = gain_vector
        self.left_gain = gain_vector[0] if gain_vector.shape[0] > 0 else 0.0
        self.right_gain = gain_vector[1] if gain_vector.shape[0] > 1 else 0.0
        gain_norm = float(np.dot(self.channel_gains, self.channel_gains))
        print(
            f"Position: {self.virtual_source_position}, "
            f"Gains: {np.array2string(self.channel_gains, precision=3)}, "
            f"Squared gain sum: {gain_norm:.4f}"
        )
        self.pan_canvas.update_virtual_position(self.virtual_source_position)

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
        plot_audio_channels(audio_chunk, self.channel_gains)
        
    def plot_scatter_positions(self):
        """Plot the speaker positions, listener position, and virtual source position."""
        plot_speaker_and_source_positions(
            self._active_speakers(dims=3),
            self.virtual_source_position,
        )

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
            active_speakers = self._active_speakers()
        active_count = len(active_speakers)
        if active_count < 2:
            return np.zeros(active_count, dtype='float32')
        if active_count == 2:
            pair_2d = [speaker[:2] for speaker in active_speakers]
            gains = calculate_gains_2d(pair_2d[0], pair_2d[1], virtual_source[:2])
            return gains.astype('float32')
        return self._solve_vbap_3d(np.asarray(active_speakers, dtype=float), np.asarray(virtual_source, dtype=float))

    def _solve_vbap_3d(self, speakers, virtual_source):
        """Select a speaker triplet and compute 3D VBAP gains."""
        from itertools import combinations

        virtual_norm = np.linalg.norm(virtual_source)
        if virtual_norm <= 1e-9:
            print("[VBAP3D] Virtual source too close to origin; returning mute gains.")
            return np.zeros(speakers.shape[0], dtype='float32')

        target_unit = virtual_source / virtual_norm
        candidates = []

        for combo in combinations(range(speakers.shape[0]), 3):
            basis_vectors = []
            skip_combo = False
            for idx in combo:
                vec = speakers[idx]
                norm = np.linalg.norm(vec)
                if norm <= 1e-9:
                    skip_combo = True
                    print(f"[VBAP3D] Speaker {idx} is at the listener; skipping combo {combo}.")
                    break
                basis_vectors.append(vec / norm)
            if skip_combo:
                continue

            spread_matrix = np.column_stack(basis_vectors)
            try:
                gains = np.linalg.solve(spread_matrix, target_unit)
            except np.linalg.LinAlgError:
                print(f"[VBAP3D] Singular loudspeaker matrix for combo {combo}.")
                continue

            if not np.all(np.isfinite(gains)):
                print(f"[VBAP3D] Non-finite gains for combo {combo}: {gains}")
                continue

            gain_norm = np.linalg.norm(gains)
            if gain_norm <= 1e-12:
                print(f"[VBAP3D] Degenerate gain vector for combo {combo}.")
                continue

            normalized_gains = gains / gain_norm
            negative_contribution = np.sum(np.clip(-normalized_gains, 0.0, None))
            negative_count = int(np.sum(normalized_gains < -1e-6))
            residual = np.linalg.norm(spread_matrix @ normalized_gains - target_unit)

            candidates.append(
                (
                    negative_count,
                    negative_contribution,
                    residual,
                    combo,
                    normalized_gains,
                )
            )
            print(
                f"[VBAP3D] Combo {combo} residual {residual:.4f}, "
                f"neg_count {negative_count}, neg_sum {negative_contribution:.4f}, "
                f"gains {normalized_gains}"
            )

        gains_vector = np.zeros(speakers.shape[0], dtype='float32')
        if not candidates:
            print("[VBAP3D] No valid loudspeaker triple found; returning silence.")
            return gains_vector

        # Prefer in-hull solutions (no negative gains), then least negative contribution, then best residual.
        candidates.sort(key=lambda item: (item[0] > 0, item[1], item[2]))
        negative_count, negative_contribution, residual, combo, best_gains = candidates[0]
        print(
            f"[VBAP3D] Selected combo {combo} "
            f"(neg_count={negative_count}, neg_sum={negative_contribution:.4f}, residual={residual:.4f})."
        )

        for slot, gain in zip(combo, best_gains):
            gains_vector[slot] = gain
        return gains_vector
