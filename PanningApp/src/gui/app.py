import sys
import os
import numpy as np
import soundfile as sf
import sounddevice as sd
from tkinter import Tk, Frame, Scale, HORIZONTAL, VERTICAL, StringVar, OptionMenu, Button, Entry, Label, Canvas
from components.play_button import PlayButton
from audio.vbap2d import calculate_gains as calculate_gains_2d

from gui.plot import plot_audio_channels, plot_speaker_and_source_positions

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class AudioPanningApp:
    def __init__(self, master):
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
            update_visuals=False,
        )

        # Virtual source starts centered in front of the listener
        self.virtual_source_position = self._ensure_vector3(
            (0.0, self.default_speaker_y_distance, self.default_speaker_z_distance)
        )

        self.play_button = PlayButton(self.frame, self.play_audio)  # Create a play button and assign the play_audio method
        self.play_button.pack(side='left', padx=5, pady=5)  # Pack the play button

        self.stop_button = Button(self.frame, text="Stop", command=self.stop_audio)  # Create a stop button and assign the stop_audio method
        self.stop_button.pack(side='left', padx=5, pady=5)  # Pack the stop button

        self._build_virtual_source_controls()

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
        self.set_virtual_position()  # Initialize VBAP gains based on default virtual source position
        self.playback_index = 0 # Initialize playback index
        
        # Experiment pan positions Menu
        self.experiment_pan_positions = self.calculate_experiment_pan_positions()
        self.selected_experiment_pan_position = StringVar()
        default_index = 8 if len(self.experiment_pan_positions) > 8 else 0
        self.selected_experiment_pan_position.set(self.experiment_pan_positions[default_index])
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

    def _build_virtual_source_controls(self):
        """Create canvas and controls for positioning the virtual source in 3D."""
        controls_frame = Frame(self.frame)
        controls_frame.pack(pady=10, padx=5)

        Label(controls_frame, text="Virtual source position").pack(anchor='w')

        canvas_container = Frame(controls_frame)
        canvas_container.pack(anchor='w', pady=(4, 0))

        self.canvas_size = 320
        self.canvas_margin = 16
        self.canvas_bounds = None

        self.panning_canvas = Canvas(
            canvas_container,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="white",
            highlightthickness=1,
            highlightbackground="gray",
        )
        self.panning_canvas.pack(side='left')
        self.panning_canvas.bind("<Button-1>", self._on_canvas_interaction)
        self.panning_canvas.bind("<B1-Motion>", self._on_canvas_interaction)

        height_frame = Frame(canvas_container)
        height_frame.pack(side='left', padx=(12, 0))
        Label(height_frame, text="Height (cm)").pack()
        self.height_scale = Scale(
            height_frame,
            from_=-100,
            to=100,
            orient=VERTICAL,
            resolution=5,
            command=self._on_height_change,
            length=200,
        )
        self.height_scale.set(self.virtual_source_position[2])
        self.height_scale.pack()

        self._height_scale_busy = False
        self.virtual_source_marker = None
        print("[VirtualSourceUI] Canvas and height controls initialized.")
        self._update_canvas_bounds()

    def create_speaker_input_fields(self):
        """Create input fields for configuring speakers using distances from the listener."""
        configuration_frame = Frame(self.frame)
        configuration_frame.pack(pady=10, anchor='w')

        Label(configuration_frame, text="Speaker layout").pack(anchor='w')

        count_frame = Frame(configuration_frame)
        count_frame.pack(anchor='w', pady=(4, 2))

        Label(count_frame, text="Number of speakers:").pack(side='left')
        self.speaker_count_var = StringVar()
        self.speaker_count_var.set(str(len(self.speakers)))
        self.speaker_count_entry = Entry(count_frame, textvariable=self.speaker_count_var, width=5)
        self.speaker_count_entry.pack(side='left', padx=(4, 4))
        Button(count_frame, text="Apply Count", command=self.update_speaker_count).pack(side='left')

        self.speaker_entries_frame = Frame(configuration_frame)
        self.speaker_entries_frame.pack(anchor='w', pady=(4, 6))
        self.speaker_controls = []

        Button(configuration_frame, text="Apply Speaker Positions", command=self.update_speaker_positions).pack(anchor='w')

        # Build the initial set of entries
        self.update_speaker_count()

    def update_speaker_positions(self):
        """Update speaker positions based on the current distance inputs."""
        new_positions = []
        for index, control in enumerate(self.speaker_controls):
            try:
                x_distance = float(control["x_entry"].get())
                y_distance = float(control["y_entry"].get())
                z_distance = float(control["z_entry"].get())
            except ValueError:
                print("Invalid input for speaker coordinates. Please enter numeric values.")
                return
            if x_distance < 0 or y_distance < 0 or z_distance < 0:
                print("Distances must be non-negative.")
                return
            # Convert the entered distances + directions into signed coordinates
            x = self._distance_with_direction(x_distance, control["x_dir"].get(), axis="x")
            y = self._distance_with_direction(y_distance, control["y_dir"].get(), axis="y")
            z = self._distance_with_direction(z_distance, control["z_dir"].get(), axis="z")
            new_positions.append((x, y, z))

        if not new_positions:
            print("No speakers configured.")
            return

        self.set_speaker_positions(new_positions)

        # Keep the virtual source within reachable bounds after updating speakers
        self._clamp_virtual_source_to_bounds()

        print(f"[SpeakerConfig] Updated speaker positions: {self._get_active_speakers()}")  # Debug print
        print(f"[SpeakerConfig] Updated virtual source: {self.virtual_source_position}")  # Debug print

        # Update experiment pan positions (if the widget exists yet)
        self.experiment_pan_positions = self.calculate_experiment_pan_positions()
        menu_widget = getattr(self, "experiment_pan_positions_menu", None)
        if self.experiment_pan_positions and menu_widget is not None:
            self.selected_experiment_pan_position.set(self.experiment_pan_positions[0])
            menu_widget['menu'].delete(0, 'end')
            for position in self.experiment_pan_positions:
                menu_widget['menu'].add_command(
                    label=position,
                    command=lambda value=position: self.selected_experiment_pan_position.set(value)
                )

    def update_speaker_count(self):
        """Rebuild speaker inputs when the user changes the desired speaker count."""
        try:
            count = int(self.speaker_count_var.get())
        except ValueError:
            print("Invalid speaker count. Please enter a whole number.")
            return
        if count < 2:
            count = 2
            self.speaker_count_var.set(str(count))
        self._build_speaker_entry_rows(count)
        self.update_speaker_positions()

    def _build_speaker_entry_rows(self, count):
        """Create entry widgets for each speaker based on the requested count."""
        for widget in self.speaker_entries_frame.winfo_children():
            widget.destroy()
        self.speaker_controls = []
        existing = list(self.speakers)
        for idx in range(count):
            base_position = existing[idx] if idx < len(existing) else self._default_speaker_position(idx, count)
            x_distance, x_direction = self._coordinate_to_ui_components(base_position[0], axis="x")
            y_distance, y_direction = self._coordinate_to_ui_components(base_position[1], axis="y")
            z_distance, z_direction = self._coordinate_to_ui_components(base_position[2], axis="z")

            row = Frame(self.speaker_entries_frame)
            row.pack(anchor='w', pady=2)

            Label(row, text=f"Speaker {idx + 1}").grid(row=0, column=0, padx=(0, 6))

            Label(row, text="x (cm):").grid(row=0, column=1)
            x_entry = Entry(row, width=6)
            x_entry.grid(row=0, column=2, padx=(2, 2))
            x_entry.insert(0, f"{x_distance:.1f}")
            x_dir_var = StringVar(value=x_direction)
            x_menu = OptionMenu(row, x_dir_var, "Left", "Right", "Center")
            x_menu.grid(row=0, column=3, padx=(2, 8))

            Label(row, text="y (cm):").grid(row=0, column=4)
            y_entry = Entry(row, width=6)
            y_entry.grid(row=0, column=5, padx=(2, 2))
            y_entry.insert(0, f"{y_distance:.1f}")
            y_dir_var = StringVar(value=y_direction)
            y_menu = OptionMenu(row, y_dir_var, "Front", "Back", "Center")
            y_menu.grid(row=0, column=6, padx=(2, 8))

            Label(row, text="z (cm):").grid(row=0, column=7)
            z_entry = Entry(row, width=6)
            z_entry.grid(row=0, column=8, padx=(2, 2))
            z_entry.insert(0, f"{z_distance:.1f}")
            z_dir_var = StringVar(value=z_direction)
            z_menu = OptionMenu(row, z_dir_var, "Above", "Below", "Level")
            z_menu.grid(row=0, column=9, padx=(2, 0))

            self.speaker_controls.append(
                {
                    "x_entry": x_entry,
                    "x_dir": x_dir_var,
                    "x_menu": x_menu,
                    "y_entry": y_entry,
                    "y_dir": y_dir_var,
                    "y_menu": y_menu,
                    "z_entry": z_entry,
                    "z_dir": z_dir_var,
                    "z_menu": z_menu,
                }
            )

    def _default_speaker_position(self, index, total_count):
        """Provide a reasonable default position for newly added speakers."""
        if total_count == 2:
            x = -self.default_speaker_x_distance / 2 if index == 0 else self.default_speaker_x_distance / 2
            y = self.default_speaker_y_distance
            z = self.default_speaker_z_distance
        else:
            x = 0.0
            y = self.default_speaker_y_distance
            z = self.default_speaker_z_distance
        return np.array([x, y, z], dtype=float)

    def _coordinate_to_ui_components(self, value, axis):
        """Convert a signed coordinate into a magnitude and direction label."""
        value = float(value)
        distance = abs(value)
        if axis == "x":
            if value < 0:
                return distance, "Left"
            if value > 0:
                return distance, "Right"
            return distance, "Center"
        if axis == "y":
            if value < 0:
                return distance, "Back"
            if value > 0:
                return distance, "Front"
            return distance, "Center"
        if axis == "z":
            if value < 0:
                return distance, "Below"
            if value > 0:
                return distance, "Above"
            return distance, "Level"
        raise ValueError(f"Unsupported axis: {axis}")

    def _distance_with_direction(self, distance, direction, axis):
        """Translate a positive distance and direction label into a signed coordinate."""
        direction_key = direction.lower()
        if axis == "x":
            if direction_key == "left":
                return -distance
            if direction_key == "right":
                return distance
            return 0.0
        if axis == "y":
            if direction_key == "back":
                return -distance
            if direction_key == "front":
                return distance
            return 0.0
        if axis == "z":
            if direction_key == "below":
                return -distance
            if direction_key == "above":
                return distance
            return 0.0
        raise ValueError(f"Unsupported axis: {axis}")

    def _clamp_virtual_source_to_bounds(self):
        """Limit the virtual source to remain within the current speaker bounds."""
        bounds = self._calculate_virtual_bounds()
        if bounds is None:
            return
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = bounds
        x = min(max(self.virtual_source_position[0], x_min), x_max)
        y = min(max(self.virtual_source_position[1], y_min), y_max)
        z = min(max(self.virtual_source_position[2], z_min), z_max)
        self.virtual_source_position = np.array([x, y, z], dtype=float)

    def _calculate_virtual_bounds(self):
        """Estimate reasonable limits for the virtual source based on speaker layout."""
        active = self._get_active_speakers()
        if not active:
            return None
        speaker_array = np.array(active, dtype=float)
        xs = speaker_array[:, 0]
        ys = speaker_array[:, 1]
        zs = speaker_array[:, 2]

        def expand(values, fallback_span):
            minimum = float(np.min(values))
            maximum = float(np.max(values))
            span = maximum - minimum
            if span <= 1e-6:
                span = fallback_span
                minimum -= span / 2.0
                maximum += span / 2.0
            # Pad the range so the listener (0,0,0) stays visible and the marker has room
            margin = max(25.0, span * 0.15)
            return minimum - margin, maximum + margin

        x_bounds = expand(np.append(xs, [0.0]), fallback_span=self.default_speaker_x_distance or 200.0)
        y_bounds = expand(np.append(ys, [0.0]), fallback_span=self.default_speaker_y_distance or 200.0)
        z_bounds = expand(np.append(zs, [0.0]), fallback_span=200.0)
        return x_bounds, y_bounds, z_bounds

    def _update_canvas_bounds(self):
        """Refresh canvas scaling to reflect current speaker layout."""
        bounds = self._calculate_virtual_bounds()
        if bounds is None:
            print("[VirtualSourceUI] No active speakers; skipping canvas bounds update.")
            return
        self.canvas_bounds = bounds
        print(f"[VirtualSourceUI] Canvas bounds updated to {self.canvas_bounds}")
        self._update_panning_canvas_graphics()
        self._update_height_scale(bounds)
        self._sync_virtual_position_ui()

    def _update_panning_canvas_graphics(self):
        """Redraw static canvas elements (border and axes)."""
        if not hasattr(self, "panning_canvas"):
            return
        self.panning_canvas.delete("ui")
        margin = self.canvas_margin
        size = self.canvas_size
        self.panning_canvas.create_rectangle(
            margin,
            margin,
            size - margin,
            size - margin,
            outline="lightgray",
            tags="ui"
        )
        bounds = self.canvas_bounds
        if bounds is None:
            return
        (x_min, x_max), (y_min, y_max), _ = bounds
        if x_max - x_min <= 0 or y_max - y_min <= 0:
            return
        zero_x, zero_y = self._world_to_canvas(0.0, 0.0)
        self.panning_canvas.create_line(
            margin,
            zero_y,
            size - margin,
            zero_y,
            fill="#dddddd",
            tags="ui"
        )
        self.panning_canvas.create_line(
            zero_x,
            margin,
            zero_x,
            size - margin,
            fill="#dddddd",
            tags="ui"
        )

    def _update_height_scale(self, bounds):
        """Adjust the range of the height control to current speaker layout."""
        if getattr(self, "height_scale", None) is None:
            return
        _, _, (z_min, z_max) = bounds
        if z_max - z_min <= 1e-6:
            z_min -= 50.0
            z_max += 50.0
        self.height_scale.config(from_=z_max, to=z_min)  # inverted so higher values are up

    def _sync_virtual_position_ui(self):
        """Update canvas marker and height control to match the virtual source position."""
        if getattr(self, "panning_canvas", None) is not None and self.canvas_bounds is not None:
            cx, cy = self._world_to_canvas(
                self.virtual_source_position[0],
                self.virtual_source_position[1],
            )
            radius = 6
            if self.virtual_source_marker is None:
                self.virtual_source_marker = self.panning_canvas.create_oval(
                    cx - radius,
                    cy - radius,
                    cx + radius,
                    cy + radius,
                    fill="#7b3fe4",
                    outline="",
                )
            else:
                self.panning_canvas.coords(
                    self.virtual_source_marker,
                    cx - radius,
                    cy - radius,
                    cx + radius,
                    cy + radius,
                )
        if getattr(self, "height_scale", None) is not None:
            if not getattr(self, "_height_scale_busy", False):
                self._height_scale_busy = True
                try:
                    self.height_scale.set(self.virtual_source_position[2])
                finally:
                    self._height_scale_busy = False

    def _canvas_to_world(self, canvas_x, canvas_y):
        """Translate canvas coordinates into world (cm) coordinates."""
        if self.canvas_bounds is None:
            return 0.0, 0.0
        (x_min, x_max), (y_min, y_max), _ = self.canvas_bounds
        margin = self.canvas_margin
        size = self.canvas_size
        usable = size - 2 * margin
        usable = max(usable, 1)
        normalized_x = (canvas_x - margin) / usable
        normalized_y = (canvas_y - margin) / usable
        normalized_x = min(max(normalized_x, 0.0), 1.0)
        normalized_y = min(max(normalized_y, 0.0), 1.0)
        world_x = x_min + normalized_x * (x_max - x_min)
        # invert y axis so top of canvas is positive y (in front)
        world_y = y_max - normalized_y * (y_max - y_min)
        return world_x, world_y

    def _world_to_canvas(self, world_x, world_y):
        """Project world coordinates back to canvas coordinates."""
        if self.canvas_bounds is None:
            return self.canvas_size / 2.0, self.canvas_size / 2.0
        (x_min, x_max), (y_min, y_max), _ = self.canvas_bounds
        margin = self.canvas_margin
        size = self.canvas_size
        usable = size - 2 * margin
        usable = max(usable, 1)
        if x_max - x_min <= 1e-6:
            cx = size / 2.0
        else:
            cx = margin + ((world_x - x_min) / (x_max - x_min)) * usable
        if y_max - y_min <= 1e-6:
            cy = size / 2.0
        else:
            cy = margin + ((y_max - world_y) / (y_max - y_min)) * usable
        return cx, cy

    def _on_canvas_interaction(self, event):
        """Handle click/drag events on the panning canvas."""
        world_x, world_y = self._canvas_to_world(event.x, event.y)
        print(f"[VirtualSourceUI] Canvas interaction -> world ({world_x:.2f}, {world_y:.2f})")
        self.set_virtual_position(x=world_x, y=world_y)

    def _on_height_change(self, value):
        """Update virtual source height from the height control."""
        if getattr(self, "_height_scale_busy", False):
            return
        try:
            z = float(value)
        except ValueError:
            return
        self._height_scale_busy = True
        try:
            print(f"[VirtualSourceUI] Height control -> z {z:.2f}")
            self.set_virtual_position(z=z)
        finally:
            self._height_scale_busy = False


    def calculate_experiment_pan_positions(self):
        """Calculate experiment pan positions based on the distance between speakers."""
        active = self._get_active_speakers(dims=2)
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

        active_speakers = self._get_active_speakers()
        if not active_speakers:
            print("No active speakers configured.")
            self.channel_gains = np.zeros(0, dtype='float32')
            self.left_gain = 0.0
            self.right_gain = 0.0
            self._sync_virtual_position_ui()
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
        self._sync_virtual_position_ui()

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
            self._get_active_speakers(dims=3),
            self.virtual_source_position,
        )

    def set_speaker_positions(self, positions, update_visuals=True):
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
        if update_visuals:
            self._update_canvas_bounds()
            self.set_virtual_position()

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
        return self._solve_vbap_3d(np.asarray(active_speakers, dtype=float), np.asarray(virtual_source, dtype=float))

    def _solve_vbap_3d(self, speakers, virtual_source):
        """Select a speaker triplet and compute 3D VBAP gains."""
        from itertools import combinations
        from audio.vbap3d import calculate_gains as calculate_gains_3d

        virtual_norm = np.linalg.norm(virtual_source)
        if virtual_norm <= 1e-9:
            return np.zeros(speakers.shape[0], dtype='float32')

        best_indices = None
        best_gains = None
        best_score = float("inf")
        target_unit = virtual_source / virtual_norm

        for combo in combinations(range(speakers.shape[0]), 3):
            c1, c2, c3 = (speakers[idx] for idx in combo)
            try:
                gains = calculate_gains_3d(c1, c2, c3, virtual_source)
            except Exception:
                print(f"[VBAP3D] Numerical failure for combo {combo}")
                continue
            if not np.all(np.isfinite(gains)):
                print(f"[VBAP3D] Non-finite gains for combo {combo}: {gains}")
                continue

            normals = []
            for vector in (c1, c2, c3):
                norm = np.linalg.norm(vector)
                if norm <= 1e-9:
                    normals.append(vector)
                else:
                    normals.append(vector / norm)
            spread_matrix = np.column_stack(normals)
            residual = np.linalg.norm(spread_matrix @ gains - target_unit)
            negative_penalty = np.sum(np.clip(-gains, 0.0, None))
            score = residual + 0.25 * negative_penalty

            if score < best_score:
                best_score = score
                best_indices = combo
                best_gains = gains
                print(f"[VBAP3D] New best combo {combo} with score {score:.4f} and gains {gains}")

        gains_vector = np.zeros(speakers.shape[0], dtype='float32')
        if best_indices is None or best_gains is None:
            print("[VBAP3D] No valid loudspeaker triple found; returning silence.")
            return gains_vector

        for slot, gain in zip(best_indices, best_gains):
            gains_vector[slot] = gain
        print(f"[VBAP3D] Selected loudspeaker triple {best_indices} -> gains {gains_vector}")
        return gains_vector
