class PanKnob:
    def __init__(self, master, callback):
        self.master = master
        self.callback = callback
        self.pan_value = 0.0  # Range from -1.4 (left) to 1.4 (right)

        # Create the knob UI element (using a simple scale for demonstration)
        self.knob = self.create_knob()

    def create_knob(self):
        import tkinter as tk

        knob = tk.Scale(self.master, from_=-1.4, to=1.4, resolution=0.01,
                        orient=tk.HORIZONTAL, label="Pan", command=self.update_pan)
        knob.pack()
        return knob

    def update_pan(self, value):
        self.pan_value = float(value)
        self.callback(self.pan_value)  # Update the audio panning based on the knob value

    def get_pan_value(self):
        return self.pan_value