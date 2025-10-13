# Audio Panning GUI

This project is a graphical user interface (GUI) application for audio panning and playback. It allows users to play audio files and adjust the pan position using a simple interface, while also experimenting with phantom width in audio panning.

## Project Structure

```
audio-panning-gui
├── src
│   ├── main.py                   # Entry point for the application
│   ├── gui                       # Contains GUI components
│   │   ├── __init__.py           # Initializes the GUI package
│   │   ├── app.py                # Main application class for GUI layout
│   │   ├── plot.py               # Plotting amplitude/time
│   │   └── components            # GUI components
│   │       ├── __init__.py       # Ensures package recognition
│   │       ├── play_button.py    # Button for playing audio
│   │       └── pan_knob.py       # Knob for adjusting pan position
│   ├── audio                     # Contains audio processing components
│   │   ├── __init__.py           # Initializes the audio package
│   │   └── vbap.py               # VBAP algorithm and phantom width
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd audio-panning-gui
   ```

2. Create/refresh the virtual environment (Python 3.13 recommended):
   ```
   python3 -m venv venv
   ```

3. Install the required dependencies **inside the virtual environment**:
   ```
   ./venv/bin/python -m pip install --upgrade pip
   ./venv/bin/python -m pip install -r requirements.txt
   ```

4. Run the application (activation optional – the launcher will re-exec inside the venv if needed):
   ```
   python src/main.py
   ```

## Usage Guidelines

- Place custom .wav files in the audio_files folder or use the default test samples.
- 
- Use the play button to start audio playback.
- Adjust the pan knob to change the audio panning position. (going beyond -1 or 1 will activate the phase-flipped copy in opposite speaker)
- Experiment with different audio files to see how the panning affects the sound.

## Overview

This application is designed to provide an intuitive interface for audio playback and panning. It aims to help users understand the effects of panning and phantom width in audio production.
