import matplotlib.pyplot as plt
import numpy as np

def plot_audio_channels(audio_chunk, left_gain, right_gain, pan_value):
    """Plot a sample of left and right channel outputs."""
    # Take a small chunk of audio (e.g., 1000 samples)
    sample_size = min(1000, len(audio_chunk))
    time = np.arange(sample_size)
    
    # Calculate output for both channels
    left_channel = audio_chunk[:sample_size] * left_gain
    right_channel = audio_chunk[:sample_size] * right_gain
    y_limit = 1.0
    
    # Create a single subplot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot left channel
    ax.plot(time, left_channel, 'b-', label='Left Channel')
    
    # Plot right channel
    ax.plot(time, right_channel, 'r-', label='Right Channel')
    
    ax.set_ylim(-y_limit, y_limit)
    ax.set_xlabel('Audio Samples', fontsize=14)
    ax.set_ylabel('Amplitude', fontsize=14)
    # ax.set_title(f'Audio Output Waveforms (Pan: {pan_value:.2f})', fontsize=16)
    ax.grid(True)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()
    

def plot_speaker_and_source_positions(speaker_positions, virtual_source_position, pan_range_scale_limit):
    """Plot the speaker positions, listener position, and virtual source position."""
    fig, ax = plt.subplots()
    # Plot listener position at (0, 0)
    ax.scatter(0, 0, c='green', label='Listener')
    # Plot left speaker position
    ax.scatter(speaker_positions[0][0], speaker_positions[0][1], c='blue', label='Left Speaker')
    # Plot right speaker position
    ax.scatter(speaker_positions[1][0], speaker_positions[1][1], c='red', label='Right Speaker')
    # Plot virtual source position
    ax.scatter(virtual_source_position[0], virtual_source_position[1], c='purple', label='Virtual Source', marker='x')
    # Annotate speaker positions
    ax.annotate(f'({speaker_positions[0][0]}, {speaker_positions[0][1]})', 
                (speaker_positions[0][0], speaker_positions[0][1]), 
                textcoords="offset points", xytext=(-10,-15), ha='center')
    ax.annotate(f'({speaker_positions[1][0]}, {speaker_positions[1][1]})', 
                (speaker_positions[1][0], speaker_positions[1][1]), 
                textcoords="offset points", xytext=(10,-15), ha='center')
    # Annotate virtual source position
    ax.annotate(f'({virtual_source_position[0]}, {virtual_source_position[1]})', 
                (virtual_source_position[0], virtual_source_position[1]), 
                textcoords="offset points", xytext=(0,15), ha='center')
    # Set limits for x and y axes, dependent on speaker positions
    ax.set_xlim(speaker_positions[0][0]*1.05 -abs(virtual_source_position[0]), speaker_positions[1][0]*1.05 + abs(virtual_source_position[0]))
    ax.set_ylim(-10, speaker_positions[1][1]*1.2)
    # Set grid resolution
    ax.xaxis.set_major_locator(plt.MultipleLocator(50))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(25))
    ax.yaxis.set_major_locator(plt.MultipleLocator(25))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(12.5))
    
    ax.set_xlabel('Distance from center (cm)', fontsize=14)
    ax.set_ylabel('Distance to listener (cm)', fontsize=14)
    # ax.set_title('Speaker and Virtual Source Positions', fontsize=16)
    ax.legend(fontsize=10)
    ax.grid(True)
    plt.show()