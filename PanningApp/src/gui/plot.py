import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # Needed for 3D projection


def plot_audio_channels(audio_chunk, channel_gains):
    """Plot a short segment of the rendered channels."""
    if channel_gains is None or len(channel_gains) == 0:
        print("No channel gains available for plotting.")
        return

    sample_size = min(2000, len(audio_chunk))
    if sample_size == 0:
        print("Empty audio chunk; nothing to plot.")
        return

    time = np.arange(sample_size)
    channels = np.asarray(channel_gains, dtype=float)
    rendered = np.outer(audio_chunk[:sample_size], channels)

    peak = np.max(np.abs(rendered))
    if not np.isfinite(peak) or peak == 0:
        peak = 1.0

    fig, ax = plt.subplots(figsize=(10, 6))
    colormap = plt.get_cmap('tab10')
    colors = colormap(np.linspace(0, 1, rendered.shape[1]))

    for idx in range(rendered.shape[1]):
        ax.plot(time, rendered[:, idx], color=colors[idx], label=f'Channel {idx + 1}')

    ax.set_ylim(-1.1 * peak, 1.1 * peak)
    ax.set_xlabel('Audio Samples', fontsize=14)
    ax.set_ylabel('Amplitude', fontsize=14)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=10, ncol=2)

    plt.tight_layout()
    plt.show()


def plot_speaker_and_source_positions(speaker_positions, virtual_source_position):
    """Plot speaker positions, listener, and virtual source in 3D space."""
    if len(speaker_positions) == 0:
        print("No speaker positions to plot.")
        return

    speaker_array = np.asarray(speaker_positions, dtype=float)
    virtual_source = np.asarray(virtual_source_position, dtype=float)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter([0], [0], [0], color='green', label='Listener')
    ax.scatter(
        speaker_array[:, 0],
        speaker_array[:, 1],
        speaker_array[:, 2],
        color='blue',
        label='Speakers',
    )
    ax.scatter(
        [virtual_source[0]],
        [virtual_source[1]],
        [virtual_source[2]],
        color='purple',
        marker='x',
        s=80,
        label='Virtual Source',
    )

    for idx, position in enumerate(speaker_array):
        ax.text(
            position[0],
            position[1],
            position[2],
            f"S{idx + 1}",
            fontsize=9,
            color='blue',
        )

    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_zlabel('Z (cm)')

    _set_equal_aspect(ax, speaker_array, virtual_source)

    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def _set_equal_aspect(ax, speaker_array, virtual_source):
    """Set equal aspect ratio for all axes in the 3D plot."""
    points = np.vstack([speaker_array, virtual_source, np.zeros((1, 3))])
    max_range = np.ptp(points, axis=0).max()
    if max_range == 0:
        max_range = 1.0
    mid = points.mean(axis=0)
    for axis, center in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], mid):
        axis(center - max_range / 2, center + max_range / 2)
