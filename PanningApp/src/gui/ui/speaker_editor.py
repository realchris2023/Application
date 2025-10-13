"""TK widgets for configuring speaker layouts using distance + direction inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np
from tkinter import Button, Entry, Frame, Label, OptionMenu, StringVar

from gui.speakers import (
    DIRECTION_LABELS,
    SpeakerLayout,
    coordinate_to_distance_direction,
    distance_with_direction,
)


@dataclass
class SpeakerEntry:
    x_entry: Entry
    x_dir: StringVar
    y_entry: Entry
    y_dir: StringVar
    z_entry: Entry
    z_dir: StringVar


class SpeakerEditor(Frame):
    """Encapsulates the speaker configuration form."""

    def __init__(
        self,
        master,
        layout: SpeakerLayout,
        apply_callback: Callable[[List[np.ndarray]], None],
        default_distances: Tuple[float, float, float],
    ):
        super().__init__(master)
        self.layout = layout
        self.default_x, self.default_y, self.default_z = default_distances
        self.apply_callback = apply_callback

        Label(self, text="Speaker layout").pack(anchor="w")

        count_frame = Frame(self)
        count_frame.pack(anchor="w", pady=(4, 2))

        Label(count_frame, text="Number of speakers:").pack(side="left")
        self.count_var = StringVar(value=str(len(self.layout.positions)))
        Entry(count_frame, textvariable=self.count_var, width=5).pack(side="left", padx=(4, 4))
        Button(count_frame, text="Apply Count", command=self._apply_count).pack(side="left")

        self.entries_frame = Frame(self)
        self.entries_frame.pack(anchor="w", pady=(4, 6))
        self.entries: List[SpeakerEntry] = []

        Button(self, text="Apply Speaker Positions", command=self._apply_positions).pack(anchor="w")

        self._rebuild_rows(int(self.count_var.get() or 2))

    # ------------------------------------------------------------------
    def _apply_count(self):
        try:
            count = int(self.count_var.get())
        except ValueError:
            print("Invalid speaker count. Please enter a whole number.")
            return
        if count < 2:
            count = 2
            self.count_var.set(str(count))
        self._rebuild_rows(count)
        self._apply_positions()

    def _apply_positions(self):
        vectors: List[np.ndarray] = []
        for idx, entry in enumerate(self.entries):
            try:
                x = float(entry.x_entry.get())
                y = float(entry.y_entry.get())
                z = float(entry.z_entry.get())
            except ValueError:
                print("Invalid input for speaker coordinates. Please enter numeric values.")
                return
            if x < 0 or y < 0 or z < 0:
                print("Distances must be non-negative.")
                return
            vectors.append(
                self.layout.ensure_vector(
                    [
                        distance_with_direction(x, entry.x_dir.get(), "x"),
                        distance_with_direction(y, entry.y_dir.get(), "y"),
                        distance_with_direction(z, entry.z_dir.get(), "z"),
                    ]
                )
            )
        if not vectors:
            print("No speakers configured.")
            return
        self.layout.set_positions(vectors)
        self.apply_callback(self.layout.positions)

    def _rebuild_rows(self, count: int):
        for widget in self.entries_frame.winfo_children():
            widget.destroy()
        self.entries.clear()

        existing = self.layout.positions
        for idx in range(count):
            base = existing[idx] if idx < len(existing) else self._default_position(idx, count)
            x_distance, x_dir = coordinate_to_distance_direction(base[0], "x")
            y_distance, y_dir = coordinate_to_distance_direction(base[1], "y")
            z_distance, z_dir = coordinate_to_distance_direction(base[2], "z")

            row = Frame(self.entries_frame)
            row.pack(anchor="w", pady=2)
            Label(row, text=f"Speaker {idx + 1}").grid(row=0, column=0, padx=(0, 6))

            x_entry = self._create_distance_field(row, 1, "Side distance (cm):", x_distance, x_dir, DIRECTION_LABELS["x"])
            y_entry = self._create_distance_field(row, 4, "Front distance (cm):", y_distance, y_dir, DIRECTION_LABELS["y"])
            z_entry = self._create_distance_field(row, 7, "Height (cm):", z_distance, z_dir, DIRECTION_LABELS["z"])

            self.entries.append(SpeakerEntry(*x_entry, *y_entry, *z_entry))

    def _create_distance_field(
        self,
        row: Frame,
        column: int,
        label_text: str,
        distance: float,
        direction: str,
        options: Iterable[str],
    ):
        Label(row, text=label_text).grid(row=0, column=column, sticky="w")
        entry = Entry(row, width=6)
        entry.grid(row=0, column=column + 1, padx=(2, 2))
        entry.insert(0, f"{distance:.1f}")
        var = StringVar(value=direction)
        OptionMenu(row, var, *options).grid(row=0, column=column + 2, padx=(2, 8))
        return entry, var

    def _default_position(self, index: int, total_count: int) -> np.ndarray:
        if total_count == 2:
            x = -self.default_x / 2 if index == 0 else self.default_x / 2
            y = self.default_y
            z = self.default_z
        else:
            x = 0.0
            y = self.default_y
            z = self.default_z
        return np.array([x, y, z], dtype=float)
