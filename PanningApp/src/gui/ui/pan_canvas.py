"""
Interactive canvas for positioning the virtual source with a faux-3D projection.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import cos, sin, radians
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tkinter import Canvas, Frame, Label, Scale, VERTICAL


@dataclass
class Bounds3D:
    """Axis-aligned bounds used to map world coordinates onto the canvas."""

    x: Tuple[float, float]
    y: Tuple[float, float]
    z: Tuple[float, float]

    @classmethod
    def from_points(cls, points: np.ndarray, defaults: Tuple[float, float, float]) -> "Bounds3D":
        """Generate bounds from the supplied speaker positions."""
        if points.size == 0:
            return cls(
                (-defaults[0], defaults[0]),
                (-defaults[1], defaults[1]),
                (-defaults[2], defaults[2]),
            )
        x_bounds = cls._expand(points[:, 0], defaults[0] or 200.0)
        y_bounds = cls._expand(points[:, 1], defaults[1] or 200.0)
        z_bounds = cls._expand(points[:, 2], defaults[2] or 200.0)
        return cls(x_bounds, y_bounds, z_bounds)

    @staticmethod
    def _expand(values: np.ndarray, fallback_span: float) -> Tuple[float, float]:
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        span = maximum - minimum
        if span <= 1e-6:
            span = fallback_span
            minimum -= span / 2.0
            maximum += span / 2.0
        margin = max(25.0, span * 0.15)
        return minimum - margin, maximum + margin


class PanCanvas(Frame):
    """Encapsulates the virtual-source panning canvas with a simple 3D projection."""

    def __init__(self, master, move_callback, initial_position: Sequence[float], canvas_size: int = 360):
        super().__init__(master)
        self.move_callback = move_callback
        self.canvas_size = canvas_size
        self.canvas_margin = 20
        self.bounds: Optional[Bounds3D] = None
        self.room_bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = None
        self.virtual_position = np.array(initial_position, dtype=float)
        self._floor_hull: List[Tuple[float, float]] = []

        Label(self, text="Virtual source position").pack(anchor="w")

        container = Frame(self)
        container.pack(anchor="w", pady=(4, 0))

        self.canvas = Canvas(
            container,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="#f7f7fb",
            highlightthickness=1,
            highlightbackground="#aaaaaa",
        )
        self.canvas.pack(side="left")
        self.canvas.bind("<Button-1>", self._on_canvas_drag)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)

        self.height_scale = Scale(
            container,
            from_=-100,
            to=100,
            orient=VERTICAL,
            resolution=5,
            length=220,
            command=self._on_height_change,
        )
        self.height_scale.set(self.virtual_position[2])
        self.height_scale.pack(side="left", padx=(12, 0))

        self._height_busy = False
        self._source_marker = None
        self._rotation = np.eye(3)
        self._rotation_T = np.eye(3)
        self._scale = 1.0
        self._offset = (self.canvas_size / 2.0, self.canvas_size / 2.0)
        self._center = np.zeros(3, dtype=float)
        self._floor_z = 0.0
        self._ceiling_z = 0.0

    # ------------------------------------------------------------------
    # Scene configuration
    # ------------------------------------------------------------------
    def configure_scene(self, speakers: Iterable[Sequence[float]], defaults: Tuple[float, float, float]):
        """Refresh bounds and redraw speakers/hull based on the latest layout."""
        speaker_array = np.array([np.asarray(s, dtype=float) for s in speakers], dtype=float)
        if speaker_array.size == 0:
            span_x, span_y, span_z = defaults
            speaker_array = np.array(
                [
                    [-span_x / 2.0, span_y, 0.0],
                    [span_x / 2.0, span_y, 0.0],
                ],
                dtype=float,
            )

        room_min = speaker_array.min(axis=0)
        room_max = speaker_array.max(axis=0)
        if abs(room_min[2] - room_max[2]) < 1e-6:
            room_max[2] = room_min[2] + 1.0
        self.room_bounds = (
            (float(room_min[0]), float(room_max[0])),
            (float(room_min[1]), float(room_max[1])),
            (float(room_min[2]), float(room_max[2])),
        )
        self._floor_z = self.room_bounds[2][0]
        self._ceiling_z = self.room_bounds[2][1]

        self.bounds = Bounds3D.from_points(speaker_array, defaults)
        self._update_projection()
        self.height_scale.config(from_=self.bounds.z[1], to=self.bounds.z[0])

        if speaker_array.shape[0] >= 3:
            self._floor_hull = self._convex_hull([(float(x), float(y)) for x, y in speaker_array[:, :2]])
        else:
            self._floor_hull = []

        self._draw_background()
        self._draw_speakers(speaker_array)
        self._draw_virtual_source()

    def update_virtual_position(self, position: Sequence[float]):
        """Keep an external virtual source position mirrored on the canvas."""
        self.virtual_position = np.asarray(position, dtype=float)
        self._draw_virtual_source()
        if not self._height_busy:
            self._height_busy = True
            try:
                self.height_scale.set(self.virtual_position[2])
            finally:
                self._height_busy = False

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------
    def _draw_background(self):
        if self.bounds is None:
            return
        self.canvas.delete("background")
        self.canvas.delete("room")
        self.canvas.delete("axis")

        self.canvas.create_rectangle(
            0,
            0,
            self.canvas_size,
            self.canvas_size,
            fill="#f7f7fb",
            outline="",
            tags="background",
        )

        buffer_box = (self.bounds.x, self.bounds.y, self.bounds.z)
        self._draw_box(buffer_box, tag="background", outline="#c5c5d6", width=1, dash=(4, 3))

        if self.room_bounds is not None:
            self._draw_box(self.room_bounds, tag="room", outline="#696dd8", width=2)
            if self._floor_hull:
                hull_coords: List[float] = []
                for px, py in self._floor_hull:
                    hull_coords.extend(self._project((px, py, self._floor_z)))
                self.canvas.create_polygon(
                    *hull_coords,
                    outline="#5f9ed1",
                    fill="#dfe9ff",
                    stipple="gray12",
                    width=1.5,
                    tags="room",
                )

        listener_xy = self._project((0.0, 0.0, self._floor_z))
        self.canvas.create_oval(
            listener_xy[0] - 4,
            listener_xy[1] - 4,
            listener_xy[0] + 4,
            listener_xy[1] + 4,
            fill="#2ca02c",
            outline="",
            tags="room",
        )

        axis_points = {
            "Front": (self._center[0], self.bounds.y[1], self._floor_z),
            "Back": (self._center[0], self.bounds.y[0], self._floor_z),
            "Left": (self.bounds.x[0], self._center[1], self._floor_z),
            "Right": (self.bounds.x[1], self._center[1], self._floor_z),
            "Up": (self._center[0], self._center[1], self.bounds.z[1]),
        }
        for label, point in axis_points.items():
            sx, sy = self._project(point)
            self.canvas.create_text(
                sx,
                sy,
                text=label,
                fill="#676767",
                font=("TkDefaultFont", 9, "bold"),
                tags="axis",
            )

    def _draw_speakers(self, speaker_array: np.ndarray):
        self.canvas.delete("speakers")
        if speaker_array.size == 0 or self.bounds is None:
            return
        heights = speaker_array[:, 2]
        z_min = float(np.min(heights))
        z_max = float(np.max(heights))

        for idx, speaker in enumerate(speaker_array):
            top = tuple(speaker.tolist())
            bottom = (speaker[0], speaker[1], self._floor_z)
            top_screen = self._project(top)
            bottom_screen = self._project(bottom)
            self.canvas.create_line(*bottom_screen, *top_screen, fill="#9a9a9a", tags="speakers")
            radius = self._height_scaled_radius(speaker[2], z_min, z_max)
            colour = self._height_colour(speaker[2], z_min, z_max)
            self.canvas.create_oval(
                top_screen[0] - radius,
                top_screen[1] - radius,
                top_screen[0] + radius,
                top_screen[1] + radius,
                fill=colour,
                outline="#1f1f1f",
                width=1,
                tags="speakers",
            )
            self.canvas.create_text(
                top_screen[0],
                top_screen[1] - radius - 8,
                text=f"S{idx + 1}",
                fill="#1f1f1f",
                font=("TkDefaultFont", 9, "bold"),
                tags="speakers",
            )

    def _draw_virtual_source(self):
        if self.bounds is None:
            return
        self.canvas.delete("virtual_source")

        top_point = tuple(self.virtual_position.tolist())
        base_point = (self.virtual_position[0], self.virtual_position[1], self._floor_z)
        top_screen = self._project(top_point)
        base_screen = self._project(base_point)
        self.canvas.create_line(*base_screen, *top_screen, fill="#7b3fe4", dash=(3, 2), tags="virtual_source")

        radius = self._height_scaled_radius(
            self.virtual_position[2],
            self.bounds.z[0],
            self.bounds.z[1],
            base=7,
            span=8,
        )
        colour = self._height_colour(
            self.virtual_position[2],
            self.bounds.z[0],
            self.bounds.z[1],
            low="#8058d3",
            high="#d884f5",
        )
        self._source_marker = self.canvas.create_oval(
            top_screen[0] - radius,
            top_screen[1] - radius,
            top_screen[0] + radius,
            top_screen[1] + radius,
            fill=colour,
            outline="#3b1f66",
            width=1.5,
            tags="virtual_source",
        )
        self.canvas.create_text(
            top_screen[0],
            top_screen[1] - radius - 10,
            text="VS",
            fill="#3b1f66",
            font=("TkDefaultFont", 8, "bold"),
            tags="virtual_source",
        )

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------
    def _on_canvas_drag(self, event):
        if self.bounds is None:
            return
        world_x, world_y = self._screen_to_world(event.x, event.y, self.virtual_position[2])
        self.move_callback(x=world_x, y=world_y)

    def _on_height_change(self, value):
        if self._height_busy:
            return
        try:
            z = float(value)
        except ValueError:
            return
        self._height_busy = True
        try:
            self.move_callback(z=z)
        finally:
            self._height_busy = False

    # ------------------------------------------------------------------
    # Coordinate transforms & helpers
    # ------------------------------------------------------------------
    def _project(self, point: Sequence[float]) -> Tuple[float, float]:
        centered = np.asarray(point, dtype=float) - self._center
        q = self._rotation @ centered
        sx = self._offset[0] + self._scale * q[0]
        sy = self._offset[1] - self._scale * q[1]
        return sx, sy

    def _screen_to_world(self, canvas_x: float, canvas_y: float, z: float) -> Tuple[float, float]:
        q0 = (canvas_x - self._offset[0]) / self._scale
        q1 = -(canvas_y - self._offset[1]) / self._scale
        rt = self._rotation_T
        denom = rt[2, 2]
        if abs(denom) < 1e-9:
            q2 = 0.0
        else:
            q2 = (z - self._center[2] - rt[2, 0] * q0 - rt[2, 1] * q1) / denom
        q = np.array([q0, q1, q2], dtype=float)
        world = self._center + rt @ q
        return float(world[0]), float(world[1])

    def _update_projection(self):
        if self.bounds is None:
            return
        self._center = np.array(
            [
                (self.bounds.x[0] + self.bounds.x[1]) / 2.0,
                (self.bounds.y[0] + self.bounds.y[1]) / 2.0,
                (self.bounds.z[0] + self.bounds.z[1]) / 2.0,
            ],
            dtype=float,
        )
        azimuth = radians(-40.0)
        elevation = radians(25.0)
        rot_y = np.array(
            [
                [cos(azimuth), 0.0, sin(azimuth)],
                [0.0, 1.0, 0.0],
                [-sin(azimuth), 0.0, cos(azimuth)],
            ],
            dtype=float,
        )
        rot_x = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, cos(elevation), -sin(elevation)],
                [0.0, sin(elevation), cos(elevation)],
            ],
            dtype=float,
        )
        self._rotation = rot_x @ rot_y
        self._rotation_T = self._rotation.T
        drawable = self.canvas_size / 2.0 - self.canvas_margin
        max_extent = 1.0
        for corner in product(self.bounds.x, self.bounds.y, self.bounds.z):
            q = self._rotation @ (np.array(corner, dtype=float) - self._center)
            max_extent = max(max_extent, abs(q[0]), abs(q[1]))
        self._scale = drawable / max_extent
        self._offset = (self.canvas_size / 2.0, self.canvas_size / 2.0)

    def _draw_box(
        self,
        box: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
        tag: str,
        outline: str,
        width: int = 1,
        dash: Optional[Tuple[int, int]] = None,
    ):
        x_min, x_max = box[0]
        y_min, y_max = box[1]
        z_min, z_max = box[2]
        corners = [
            (x_min, y_min, z_min),
            (x_max, y_min, z_min),
            (x_min, y_max, z_min),
            (x_max, y_max, z_min),
            (x_min, y_min, z_max),
            (x_max, y_min, z_max),
            (x_min, y_max, z_max),
            (x_max, y_max, z_max),
        ]
        edges = [
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 3),
            (4, 5),
            (4, 6),
            (5, 7),
            (6, 7),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]
        for start_idx, end_idx in edges:
            start = self._project(corners[start_idx])
            end = self._project(corners[end_idx])
            self.canvas.create_line(*start, *end, fill=outline, width=width, dash=dash, tags=tag)

    @staticmethod
    def _convex_hull(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
        unique_points = sorted(set(points))
        if len(unique_points) <= 2:
            return list(unique_points)

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower: List[Tuple[float, float]] = []
        for pt in unique_points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], pt) <= 0:
                lower.pop()
            lower.append(pt)

        upper: List[Tuple[float, float]] = []
        for pt in reversed(unique_points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], pt) <= 0:
                upper.pop()
            upper.append(pt)

        return lower[:-1] + upper[:-1]

    @staticmethod
    def _height_scaled_radius(value, minimum, maximum, base=5, span=8):
        if maximum - minimum <= 1e-6:
            return base + span / 2
        norm = (value - minimum) / (maximum - minimum)
        return base + span * norm

    @staticmethod
    def _height_colour(value, minimum, maximum, low="#1f77b4", high="#ff7f0e"):
        if maximum - minimum <= 1e-6:
            return low
        ratio = min(max((value - minimum) / (maximum - minimum), 0.0), 1.0)
        return PanCanvas._interpolate_colour(low, high, ratio)

    @staticmethod
    def _interpolate_colour(start_hex: str, end_hex: str, ratio: float) -> str:
        start = np.array([int(start_hex[i : i + 2], 16) for i in (1, 3, 5)], dtype=float)
        end = np.array([int(end_hex[i : i + 2], 16) for i in (1, 3, 5)], dtype=float)
        blended = start + ratio * (end - start)
        blended = blended.clip(0, 255).astype(int)
        return "#{:02x}{:02x}{:02x}".format(*blended)

