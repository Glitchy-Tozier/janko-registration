from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from manim import VMobject

from janko_registration.piano.janko_piano_base import (
    Piano,
    create_janko_key,
)


@dataclass
class KeyGeometry:
    """Geometric representation of one Janko key in canonical coordinates."""

    index: int
    polygons: list[np.ndarray]


@dataclass
class PianoGeometry:
    """Geometric representation of the complete canonical Janko piano."""

    keys: list[KeyGeometry]

    @property
    def all_points(self) -> np.ndarray:
        """Return every polygon point belonging to every key."""
        return np.concatenate(
            [polygon for key in self.keys for polygon in key.polygons],
            axis=0,
        )

    @property
    def bounding_box(self) -> np.ndarray:
        """
        Return the axis-aligned bounding box of the entire piano.

        The returned points are ordered as:
            top-left
            top-right
            bottom-right
            bottom-left
        """
        all_points = self.all_points

        min_x = all_points[:, 0].min()
        max_x = all_points[:, 0].max()
        min_y = all_points[:, 1].min()
        max_y = all_points[:, 1].max()

        return np.array(
            [
                [min_x, max_y],
                [max_x, max_y],
                [max_x, min_y],
                [min_x, min_y],
            ],
            dtype=np.float64,
        )


def _sample_cubic_bezier(
    start_point: np.ndarray,
    control_point_1: np.ndarray,
    control_point_2: np.ndarray,
    end_point: np.ndarray,
    sample_count: int = 16,
) -> np.ndarray:
    """
    Sample points along a cubic Bézier curve.

    A cubic Bézier curve is defined by four points:
    the start point, two control points, and the end point.

    For a parameter t between 0 and 1, the curve is:

        B(t) =
            (1-t)^3 P0
            + 3(1-t)^2 t P1
            + 3(1-t)t^2 P2
            + t^3 P3

    We evaluate that formula for several values of t. The result is
    a polygonal approximation of the curve.

    The ``[:, None]`` operations turn the 1D array of t values into
    a column vector so that NumPy broadcasts it against the x/y
    coordinates of the four control points.
    """
    parameter_values = np.linspace(
        0.0,
        1.0,
        sample_count,
    )

    t = parameter_values[:, None]
    one_minus_t = 1.0 - t

    return (
        one_minus_t**3 * start_point
        + 3 * one_minus_t**2 * t * control_point_1
        + 3 * one_minus_t * t**2 * control_point_2
        + t**3 * end_point
    )


def vmobject_to_polygon(
    vmobject: VMobject,
    samples_per_curve: int = 16,
) -> np.ndarray:
    """
    Convert a Manim VMobject into a polygonal 2D boundary.

    Manim stores paths as cubic Bézier curves. We sample each curve
    and concatenate those samples into one polygon.
    """
    cubic_bezier_curves = vmobject.get_cubic_bezier_tuples()

    sampled_curves = []

    for curve in cubic_bezier_curves:
        (
            start_point,
            control_point_1,
            control_point_2,
            end_point,
        ) = [np.asarray(point[:2], dtype=np.float64) for point in curve]

        sampled_points = _sample_cubic_bezier(
            start_point,
            control_point_1,
            control_point_2,
            end_point,
            sample_count=samples_per_curve,
        )

        # The last point of one Bézier segment is also the first point
        # of the next segment. Remove it here to avoid duplicating points.
        sampled_curves.append(sampled_points[:-1])

    if not sampled_curves:
        raise ValueError("VMobject contains no cubic Bézier curves.")

    return np.concatenate(sampled_curves, axis=0)


def extract_piano_geometry(
    piano: Piano,
    samples_per_curve: int = 16,
) -> PianoGeometry:
    """
    Extract the positioned geometry of a full Manim piano.

    Key.vmob contains each key in its local coordinate system. The
    positioned VMobjects returned by Piano.get_positioned_vmobs()
    contain the actual position of each key within the complete piano.
    """
    positioned_vmobjects = piano.get_positioned_vmobs(
        abs_start_idx=0,
        abs_end_idx=88,
        abs_positioning=False,
    )

    key_geometries: list[KeyGeometry] = []

    for key, positioned_vmob in zip(
        piano.keys,
        positioned_vmobjects,
    ):
        polygons = []

        for submob in positioned_vmob:
            polygon = vmobject_to_polygon(
                submob,
                samples_per_curve=samples_per_curve,
            )
            polygons.append(polygon)

        key_geometries.append(
            KeyGeometry(
                index=key.full_piano_idx,
                polygons=polygons,
            )
        )

    return PianoGeometry(keys=key_geometries)


def create_full_janko_geometry() -> PianoGeometry:
    """
    Create the canonical geometry of the complete 88-key Janko piano.

    This is the single source of truth for the synthetic data generator:
    the generator never recreates the piano geometry independently.
    """
    piano = Piano(
        create_janko_key,
        vgroup_start=0,
        vgroup_end=88,
        add_bounding_box=False,
    )

    return extract_piano_geometry(piano)
