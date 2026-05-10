"""Trajectory naming utilities."""

from __future__ import annotations


def _format_number(value: float | int, *, keep_decimal_for_int: bool = False) -> str:
    number = float(value)
    if number.is_integer() and not keep_decimal_for_int:
        text = str(int(number))
    else:
        text = f"{number:.3f}".rstrip("0").rstrip(".")
        if keep_decimal_for_int and "." not in text:
            text = f"{text}.0"
    return text.replace(".", "p")


def format_arc_trajectory_name(
    arc_type: str,
    arc_angle: float | int,
    arc_radius_scale: float | int | None = 1.0,
) -> str:
    """Return the canonical folder/json stem for an automatic arc trajectory."""
    scale = 1.0 if arc_radius_scale is None else float(arc_radius_scale)
    angle_text = _format_number(arc_angle)
    scale_text = _format_number(scale, keep_decimal_for_int=True)
    return f"arc_{arc_type}_{angle_text}_scale_{scale_text}"
