from __future__ import annotations

from math import cos, pi, tau

import fullcontrol as fc


def _reflectXYpolar_list(steps: list[fc.Point], p_reflect: fc.Point, angle_reflect: float) -> list[fc.Point]:
    new_steplist: list[fc.Point] = []
    for i in range(len(steps)):
        step_now = (len(steps) - 1) - i
        if type(steps[step_now]).__name__ != "Point":
            raise Exception(
                f"list of steps contained a {type(steps[step_now]).__name__}. only Points can be included in the list being reflected for now. Other types of objects needs careful consideration in terms of sequencing."
            )
        new_steplist.append(fc.reflectXYpolar(steps[step_now], p_reflect, angle_reflect))
    return new_steplist


def add_lampshade_frame(
    *,
    steps: list,
    centre: fc.Point,
    z: float,
    frame_rad_inner: float,
    frame_rad_max: float,
    ew: float,
    eh: float,
    print_speed: float,
    contact_points: int = 4,
    amp: float = 17.5,
    frame_width_factor: float = 2.5,
    frame_line_spacing_ratio: float = 0.2,
    layer_ratio: int = 2,
    segs_frame: int = 64,
    start_angle: float = 0.75 * tau,
) -> None:
    """Append the lampshade-style inner frame path.

    This is a direct extraction of the frame logic used by the lampshade model,
    with the only behavioral change being the number of repeated sectors
    ("contact_points").
    """

    contact_points = max(1, int(contact_points))
    t_steps_frame_line = fc.linspace(0, 1, int(segs_frame) + 1)

    wave_steps: list[fc.Point] = []
    for t_now in t_steps_frame_line:
        x_now = centre.x + (frame_line_spacing_ratio * (frame_width_factor * ew)) + (amp * t_now) * (
            (0.5 - 0.5 * cos((t_now**0.66) * 3 * tau)) ** 1
        )
        y_now = centre.y - frame_rad_inner - ((frame_rad_max - frame_rad_inner) * (1 - t_now))
        wave_steps.append(fc.Point(x=x_now, y=y_now, z=z))

    wave_steps.extend(
        fc.arcXY(
            centre,
            frame_rad_inner,
            start_angle,
            pi / contact_points,
            int(64 / contact_points),
        )
    )
    wave_steps.extend(_reflectXYpolar_list(wave_steps, centre, start_angle + pi / contact_points))
    wave_steps = fc.move_polar(wave_steps, centre, 0, tau / contact_points, copy=True, copy_quantity=contact_points)

    steps.append(fc.ExtrusionGeometry(width=ew * frame_width_factor, height=eh * layer_ratio))
    steps.append(fc.Printer(print_speed=print_speed / (frame_width_factor * layer_ratio)))
    steps.extend(wave_steps)
