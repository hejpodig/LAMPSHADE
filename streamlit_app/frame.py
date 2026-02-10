from __future__ import annotations

from math import cos, pi, tau

from dataclasses import dataclass

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


def add_cardinal_frame(
    *,
    steps: list,
    centre: fc.Point,
    z: float,
    frame_rad_inner: float,
    extent_east: float,
    extent_west: float,
    extent_north: float,
    extent_south: float,
    ew: float,
    eh: float,
    print_speed: float,
    frame_width_factor: float = 2.5,
    layer_ratio: int = 2,
    segs_inner: int = 96,
) -> None:
    """Append a centered 4-arm frame aligned to north/south/east/west.

    The arms terminate at the shape's furthest extents in each cardinal
    direction, so the frame doesn't protrude outside the silhouette.
    """

    centre_z = fc.Point(x=float(centre.x), y=float(centre.y), z=float(z))
    inner_r = max(0.0, float(frame_rad_inner))

    east = max(inner_r, float(extent_east))
    west = max(inner_r, float(extent_west))
    north = max(inner_r, float(extent_north))
    south = max(inner_r, float(extent_south))

    steps.append(fc.ExtrusionGeometry(width=float(ew) * float(frame_width_factor), height=float(eh) * int(layer_ratio)))
    steps.append(fc.Printer(print_speed=float(print_speed) / (float(frame_width_factor) * int(layer_ratio))))

    # Inner ring around the hole.
    inner_ring = fc.arcXY(centre_z, inner_r, 0.0, tau, int(segs_inner))
    if len(inner_ring) > 0:
        steps.extend(fc.travel_to(inner_ring[0]))
        steps.extend(inner_ring)

    # Cardinal arms: travel to the inner ring point, then extrude outwards.
    arm_specs = [
        (fc.Point(x=centre_z.x, y=centre_z.y + inner_r, z=centre_z.z), fc.Point(x=centre_z.x, y=centre_z.y + north, z=centre_z.z)),
        (fc.Point(x=centre_z.x, y=centre_z.y - inner_r, z=centre_z.z), fc.Point(x=centre_z.x, y=centre_z.y - south, z=centre_z.z)),
        (fc.Point(x=centre_z.x + inner_r, y=centre_z.y, z=centre_z.z), fc.Point(x=centre_z.x + east, y=centre_z.y, z=centre_z.z)),
        (fc.Point(x=centre_z.x - inner_r, y=centre_z.y, z=centre_z.z), fc.Point(x=centre_z.x - west, y=centre_z.y, z=centre_z.z)),
    ]

    for p_inner, p_outer in arm_specs:
        steps.extend(fc.travel_to(p_inner))
        steps.append(p_outer)


@dataclass(frozen=True)
class CardinalEndpoints:
    east: fc.Point
    west: fc.Point
    north: fc.Point
    south: fc.Point


def _clamp_along_perp_to_bbox(
    *,
    base: fc.Point,
    perp_x: float,
    perp_y: float,
    desired_s: float,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
) -> float:
    s_min = float("-inf")
    s_max = float("inf")

    # base.x + perp_x*s <= max_x
    if perp_x > 1e-12:
        s_max = min(s_max, (max_x - base.x) / perp_x)
    elif perp_x < -1e-12:
        s_min = max(s_min, (max_x - base.x) / perp_x)
    # base.x + perp_x*s >= min_x
    if perp_x > 1e-12:
        s_min = max(s_min, (min_x - base.x) / perp_x)
    elif perp_x < -1e-12:
        s_max = min(s_max, (min_x - base.x) / perp_x)

    # base.y + perp_y*s <= max_y
    if perp_y > 1e-12:
        s_max = min(s_max, (max_y - base.y) / perp_y)
    elif perp_y < -1e-12:
        s_min = max(s_min, (max_y - base.y) / perp_y)
    # base.y + perp_y*s >= min_y
    if perp_y > 1e-12:
        s_min = max(s_min, (min_y - base.y) / perp_y)
    elif perp_y < -1e-12:
        s_max = min(s_max, (min_y - base.y) / perp_y)

    if s_min > s_max:
        return 0.0
    return max(s_min, min(s_max, desired_s))


def add_patterned_cardinal_frame(
    *,
    steps: list,
    centre: fc.Point,
    z: float,
    frame_rad_inner: float,
    bbox_min_x: float,
    bbox_max_x: float,
    bbox_min_y: float,
    bbox_max_y: float,
    endpoints: CardinalEndpoints,
    ew: float,
    eh: float,
    print_speed: float,
    frame_width_factor: float = 2.5,
    layer_ratio: int = 2,
    segs_inner: int = 96,
    segs_arm: int = 80,
    amp: float = 12.0,
) -> None:
    """Append a centered patterned 4-arm frame aligned to N/S/E/W.

    - Uses an inner ring around the hole.
    - Each arm is a wavy curve that starts/ends exactly on the inner ring and
      the supplied endpoint.
    - Every point is clamped to the supplied bounding box (typically inset by
      half the extrusion width), so the frame does not protrude.
    """

    centre_z = fc.Point(x=float(centre.x), y=float(centre.y), z=float(z))

    # Ensure bbox is sane.
    min_x = float(min(bbox_min_x, bbox_max_x))
    max_x = float(max(bbox_min_x, bbox_max_x))
    min_y = float(min(bbox_min_y, bbox_max_y))
    max_y = float(max(bbox_min_y, bbox_max_y))

    inner_r = max(0.0, float(frame_rad_inner))
    ew_eff = float(ew) * float(frame_width_factor)
    eh_eff = float(eh) * int(layer_ratio)

    steps.append(fc.ExtrusionGeometry(width=ew_eff, height=eh_eff))
    steps.append(fc.Printer(print_speed=float(print_speed) / (float(frame_width_factor) * int(layer_ratio))))

    # Inner ring (clamped to bbox, although it should already be inside).
    inner_ring = fc.arcXY(centre_z, inner_r, 0.0, tau, int(segs_inner))
    if len(inner_ring) > 0:
        steps.extend(fc.travel_to(inner_ring[0]))
        for p in inner_ring:
            steps.append(
                fc.Point(
                    x=max(min_x, min(max_x, p.x)),
                    y=max(min_y, min(max_y, p.y)),
                    z=p.z,
                )
            )

    # Arm helper.
    def add_arm(end: fc.Point) -> None:
        dx = float(end.x) - float(centre_z.x)
        dy = float(end.y) - float(centre_z.y)
        d = (dx * dx + dy * dy) ** 0.5
        if d <= 1e-9:
            return
        ux = dx / d
        uy = dy / d
        start = fc.Point(x=centre_z.x + ux * inner_r, y=centre_z.y + uy * inner_r, z=centre_z.z)

        # Perp unit.
        perp_x = -uy
        perp_y = ux

        # Cap amplitude relative to bbox.
        max_amp = 0.45 * min(max_x - min_x, max_y - min_y)
        arm_amp = max(0.0, min(float(amp), max_amp))

        pts: list[fc.Point] = []
        for i in range(int(segs_arm) + 1):
            t = i / float(segs_arm)
            base_x = float(start.x) + (float(end.x) - float(start.x)) * t
            base_y = float(start.y) + (float(end.y) - float(start.y)) * t
            base = fc.Point(x=base_x, y=base_y, z=centre_z.z)

            desired_s = (0.5 - 0.5 * cos(2.0 * pi * t)) * arm_amp
            # Alternate side for some texture.
            desired_s *= cos(pi * t)
            s = _clamp_along_perp_to_bbox(
                base=base,
                perp_x=perp_x,
                perp_y=perp_y,
                desired_s=desired_s,
                min_x=min_x,
                max_x=max_x,
                min_y=min_y,
                max_y=max_y,
            )
            pts.append(fc.Point(x=base.x + perp_x * s, y=base.y + perp_y * s, z=base.z))

        if len(pts) > 0:
            steps.extend(fc.travel_to(pts[0]))
            steps.extend(pts)

    add_arm(endpoints.north)
    add_arm(endpoints.south)
    add_arm(endpoints.east)
    add_arm(endpoints.west)
