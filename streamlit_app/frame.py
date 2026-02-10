from __future__ import annotations

from math import atan2, cos, pi, sin, tau

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


def _clamp_point_to_bbox_ray(
    *,
    p: fc.Point,
    centre: fc.Point,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
) -> fc.Point:
    """Clamp a point to an axis-aligned bbox by scaling the ray from centre->p.

    This preserves the point's direction from the centre better than clamping
    x/y independently.
    """

    dx = float(p.x) - float(centre.x)
    dy = float(p.y) - float(centre.y)

    # If already inside (or degenerate), keep as-is.
    if (min_x <= p.x <= max_x) and (min_y <= p.y <= max_y):
        return p
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return fc.Point(x=max(min_x, min(max_x, float(p.x))), y=max(min_y, min(max_y, float(p.y))), z=float(p.z))

    t_candidates: list[float] = [1.0]

    if abs(dx) >= 1e-12:
        t_candidates.append((max_x - float(centre.x)) / dx)
        t_candidates.append((min_x - float(centre.x)) / dx)
    if abs(dy) >= 1e-12:
        t_candidates.append((max_y - float(centre.y)) / dy)
        t_candidates.append((min_y - float(centre.y)) / dy)

    # Choose the largest t in [0,1] that puts us inside.
    t_best = 0.0
    for t in t_candidates:
        if not (0.0 <= t <= 1.0):
            continue
        x = float(centre.x) + dx * t
        y = float(centre.y) + dy * t
        if (min_x - 1e-9) <= x <= (max_x + 1e-9) and (min_y - 1e-9) <= y <= (max_y + 1e-9):
            t_best = max(t_best, t)

    return fc.Point(x=float(centre.x) + dx * t_best, y=float(centre.y) + dy * t_best, z=float(p.z))


def add_legacy_pattern_frame_clamped(
    *,
    steps: list,
    centre: fc.Point,
    z: float,
    frame_rad_inner: float,
    frame_rad_max: float,
    bbox_min_x: float,
    bbox_max_x: float,
    bbox_min_y: float,
    bbox_max_y: float,
    ew: float,
    eh: float,
    print_speed: float,
    contact_points: int = 4,
    amp: float = 17.5,
    frame_width_factor: float = 2.5,
    frame_line_spacing_ratio: float = 0.2,
    layer_ratio: int = 2,
    segs_frame: int = 64,
    start_angle: float = -pi / 4,
) -> None:
    """Legacy lampshade frame pattern (wave + arc + reflect + rotate), clamped.

    This matches the original lampshade frame pattern as closely as possible,
    while ensuring no point protrudes outside the per-layer bounds.
    """

    contact_points = max(1, int(contact_points))
    centre_z = fc.Point(x=float(centre.x), y=float(centre.y), z=float(z))
    frame_rad_inner = max(0.0, float(frame_rad_inner))
    frame_rad_max = max(frame_rad_inner, float(frame_rad_max))

    min_x = float(min(bbox_min_x, bbox_max_x))
    max_x = float(max(bbox_min_x, bbox_max_x))
    min_y = float(min(bbox_min_y, bbox_max_y))
    max_y = float(max(bbox_min_y, bbox_max_y))

    t_steps_frame_line = fc.linspace(0, 1, int(segs_frame) + 1)

    wave_steps: list[fc.Point] = []
    for t_now in t_steps_frame_line:
        x_now = centre_z.x + (frame_line_spacing_ratio * (frame_width_factor * float(ew))) + (float(amp) * t_now) * (
            (0.5 - 0.5 * cos((t_now**0.66) * 3 * tau)) ** 1
        )
        y_now = centre_z.y - frame_rad_inner - ((frame_rad_max - frame_rad_inner) * (1 - t_now))
        wave_steps.append(fc.Point(x=float(x_now), y=float(y_now), z=float(z)))

    wave_steps.extend(
        fc.arcXY(
            centre_z,
            frame_rad_inner,
            float(start_angle),
            pi / contact_points,
            int(64 / contact_points),
        )
    )

    wave_steps.extend(_reflectXYpolar_list(wave_steps, centre_z, float(start_angle) + pi / contact_points))
    wave_steps = fc.move_polar(wave_steps, centre_z, 0, tau / contact_points, copy=True, copy_quantity=contact_points)

    # Clamp points to the per-layer bbox.
    clamped_steps: list[fc.Point] = [
        _clamp_point_to_bbox_ray(p=p, centre=centre_z, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y) for p in wave_steps
    ]

    steps.append(fc.ExtrusionGeometry(width=float(ew) * float(frame_width_factor), height=float(eh) * int(layer_ratio)))
    steps.append(fc.Printer(print_speed=float(print_speed) / (float(frame_width_factor) * int(layer_ratio))))
    steps.extend(clamped_steps)


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


def add_nsew_wave_frame_connected_to_shell(
    *,
    steps: list,
    centre: fc.Point,
    z: float,
    shell_points: list[fc.Point],
    frame_rad_inner: float,
    ew: float,
    eh: float,
    print_speed: float,
    frame_width_factor: float = 2.5,
    layer_ratio: int = 2,
    segs_inner: int = 128,
    segs_arm: int = 96,
    amp: float = 12.0,
    wave_count: int = 3,
    embed_segments: int = 2,
) -> None:
    """Append a 4-contact (N/S/E/W) frame that is *connected* to the shell.

    Key properties:
    - Hole is always centered on `centre` (inner ring around `frame_rad_inner`).
    - Contact points come from the actual `shell_points` at the same Z (furthest
      north/south/east/west), so endpoints sit on the shell.
    - Each arm uses exactly `wave_count` waves; shorter arms compress the waves.
    - Adds a short "embed" along the shell path after the endpoint so the frame
      deposits overlap with the shell path (reduces visible gaps).
    """

    if not shell_points:
        return

    centre_z = fc.Point(x=float(centre.x), y=float(centre.y), z=float(z))
    n = len(shell_points)
    if n < 4:
        return

    # Ensure frame inner radius accounts for half line width so the printed bead
    # doesn't intrude into the hole.
    inner_r = max(0.0, float(frame_rad_inner))
    ew_eff = float(ew) * float(frame_width_factor)
    eh_eff = float(eh) * int(layer_ratio)

    # Coarse bounds (axis-aligned) for clamping wave offsets.
    min_x = min(float(p.x) for p in shell_points) + 0.5 * ew_eff
    max_x = max(float(p.x) for p in shell_points) - 0.5 * ew_eff
    min_y = min(float(p.y) for p in shell_points) + 0.5 * ew_eff
    max_y = max(float(p.y) for p in shell_points) - 0.5 * ew_eff

    # If the shell is very small (or ew_eff is huge), fall back to unclamped.
    clamp_enabled = (min_x <= max_x) and (min_y <= max_y)

    def _argmax(points: list[fc.Point], key) -> int:
        best_i = 0
        best_v = key(points[0])
        for i in range(1, len(points)):
            v = key(points[i])
            if v > best_v:
                best_v = v
                best_i = i
        return best_i

    def _argmin(points: list[fc.Point], key) -> int:
        best_i = 0
        best_v = key(points[0])
        for i in range(1, len(points)):
            v = key(points[i])
            if v < best_v:
                best_v = v
                best_i = i
        return best_i

    idx_n = _argmax(shell_points, lambda p: float(p.y))
    idx_s = _argmin(shell_points, lambda p: float(p.y))
    idx_e = _argmax(shell_points, lambda p: float(p.x))
    idx_w = _argmin(shell_points, lambda p: float(p.x))

    targets: list[tuple[str, int]] = [
        ("north", idx_n),
        ("south", idx_s),
        ("east", idx_e),
        ("west", idx_w),
    ]

    steps.append(fc.ExtrusionGeometry(width=ew_eff, height=eh_eff))
    steps.append(fc.Printer(print_speed=float(print_speed) / (float(frame_width_factor) * int(layer_ratio))))

    # Drop the closing point if the shell is explicitly closed.
    if n >= 2:
        p0 = shell_points[0]
        p_last = shell_points[-1]
        if abs(float(p0.x) - float(p_last.x)) < 1e-9 and abs(float(p0.y) - float(p_last.y)) < 1e-9:
            shell_points = shell_points[:-1]
            n = len(shell_points)
            if n < 4:
                return

    def _one_sided_wave(
        *,
        start: fc.Point,
        end: fc.Point,
        perp_x: float,
        perp_y: float,
        sign: float,
    ) -> list[fc.Point]:
        dx = float(end.x) - float(start.x)
        dy = float(end.y) - float(start.y)
        if (dx * dx + dy * dy) ** 0.5 <= 1e-9:
            return []

        pts: list[fc.Point] = []
        wave_n = max(1, int(wave_count))
        for i in range(int(segs_arm) + 1):
            t = i / float(segs_arm)
            base_x = float(start.x) + dx * t
            base_y = float(start.y) + dy * t
            base = fc.Point(x=base_x, y=base_y, z=centre_z.z)

            # One-sided waveform: always >= 0. Exactly `wave_n` bumps.
            bumps = 0.5 - 0.5 * cos((t**0.66) * wave_n * 2.0 * pi)
            envelope = sin(pi * t)  # forces 0 at both ends
            desired_s = float(sign) * float(amp) * envelope * bumps

            if clamp_enabled:
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
            else:
                s = desired_s

            pts.append(fc.Point(x=base.x + perp_x * s, y=base.y + perp_y * s, z=base.z))

        return pts

    def add_arm_to_shell(idx: int) -> None:
        end_raw = shell_points[idx]
        end = fc.Point(x=float(end_raw.x), y=float(end_raw.y), z=float(z))

        # Radial axis (for mirroring) from the centre to the shell contact.
        theta = atan2(float(end.y) - float(centre_z.y), float(end.x) - float(centre_z.x))
        ux = cos(theta)
        uy = sin(theta)
        perp_x = -uy
        perp_y = ux

        # Use the legacy sector half-angle so each arm has two mirrored wave sides.
        beta = pi / 4.0
        inner_a = fc.polar_to_point(centre_z, inner_r, theta - beta)
        inner_b = fc.polar_to_point(centre_z, inner_r, theta + beta)

        wave_a = _one_sided_wave(start=end, end=inner_a, perp_x=perp_x, perp_y=perp_y, sign=+1.0)
        if not wave_a:
            return

        wave_b_e_to_inner = _one_sided_wave(start=end, end=inner_b, perp_x=perp_x, perp_y=perp_y, sign=-1.0)
        if not wave_b_e_to_inner:
            return
        wave_b = list(reversed(wave_b_e_to_inner))

        # Inner arc connecting the mirrored waves (together, arms cover the full ring).
        arc_len = 2.0 * beta
        segs_arc = max(8, int(float(segs_inner) * (arc_len / tau)))
        inner_arc = fc.arcXY(centre_z, inner_r, theta - beta, arc_len, segs_arc)

        steps.extend(fc.travel_to(wave_a[0]))
        steps.extend(wave_a)
        if inner_arc:
            steps.extend(inner_arc)
        steps.extend(wave_b)

        # Embed into the shell path for stronger merge at the contact.
        embed_n = max(0, int(embed_segments))
        for k in range(1, embed_n + 1):
            p = shell_points[(idx + k) % n]
            steps.append(fc.Point(x=float(p.x), y=float(p.y), z=float(z)))

    # Print all 4 arms.
    for _, idx in targets:
        add_arm_to_shell(idx)


def add_cardinal_three_arc_frame(
    *,
    steps: list,
    centre: fc.Point,
    z: float,
    shell_points: list[fc.Point],
    frame_rad_inner: float,
    ew: float,
    eh: float,
    print_speed: float,
    frame_width_factor: float = 2.5,
    layer_ratio: int = 2,
    segs_arm: int = 120,
    segs_inner_quarter: int = 48,
    amp: float = 17.5,
    embed_segments: int = 2,
) -> None:
    """Cardinal (S->E->N->W) frame with exactly 3 arches per arm side.

    Matches the requested sequencing:
    - Start printing at the SOUTH outer contact point.
    - For each arm: 3 arches outward + 3 arches back (double lines).
    - Between arms: inner circle quarter-turn (pi/2).

    The outer endpoints come from actual `shell_points` at the same Z.
    """

    if not shell_points:
        return

    n = len(shell_points)
    if n < 8:
        return

    centre_z = fc.Point(x=float(centre.x), y=float(centre.y), z=float(z))

    inner_r = max(0.0, float(frame_rad_inner))
    ew_eff = float(ew) * float(frame_width_factor)
    eh_eff = float(eh) * int(layer_ratio)

    # Bounds for clamping the wave offsets.
    min_x = min(float(p.x) for p in shell_points) + 0.5 * ew_eff
    max_x = max(float(p.x) for p in shell_points) - 0.5 * ew_eff
    min_y = min(float(p.y) for p in shell_points) + 0.5 * ew_eff
    max_y = max(float(p.y) for p in shell_points) - 0.5 * ew_eff
    clamp_enabled = (min_x <= max_x) and (min_y <= max_y)

    def _argmax(points: list[fc.Point], key) -> int:
        best_i = 0
        best_v = key(points[0])
        for i in range(1, len(points)):
            v = key(points[i])
            if v > best_v:
                best_v = v
                best_i = i
        return best_i

    def _argmin(points: list[fc.Point], key) -> int:
        best_i = 0
        best_v = key(points[0])
        for i in range(1, len(points)):
            v = key(points[i])
            if v < best_v:
                best_v = v
                best_i = i
        return best_i

    # The user-specified start point is the shell loop start/end for this layer.
    # Use it as the SOUTH arm contact, regardless of whether it's the absolute
    # minimum-y point.
    idx_s = 0
    idx_e = _argmax(shell_points, lambda p: float(p.x))
    idx_n = _argmax(shell_points, lambda p: float(p.y))
    idx_w = _argmin(shell_points, lambda p: float(p.x))

    # Order is S -> E -> N -> W.
    arms: list[tuple[str, int]] = [("south", idx_s), ("east", idx_e), ("north", idx_n), ("west", idx_w)]

    def _unit_from_centre(p: fc.Point) -> tuple[float, float, float]:
        dx = float(p.x) - float(centre_z.x)
        dy = float(p.y) - float(centre_z.y)
        theta = atan2(dy, dx)
        return cos(theta), sin(theta), theta

    def _inner_point(theta: float) -> fc.Point:
        return fc.polar_to_point(centre_z, inner_r, theta)

    def _embed_from(idx: int) -> None:
        embed_n = max(0, int(embed_segments))
        for k in range(1, embed_n + 1):
            p = shell_points[(idx + k) % n]
            steps.append(fc.Point(x=float(p.x), y=float(p.y), z=float(z)))

    def _diag_scale_for_bump3(*, theta: float, start: fc.Point, end: fc.Point, perp_x: float, perp_y: float) -> float:
        """Scale factor so bump #3 peak on the +perp side reaches the diagonal to the next arm.

        We target the diagonal ray at angle theta + pi/4.
        """

        wave_n = 3
        t_peak3 = (wave_n - 0.5) / wave_n  # 2.5/3
        bx = float(start.x) + (float(end.x) - float(start.x)) * t_peak3
        by = float(start.y) + (float(end.y) - float(start.y)) * t_peak3
        b = (bx - float(centre_z.x), by - float(centre_z.y))

        diag_theta = float(theta) + (pi / 4.0)
        u_dx = cos(diag_theta)
        u_dy = sin(diag_theta)

        # Solve det(u_d, b + perp*s) = 0  ->  s = -det(u_d, b)/det(u_d, perp)
        det_ud_b = (u_dx * b[1]) - (u_dy * b[0])
        det_ud_perp = (u_dx * perp_y) - (u_dy * perp_x)
        if abs(det_ud_perp) < 1e-9:
            return 1.0
        s_needed = -det_ud_b / det_ud_perp
        if s_needed <= 0.0:
            return 1.0

        envelope_peak = sin(pi * t_peak3)
        if envelope_peak <= 1e-6:
            return 1.0

        base_amp = float(amp) * envelope_peak
        if base_amp <= 1e-9:
            return 1.0

        scale = s_needed / base_amp
        return max(1.0, min(3.0, float(scale)))

    def _three_arch_wave(
        *,
        start: fc.Point,
        end: fc.Point,
        perp_x: float,
        perp_y: float,
        sign: float,
        bump3_scale: float,
    ) -> list[fc.Point]:
        dx = float(end.x) - float(start.x)
        dy = float(end.y) - float(start.y)
        d = (dx * dx + dy * dy) ** 0.5
        if d <= 1e-9:
            return []

        wave_n = 3
        bump_scales = [0.5, 1.0, float(bump3_scale)]

        pts: list[fc.Point] = []
        for i in range(int(segs_arm) + 1):
            t = i / float(segs_arm)
            base_x = float(start.x) + dx * t
            base_y = float(start.y) + dy * t
            base = fc.Point(x=base_x, y=base_y, z=centre_z.z)

            bump_pos = t * wave_n
            bump_idx = int(min(wave_n - 1, max(0, int(bump_pos))))
            u = bump_pos - bump_idx
            # 0..1 bump shape, peak at u=0.5
            bump = 0.5 - 0.5 * cos(2.0 * pi * u)

            envelope = sin(pi * t)
            desired_s = float(sign) * float(amp) * envelope * bump * bump_scales[bump_idx]

            if clamp_enabled:
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
            else:
                s = desired_s

            pts.append(fc.Point(x=base.x + perp_x * s, y=base.y + perp_y * s, z=base.z))

        return pts

    steps.append(fc.ExtrusionGeometry(width=ew_eff, height=eh_eff))
    steps.append(fc.Printer(print_speed=float(print_speed) / (float(frame_width_factor) * int(layer_ratio))))

    # Start at SOUTH outer contact (print starts here).
    south_outer = shell_points[idx_s]
    steps.extend(fc.travel_to(fc.Point(x=float(south_outer.x), y=float(south_outer.y), z=float(z))))

    # First: go from SOUTH outer to SOUTH inner (this is the "back" line for the first arm).
    ux, uy, theta_s = _unit_from_centre(south_outer)
    perp_x, perp_y = -uy, ux  # +perp points toward the next arm in our order
    inner_s = _inner_point(theta_s)
    bump3_scale_s = _diag_scale_for_bump3(theta=theta_s, start=fc.Point(x=float(south_outer.x), y=float(south_outer.y), z=float(z)), end=inner_s, perp_x=perp_x, perp_y=perp_y)
    first_in = _three_arch_wave(
        start=fc.Point(x=float(south_outer.x), y=float(south_outer.y), z=float(z)),
        end=inner_s,
        perp_x=perp_x,
        perp_y=perp_y,
        sign=-1.0,
        bump3_scale=1.0,
    )
    steps.extend(first_in)

    # Now do all arms with: inner->outer (toward next) + embed + outer->inner (opposite), with a quarter ring in between.
    for arm_i, (_, idx_outer) in enumerate(arms):
        outer = shell_points[idx_outer]
        ux, uy, theta = _unit_from_centre(outer)
        perp_x, perp_y = -uy, ux
        inner = _inner_point(theta)

        # Ensure we're at the correct inner point before heading out.
        steps.append(fc.Point(x=float(inner.x), y=float(inner.y), z=float(z)))

        bump3_scale = _diag_scale_for_bump3(theta=theta, start=inner, end=fc.Point(x=float(outer.x), y=float(outer.y), z=float(z)), perp_x=perp_x, perp_y=perp_y)
        out_pts = _three_arch_wave(
            start=inner,
            end=fc.Point(x=float(outer.x), y=float(outer.y), z=float(z)),
            perp_x=perp_x,
            perp_y=perp_y,
            sign=+1.0,
            bump3_scale=bump3_scale,
        )
        steps.extend(out_pts)
        _embed_from(idx_outer)

        back_pts = _three_arch_wave(
            start=fc.Point(x=float(outer.x), y=float(outer.y), z=float(z)),
            end=inner,
            perp_x=perp_x,
            perp_y=perp_y,
            sign=-1.0,
            bump3_scale=1.0,
        )
        steps.extend(back_pts)

        # Quarter turn on the inner circle to the next arm (skip after the last).
        if arm_i < len(arms) - 1:
            _, idx_next_outer = arms[arm_i + 1]
            next_outer = shell_points[idx_next_outer]
            _, _, next_theta = _unit_from_centre(next_outer)
            arc = fc.arcXY(centre_z, inner_r, theta, pi / 2.0, int(segs_inner_quarter))
            if arc:
                steps.extend(arc)
