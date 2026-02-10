from __future__ import annotations

from math import cos, pi, sin, tau

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

    # Inner ring.
    ring = fc.arcXY(centre_z, inner_r, 0.0, tau, int(segs_inner))
    if ring:
        steps.extend(fc.travel_to(ring[0]))
        steps.extend(ring)

    # Pick start points on the ring (nearest by direction) to ensure connection.
    def pick_ring_start(end: fc.Point) -> fc.Point:
        if not ring:
            # Idealized start on the true circle.
            dx = float(end.x) - centre_z.x
            dy = float(end.y) - centre_z.y
            d = (dx * dx + dy * dy) ** 0.5
            if d <= 1e-9:
                return centre_z
            return fc.Point(x=centre_z.x + dx / d * inner_r, y=centre_z.y + dy / d * inner_r, z=centre_z.z)

        dx = float(end.x) - centre_z.x
        dy = float(end.y) - centre_z.y
        d = (dx * dx + dy * dy) ** 0.5
        if d <= 1e-9:
            return ring[0]
        ux, uy = dx / d, dy / d
        best = ring[0]
        best_dot = (float(best.x) - centre_z.x) * ux + (float(best.y) - centre_z.y) * uy
        for p in ring[1:]:
            dot = (float(p.x) - centre_z.x) * ux + (float(p.y) - centre_z.y) * uy
            if dot > best_dot:
                best_dot = dot
                best = p
        return best

    def add_arm_to_shell(idx: int) -> None:
        end_raw = shell_points[idx]
        end = fc.Point(x=float(end_raw.x), y=float(end_raw.y), z=float(z))

        start = pick_ring_start(end)
        dx = float(end.x) - float(start.x)
        dy = float(end.y) - float(start.y)
        d = (dx * dx + dy * dy) ** 0.5
        if d <= 1e-9:
            return

        ux = dx / d
        uy = dy / d
        perp_x = -uy
        perp_y = ux

        arm_pts: list[fc.Point] = []
        wave_n = max(1, int(wave_count))
        for i in range(int(segs_arm) + 1):
            t = i / float(segs_arm)
            base_x = float(start.x) + dx * t
            base_y = float(start.y) + dy * t

            envelope = sin(pi * t)  # 0 at ends.
            desired_s = float(amp) * envelope * sin(wave_n * 2.0 * pi * t)

            if clamp_enabled:
                s = _clamp_along_perp_to_bbox(
                    base=fc.Point(x=base_x, y=base_y, z=centre_z.z),
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

            arm_pts.append(fc.Point(x=base_x + perp_x * s, y=base_y + perp_y * s, z=centre_z.z))

        if not arm_pts:
            return

        steps.extend(fc.travel_to(arm_pts[0]))
        steps.extend(arm_pts)

        # Embed into the shell path for stronger merge.
        embed_n = max(0, int(embed_segments))
        for k in range(1, embed_n + 1):
            p = shell_points[(idx + k) % n]
            steps.append(fc.Point(x=float(p.x), y=float(p.y), z=float(z)))

    # Print all 4 arms.
    for _, idx in targets:
        add_arm_to_shell(idx)
