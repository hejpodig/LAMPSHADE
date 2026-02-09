from __future__ import annotations

from dataclasses import dataclass
from math import cos, exp, pi, sin, tau

import fullcontrol as fc
import lab.fullcontrol as fclab


@dataclass(frozen=True)
class LampshadeParams:
    Output: str = "Detailed Plot"  # Simple Plot | Detailed Plot | GCode
    Annotations: bool = True
    Printer_name: str = "generic"
    Design_name: str = "fc_lampshade"
    Nozzle_temp: int = 220
    Bed_temp: int = 40
    Fan_percent: int = 100
    Material_flow_percent: int = 100
    Print_speed_percent: int = 200

    Height: int = 150
    Nominal_radius: int = 34
    Tip_length: int = 20
    Star_tips: int = 6
    Main_bulge: float = 22.5
    Secondary_bulges: float = 15.0
    Secondary_bulge_count: int = 2
    Twist_turns: float = 0.0

    Inner_frame_hole_diameter: int = 30
    Inner_frame_height: int = 3
    Inner_frame_wave_amplitude: float = 17.5
    Centre_XY: int = 104

    zag_min: float = 1.0
    zag_max: float = 5.0
    zigzag_freq_factor: float = 1.0
    zigzag_radius_factor: float = 1.0
    zigzag_rounding_radius: int = 0

    # Advanced parameters (kept aligned to the notebook)
    rip_depth: float = 0.5
    rip_freq: int = 30
    swerve: float = 0.02
    segs_shell: int = 300
    x_1: float = 6
    x_2: float = 30
    frame_width_factor: float = 2.5
    frame_line_spacing_ratio: float = 0.2
    layer_ratio: int = 2
    start_angle: float = 0.75 * tau
    frame_overlap: float = 2.5
    segs_frame: int = 64
    EH: float = 0.2
    EW: float = 0.5
    initial_print_speed: int = 500
    main_print_speed: int = 1500
    speedchange_layers: int = 5
    initial_z_factor: float = 0.7

    viewer_point_stride: int = 1
    viewer_layer_stride: int = 1


def build_lampshade_steps(params: LampshadeParams):
    target = "visualize" if params.Output in ["Detailed Plot", "Simple Plot"] else "gcode"

    height, r_0, tip_len, n_tip = params.Height, params.Nominal_radius, params.Tip_length, params.Star_tips
    bulge1, bulge2 = params.Main_bulge, params.Secondary_bulges
    bulge2_count = max(0, int(params.Secondary_bulge_count))
    frame_rad_inner = params.Inner_frame_hole_diameter / 2
    frame_height = params.Inner_frame_height
    amp_1 = params.Inner_frame_wave_amplitude
    centre_xy = params.Centre_XY

    frame_rad_max = r_0 + tip_len + params.frame_overlap
    frame_rad_inner += params.EW / 2

    EH = float(params.EH)
    EW = float(params.EW)
    segs_shell = int(params.segs_shell)

    if params.Output == "Simple Plot":
        EH, segs_shell = EH * 30, max(1, n_tip * 20)
    elif params.Output == "Detailed Plot":
        EH = EH * 10

    shell_layers = int(height / EH)
    frame_layers = int(frame_height / EH) if frame_height > 0 else 0
    initial_z = EH * params.initial_z_factor

    steps = []
    segs_shell_wave = int(segs_shell)

    _round_t = max(0.0, min(10.0, float(params.zigzag_rounding_radius))) / 10.0
    _zigzag_cycles = max(1e-6, (segs_shell_wave / 2.0) * float(params.zigzag_freq_factor))
    _desired_pts_per_cycle = 2.0 + (_round_t * 10.0)
    segs_shell_samples = int(max(20.0, max(float(segs_shell), _zigzag_cycles * _desired_pts_per_cycle)))
    t_steps_shell = fc.linspace(0, 1, segs_shell_samples + 1)
    t_steps_frame_line = fc.linspace(0, 1, int(params.segs_frame) + 1)

    for layer in range(shell_layers):
        if target == "visualize" and params.viewer_layer_stride > 1 and layer != 0 and (layer % params.viewer_layer_stride != 0):
            continue

        if layer <= params.speedchange_layers and params.speedchange_layers > 0:
            print_speed = params.initial_print_speed + (params.main_print_speed - params.initial_print_speed) * (
                layer / params.speedchange_layers
            )
        else:
            print_speed = params.main_print_speed

        z_now = initial_z + layer * EH
        z_fraction = z_now / height
        twist_angle = (tau * float(params.Twist_turns) * z_fraction) if params.Twist_turns else 0.0
        centre_now = fc.Point(x=centre_xy, y=centre_xy, z=z_now)
        shell_steps, wave_steps = [], []

        for t_now in t_steps_shell[: int((segs_shell_samples / max(n_tip, 1)) / 2) + 1]:
            a_now = params.start_angle + (tau * t_now)
            angular_swerve = -(
                (params.swerve * tau * sin(t_now * n_tip * tau + (tau / 2)))
                * (
                    ((1 / (1 + exp(params.x_1 - z_fraction * params.x_2))) * (1 / (1 + exp(params.x_1 - (1 - z_fraction) * params.x_2))))
                    - (0.5 * (sin(z_fraction * 0.5 * tau)) ** 20)
                )
            )
            star_shape_wave = tip_len * (0.5 + 0.5 * (cos(t_now * n_tip * tau))) ** 2.5
            primary_z_wave = bulge1 * (sin(z_fraction * 0.5 * tau)) ** 1
            if bulge2_count <= 0:
                secondary_z_waves = 0
            else:
                secondary_z_waves = bulge2 * (0.5 + 0.5 * (cos((z_fraction + 0.15) * float(bulge2_count) * tau))) ** 1.5

            _phase = t_now * _zigzag_cycles
            _frac = _phase - int(_phase)
            _tri = 1.0 - abs((2.0 * _frac) - 1.0)
            _rcos = 0.5 - (0.5 * cos(2.0 * pi * _phase))
            zigzag_base = ((1.0 - _round_t) * _tri) + (_round_t * _rcos)

            zigzag_depth = params.zag_min + (params.zag_max * (0.5 + 0.5 * (cos(t_now * n_tip * tau))) ** 2)
            zigzag_wave = (zigzag_base * zigzag_depth) * params.zigzag_radius_factor if params.Output != "Simple Plot" else 0
            tiny_z_ripples = (params.rip_depth * (sin(z_fraction * params.rip_freq * tau)) ** 2) if params.Output != "Simple Plot" else 0
            r_now = r_0 + star_shape_wave + primary_z_wave + secondary_z_waves + zigzag_wave + tiny_z_ripples
            shell_steps.append(fc.polar_to_point(centre_now, r_now, a_now + angular_swerve))

        shell_steps.extend(fclab.reflectXYpolar_list(shell_steps, centre_now, params.start_angle + pi / max(n_tip, 1)))
        shell_steps = fc.move_polar(shell_steps, centre_now, 0, tau / max(n_tip, 1), copy=True, copy_quantity=n_tip)
        if params.Twist_turns:
            shell_steps = fc.move_polar(shell_steps, centre_now, 0, twist_angle)

        if target == "visualize" and params.viewer_point_stride > 1:
            eff_stride = int(params.viewer_point_stride)
            if params.Output != "Simple Plot" and params.zigzag_radius_factor > 0 and (params.zag_min > 0 or params.zag_max > 0):
                points_per_cycle = float(segs_shell_samples) / max(_zigzag_cycles, 1e-6)
                min_pts_per_cycle = 3.0 + (_round_t * 9.0)
                max_stride = int(points_per_cycle // min_pts_per_cycle)
                eff_stride = max(1, min(eff_stride, max(1, max_stride)))
            shell_steps = shell_steps[::eff_stride]
            if len(shell_steps) > 0:
                shell_steps.append(shell_steps[0])

        steps.extend([fc.ExtrusionGeometry(width=EW, height=EH), fc.Printer(print_speed=print_speed)] + shell_steps)

        if (
            (target == "gcode" and layer % params.layer_ratio == params.layer_ratio - 1 and layer < frame_layers)
            or (target == "visualize" and layer == 0 and frame_height > 0)
        ):
            for t_now in t_steps_frame_line:
                x_now = centre_xy + (params.frame_line_spacing_ratio * (params.frame_width_factor * EW)) + (amp_1 * t_now) * (
                    (0.5 - 0.5 * cos((t_now**0.66) * 3 * tau)) ** 1
                )
                y_now = centre_xy - frame_rad_inner - ((frame_rad_max - frame_rad_inner) * (1 - t_now))
                wave_steps.append(fc.Point(x=x_now, y=y_now, z=z_now))
            wave_steps.extend(fc.arcXY(centre_now, frame_rad_inner, params.start_angle, pi / max(n_tip, 1), int(64 / max(n_tip, 1))))
            wave_steps.extend(fclab.reflectXYpolar_list(wave_steps, centre_now, params.start_angle + pi / max(n_tip, 1)))
            wave_steps = fc.move_polar(wave_steps, centre_now, 0, tau / max(n_tip, 1), copy=True, copy_quantity=n_tip)
            if params.Twist_turns:
                wave_steps = fc.move_polar(wave_steps, centre_now, 0, twist_angle)

            steps.append(fc.ExtrusionGeometry(width=EW * params.frame_width_factor, height=EH * params.layer_ratio))
            steps.append(fc.Printer(print_speed=print_speed / (params.frame_width_factor * params.layer_ratio)))
            steps.extend(wave_steps)

    if params.Output == "Simple Plot":
        steps.append(
            fc.PlotAnnotation(point=fc.Point(x=centre_xy, y=50, z=0), label="Not all layers previewed - nor ripple texture")
        )
    if params.Output == "Detailed Plot":
        steps.append(fc.PlotAnnotation(point=fc.Point(x=centre_xy, y=50, z=0), label="Not all layers previewed"))
    steps.append(
        fc.PlotAnnotation(
            point=fc.Point(x=centre_xy, y=25, z=0),
            label=f"Speed increases from {params.initial_print_speed} to {params.main_print_speed} mm/min during first {params.speedchange_layers} layers",
        )
    )
    steps.append(
        fc.PlotAnnotation(
            point=fc.Point(x=centre_xy, y=0, z=0),
            label="Avoid larger overhangs than default design - ripple texture exacerbates overhangs",
        )
    )
    steps.append(
        fc.PlotAnnotation(
            point=fc.Point(x=centre_xy, y=centre_xy, z=height + 10),
            label="Try doubling speed - you may need to increase nozzle temperature",
        )
    )

    gcode_controls = fc.GcodeControls(
        printer_name=params.Printer_name,
        save_as=params.Design_name,
        initialization_data={
            "primer": "front_lines_then_y",
            "print_speed": params.initial_print_speed,
            "nozzle_temp": params.Nozzle_temp,
            "bed_temp": params.Bed_temp,
            "fan_percent": params.Fan_percent,
            "material_flow_percent": params.Material_flow_percent,
            "print_speed_percent": params.Print_speed_percent,
            "extrusion_width": EW,
            "extrusion_height": EH,
        },
    )
    plot_controls = fc.PlotControls(
        style="line",
        zoom=0.6,
        initialization_data={"extrusion_width": EW, "extrusion_height": EH},
    )
    plot_controls.hide_annotations = not params.Annotations

    return steps, plot_controls, gcode_controls
