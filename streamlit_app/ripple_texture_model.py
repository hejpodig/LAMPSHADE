from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin, tau

import fullcontrol as fc


@dataclass
class RippleTextureParams:
    # UI / output
    Output: str = "Detailed Plot"  # Simple Plot | Detailed Plot | GCode
    Viewer: str = "Normal viewer"  # Fast viewer | Normal viewer | High detail
    Annotations: bool = True

    # Printer / GCode
    Printer_name: str = "prusa_i3"
    Nozzle_temp: int = 210
    Bed_temp: int = 40
    print_speed: float = 500.0
    Fan_percent: int = 100
    Material_flow_percent: int = 100
    Print_speed_percent: int = 100
    Design_name: str = "ripples"

    # Design
    inner_rad: float = 15.0
    height: float = 40.0
    skew_percent: float = 10.0
    star_tips: int = 4
    tip_length: float = 5.0
    bulge: float = 2.0

    nozzle_dia: float = 0.4
    ripples_per_layer: int = 50
    rip_depth: float = 1.0
    shape_factor: float = 1.5
    ripple_segs: int = 2
    first_layer_E_factor: float = 0.4

    centre_x: float = 50.0
    centre_y: float = 50.0

    # Viewer sampling (currently unused by the ripple generator, but kept for parity with the app)
    viewer_point_stride: int = 2
    viewer_layer_stride: int = 2


def build_ripple_texture_steps(params: RippleTextureParams) -> tuple[list, fc.PlotControls, fc.GcodeControls]:
    """Build FullControl steps for the ripple texture demo.

    Ported from fullcontrol/models/ripple_texture.ipynb.
    """

    nozzle_dia = max(0.05, float(params.nozzle_dia))
    ew = nozzle_dia * 2.5
    eh = nozzle_dia * 0.6

    height = max(float(params.height), eh)
    layers = max(1, int(height / eh))

    ripples_per_layer = max(1, int(params.ripples_per_layer))
    ripple_segs = max(2, int(params.ripple_segs))
    layer_segs = max(1, int(round((ripples_per_layer + 0.5) * ripple_segs)))

    initial_z = 0.8 * eh
    model_offset = fc.Vector(x=float(params.centre_x), y=float(params.centre_y), z=float(initial_z))

    print_speed = float(params.print_speed)
    steps: list = []
    steps.append(fc.Printer(print_speed=print_speed / 2.0))

    centre_now = fc.Point(x=0, y=0, z=0)
    inner_rad = float(params.inner_rad)
    rip_depth = float(params.rip_depth)
    tip_length = float(params.tip_length)
    bulge = float(params.bulge)
    shape_factor = float(params.shape_factor)
    skew_percent = float(params.skew_percent)
    star_tips = int(params.star_tips)
    first_layer_E_factor = float(params.first_layer_E_factor)

    total_steps = layers * layer_segs
    for t in range(int(total_steps)):
        t_val = t / float(layer_segs)  # 0..layers

        a_now = t_val * tau * (1.0 + (skew_percent / 100.0) / max(layers, 1))
        a_now -= tau / 4.0

        ripple_wave = rip_depth * (0.5 + (0.5 * cos((ripples_per_layer + 0.5) * (t_val * tau))))
        star_wave = tip_length * (0.5 - 0.5 * cos(star_tips * (t_val * tau))) ** shape_factor if star_tips else 0.0
        bulge_wave = bulge * (sin((centre_now.z / height) * (0.5 * tau))) if height else 0.0

        r_now = inner_rad + ripple_wave + star_wave + bulge_wave

        centre_now.z = t_val * eh
        if t < layer_segs:
            steps.append(fc.ExtrusionGeometry(height=eh + eh * t_val * first_layer_E_factor, width=ew))
        elif t == layer_segs:
            steps.append(fc.ExtrusionGeometry(height=eh, width=ew))
            steps.append(fc.Printer(print_speed=print_speed))

        steps.append(fc.polar_to_point(centre_now, r_now, a_now))

    steps = fc.move(steps, model_offset)

    gcode_controls = fc.GcodeControls(
        printer_name=params.Printer_name,
        save_as=params.Design_name,
        initialization_data={
            "primer": "front_lines_then_y",
            "print_speed": print_speed,
            "nozzle_temp": int(params.Nozzle_temp),
            "bed_temp": int(params.Bed_temp),
            "fan_percent": int(params.Fan_percent),
            "material_flow_percent": int(params.Material_flow_percent),
            "print_speed_percent": int(params.Print_speed_percent),
            "extrusion_width": ew,
            "extrusion_height": eh,
        },
    )

    plot_controls = fc.PlotControls(
        style="line",
        zoom=0.5,
        initialization_data={"extrusion_width": ew, "extrusion_height": eh},
    )
    plot_controls.hide_annotations = not params.Annotations

    return steps, plot_controls, gcode_controls
