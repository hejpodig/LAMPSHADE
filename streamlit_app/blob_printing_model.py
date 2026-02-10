from __future__ import annotations

from dataclasses import dataclass
from math import tau

import fullcontrol as fc

from frame import add_cardinal_frame


@dataclass
class BlobPrintingParams:
    # UI / output
    Output: str = "Detailed Plot"  # Simple Plot | Detailed Plot | GCode
    Viewer: str = "Normal viewer"  # Fast viewer | Normal viewer | High detail
    Annotations: bool = True

    # Printer / GCode
    Printer_name: str = "prusa_i3"
    Nozzle_temp: int = 210
    Bed_temp: int = 40
    Fan_percent: int = 100
    Material_flow_percent: int = 100
    Print_speed_percent: int = 100
    Design_name: str = "blobs"

    # Design (matches fullcontrol/models/blob_printing.ipynb)
    tube_radius: float = 75.0
    layers: int = 125
    dense_layers: int = 2

    blob_size: float = 1.6
    blob_overlap_percent: float = 33.0
    extrusion_speed: float = 100.0

    centre_x: float = 125.0
    centre_y: float = 125.0

    # Viewer sampling (kept for parity with the app)
    viewer_point_stride: int = 2
    viewer_layer_stride: int = 2


def build_blob_printing_steps(params: BlobPrintingParams) -> tuple[list, fc.PlotControls, fc.GcodeControls]:
    """Build FullControl steps for the blob printing demo.

    Ported from fullcontrol/models/blob_printing.ipynb.
    """

    tube_radius = float(params.tube_radius)
    blob_size = max(0.01, float(params.blob_size))
    blob_overlap_percent = float(params.blob_overlap_percent)
    layers = max(1, int(params.layers))
    dense_layers = max(0, int(params.dense_layers))
    extrusion_speed = max(0.0, float(params.extrusion_speed))

    blob_height = blob_size / 2.0
    blob_spacing = blob_size * (1.0 - (blob_overlap_percent / 100.0))
    blob_spacing = max(blob_spacing, blob_size * 0.01)

    blob_vol = blob_height * (blob_size**2)
    initial_z = 0.95 * blob_height

    def move_and_blob(steps: list, point: fc.Point, volume: float, extrusion_speed_now: float) -> None:
        # StationaryExtrusion doesn't create a path segment, so add an empty PlotAnnotation.
        # The visualizer uses this to draw a node marker (matching the demo notebook).
        steps.extend([point, fc.StationaryExtrusion(volume=volume, speed=extrusion_speed_now), fc.PlotAnnotation(label="")])

    blobs_per_layer = int(tau * tube_radius / blob_spacing)
    blobs_per_layer = max(blobs_per_layer, 2)
    if blobs_per_layer % 2 != 0:
        blobs_per_layer += 1
    angle_between_blobs = tau / blobs_per_layer

    steps: list = []

    # Add the lampshade-style inner frame (4 contact sectors) to help bed adhesion.
    # Fixed dimensions to match the lampshade defaults.
    frame_hole_diameter = 30.0
    frame_height = 3.0
    frame_rad_inner = (frame_hole_diameter / 2.0) + (blob_size / 2.0)
    frame_layers = int(frame_height / blob_height) if frame_height > 0 else 0
    for layer in range(max(0, frame_layers)):
        add_cardinal_frame(
            steps=steps,
            centre=fc.Point(x=0, y=0, z=0),
            z=float(layer) * float(blob_height),
            frame_rad_inner=float(frame_rad_inner),
            extent_east=float(tube_radius),
            extent_west=float(tube_radius),
            extent_north=float(tube_radius),
            extent_south=float(tube_radius),
            ew=float(blob_size),
            eh=float(blob_height),
            print_speed=100.0,
        )

    # primer line
    steps.extend(
        [
            fc.Extruder(on=True),
            fc.Point(x=tube_radius + 20 * blob_spacing, y=0, z=0),
            fc.Printer(print_speed=100),
            fc.ExtrusionGeometry(width=blob_size, height=blob_height),
            fc.Point(x=tube_radius + 10 * blob_spacing, y=0, z=0),
            fc.Extruder(on=False),
        ]
    )

    # primer blobs
    primer_blob_pts = fc.segmented_line(
        fc.Point(x=tube_radius + 10 * blob_spacing, y=0, z=0),
        fc.Point(x=tube_radius, y=0, z=0),
        10,
    )
    for blob_pt in primer_blob_pts[1:-1]:
        move_and_blob(steps, blob_pt, blob_vol, extrusion_speed)

    # print all the blobs
    for layer in range(layers):
        for blob in range(blobs_per_layer):
            if (layer < dense_layers or layer >= layers - dense_layers) or (blob % 2 == 0):
                move_and_blob(
                    steps,
                    fc.polar_to_point(
                        centre=fc.Point(x=0, y=0, z=layer * blob_height),
                        radius=tube_radius,
                        angle=angle_between_blobs * blob,
                    ),
                    blob_vol,
                    extrusion_speed,
                )
        # move directly over the top of the first point so the nozzle moves directly up in Z to begin the next layer
        steps.append(fc.Point(x=tube_radius, y=0))

    # Offset procedure
    steps = fc.move(steps, fc.Vector(x=float(params.centre_x), y=float(params.centre_y), z=float(initial_z)))

    if params.Annotations:
        steps.append(
            fc.PlotAnnotation(
                point=fc.Point(x=float(params.centre_x), y=float(params.centre_y), z=blob_height * layers * 2),
                label="Nodes in this preview show where blobs are deposited, but do not represent the size of blobs",
            )
        )
        steps.append(
            fc.PlotAnnotation(
                point=fc.Point(x=float(params.centre_x), y=float(params.centre_y), z=blob_height * layers * 1.5),
                label=f"For this blob volume ({blob_vol:.1f} mm3) a good blob extrusion speed may take about {blob_vol/4:.1f}-{blob_vol/2:.1f} seconds per blob",
            )
        )

    gcode_controls = fc.GcodeControls(
        printer_name=params.Printer_name,
        save_as=params.Design_name,
        initialization_data={
            "print_speed": 100,
            "nozzle_temp": int(params.Nozzle_temp),
            "bed_temp": int(params.Bed_temp),
            "fan_percent": int(params.Fan_percent),
            "material_flow_percent": int(params.Material_flow_percent),
            "print_speed_percent": int(params.Print_speed_percent),
            "extrusion_width": blob_size,
            "extrusion_height": blob_height,
        },
    )

    plot_controls = fc.PlotControls(
        style="line",
        zoom=0.9,
        initialization_data={"extrusion_width": blob_size, "extrusion_height": blob_height},
    )
    plot_controls.hide_annotations = not params.Annotations

    return steps, plot_controls, gcode_controls
