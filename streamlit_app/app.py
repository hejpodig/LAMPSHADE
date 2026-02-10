import json
import hashlib
import sys
import base64
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
from streamlit_clickable_images import clickable_images

# Ensure local FullControl sources are importable when running from repo root.
# This repo layout is:
#   <repo>/fullcontrol/fullcontrol  (package)
#   <repo>/fullcontrol/lab          (package)
_REPO_ROOT = Path(__file__).resolve().parents[1]
_FULLCONTROL_SRC_ROOT = _REPO_ROOT / "fullcontrol"
if str(_FULLCONTROL_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_FULLCONTROL_SRC_ROOT))

import fullcontrol as fc

# Streamlit executes this file with `streamlit_app/` as the script directory.
# Import sibling modules directly so it works both locally and on Streamlit Cloud.
from lampshade_model import LampshadeParams, build_lampshade_steps
from ripple_texture_model import RippleTextureParams, build_ripple_texture_steps
from plotting import plotdata_to_figure


st.set_page_config(page_title="FullControl Lampshade", layout="wide")

st.title("FullControl Lampshade")
st.caption(
    "This app was edited from the already functional fullcontrol.xyz lampshade editor. "
    "If you dont know how to edit GCode or basic troubleshooting this tool is not made for you."
)


_PRESET_VERSION = 1


def _ensure_defaults(defaults: dict) -> None:
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _preset_payload() -> dict:
    keys = [
        "Output",
        "Viewer",
        "Annotations",
        "Printer_name",
        "Nozzle_temp",
        "Bed_temp",
        "Fan_percent",
        "Material_flow_percent",
        "Print_speed_percent",
        "Design_name",
        "Height",
        "Inner_frame_hole_diameter",
        "Radius_bottom",
        "Radius_middle",
        "Radius_top",
        "Radius_middle_z",
        "Tip_length",
        "Star_tips",
        "Main_bulge",
        "Secondary_bulges",
        "Secondary_bulge_count",
        "Twist_turns",
        "Inner_frame_height",
        "Inner_frame_wave_amplitude",
        "Centre_XY",
        "zag_min",
        "zag_max",
        "zigzag_freq_factor",
        "zigzag_radius_factor",
        "zigzag_rounding_radius",
        "EH",
        "EW",
        "initial_print_speed",
        "main_print_speed",
        "speedchange_layers",
        "initial_z_factor",
    ]
    return {
        "version": _PRESET_VERSION,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "values": {k: st.session_state.get(k) for k in keys},
    }


def _apply_preset(preset: dict) -> None:
    if not isinstance(preset, dict):
        raise ValueError("Preset is not a JSON object")
    values = preset.get("values")
    if not isinstance(values, dict):
        raise ValueError("Preset missing 'values' object")

    int_keys = {
        "Nozzle_temp",
        "Bed_temp",
        "Fan_percent",
        "Material_flow_percent",
        "Print_speed_percent",
        "Height",
        "Inner_frame_hole_diameter",
        "Radius_bottom",
        "Radius_middle",
        "Radius_top",
        "Radius_middle_z",
        "Tip_length",
        "Star_tips",
        "Secondary_bulge_count",
        "Inner_frame_height",
        "Centre_XY",
        "zigzag_rounding_radius",
        "initial_print_speed",
        "main_print_speed",
        "speedchange_layers",
    }
    float_keys = {
        "Main_bulge",
        "Secondary_bulges",
        "Twist_turns",
        "Inner_frame_wave_amplitude",
        "zag_min",
        "zag_max",
        "zigzag_freq_factor",
        "zigzag_radius_factor",
        "EH",
        "EW",
        "initial_z_factor",
    }

    for k, v in values.items():
        if v is None:
            continue
        if k in int_keys:
            try:
                v = int(v)
            except Exception:
                pass
        elif k in float_keys:
            try:
                v = float(v)
            except Exception:
                pass
        st.session_state[k] = v


def _top_view_radius_figure(radius_mm: float) -> go.Figure:
    r = float(radius_mm)
    r = max(1.0, r)

    fig = go.Figure()
    fig.add_shape(type="circle", x0=-r, y0=-r, x1=r, y1=r, line=dict(width=2))
    fig.add_trace(go.Scatter(x=[0, r], y=[0, 0], mode="lines", line=dict(width=4), showlegend=False))
    fig.add_annotation(
        x=r,
        y=0,
        ax=0,
        ay=0,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        text=f"Middle radius: {r:.0f} mm",
        showarrow=True,
        arrowhead=3,
        arrowsize=1.2,
        arrowwidth=2,
        xanchor="left",
        yanchor="bottom",
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=220,
        showlegend=False,
        xaxis=dict(visible=False, range=[-1.2 * r, 1.2 * r], scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False, range=[-1.2 * r, 1.2 * r]),
    )
    return fig


def _profile_height_and_radii_figure(
    height_mm: float,
    r_bottom_mm: float,
    r_middle_mm: float,
    r_top_mm: float,
    middle_z_fraction: float,
) -> go.Figure:
    h = max(1.0, float(height_mm))
    rb = max(1.0, float(r_bottom_mm))
    rm = max(1.0, float(r_middle_mm))
    rt = max(1.0, float(r_top_mm))
    rmax = max(rb, rm, rt)
    mz = max(0.0, min(1.0, float(middle_z_fraction)))

    fig = go.Figure()

    # Height reference line
    fig.add_trace(go.Scatter(x=[0, 0], y=[0, h], mode="lines", line=dict(width=4), showlegend=False))
    fig.add_annotation(
        x=0,
        y=h,
        ax=0,
        ay=0,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        text=f"Height: {h:.0f} mm",
        showarrow=True,
        arrowhead=3,
        arrowsize=1.2,
        arrowwidth=2,
        xanchor="left",
        yanchor="bottom",
    )

    def _add_radius(y: float, r: float, label: str):
        fig.add_trace(go.Scatter(x=[0, r], y=[y, y], mode="lines", line=dict(width=4), showlegend=False))
        fig.add_annotation(
            x=r,
            y=y,
            ax=0,
            ay=y,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            text=f"{label}: {r:.0f} mm",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.2,
            arrowwidth=2,
            xanchor="left",
            yanchor="bottom",
        )

    _add_radius(0.0, rb, "Bottom radius")
    _add_radius(mz * h, rm, "Middle radius")
    _add_radius(h, rt, "Top radius")

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=220,
        showlegend=False,
        xaxis=dict(visible=False, range=[-0.15 * rmax, 1.35 * rmax]),
        yaxis=dict(visible=False, range=[-0.1 * h, 1.1 * h]),
    )
    return fig


def _viewer_presets(viewer_mode: str) -> tuple[int, int]:
    if viewer_mode == "Fast viewer":
        return 5, 4
    if viewer_mode == "High detail":
        return 1, 1
    return 2, 2


def _build_params_from_ui() -> LampshadeParams | RippleTextureParams:
    defaults = {
        "Design": "Lampshade",
        "Output": "Detailed Plot",
        "Viewer": "Normal viewer",
        "Annotations": True,
        "Printer_name": "generic",
        "Nozzle_temp": 220,
        "Bed_temp": 40,
        "Fan_percent": 100,
        "Material_flow_percent": 100,
        "Print_speed_percent": 200,
        "Design_name": "fc_lampshade",
        "Height": 150,
        "Inner_frame_hole_diameter": 30,
        "Radius_bottom": 34,
        "Radius_middle": 34,
        "Radius_top": 34,
        "Radius_middle_z": 50,
        "Tip_length": 20,
        "Star_tips": 6,
        "Main_bulge": 22.5,
        "Secondary_bulges": 15.0,
        "Secondary_bulge_count": 2,
        "Twist_turns": 0.0,
        "Inner_frame_height": 3,
        "Inner_frame_wave_amplitude": 17.5,
        "Centre_XY": 104,
        "zag_min": 1.0,
        "zag_max": 5.0,
        "zigzag_freq_factor": 1.0,
        "zigzag_radius_factor": 1.0,
        "zigzag_rounding_radius": 0,
        "EH": 0.2,
        "EW": 0.5,
        "initial_print_speed": 500,
        "main_print_speed": 1500,
        "speedchange_layers": 5,
        "initial_z_factor": 0.7,

        # Ripple texture defaults (prefixed so switching designs doesn't overwrite Lampshade values)
        "RT_inner_rad": 15.0,
        "RT_height": 40,
        "RT_skew_percent": 10.0,
        "RT_star_tips": 4,
        "RT_tip_length": 5.0,
        "RT_bulge": 2.0,
        "RT_nozzle_dia": 0.4,
        "RT_ripples_per_layer": 50,
        "RT_rip_depth": 1.0,
        "RT_shape_factor": 1.5,
    }
    _ensure_defaults(defaults)

    with st.sidebar:
        generate = st.button("Generate / Update", type="primary", use_container_width=True)

        design = st.selectbox(
            "Design",
            ["Lampshade", "Ripple texture"],
            key="Design",
        )
        if st.session_state.get("_design_last") != design:
            st.session_state["last_params"] = None
            st.session_state["last_result"] = None
            st.session_state["_design_last"] = design

        if design == "Lampshade":
            with st.expander("Body", expanded=True):
                height = st.slider(
                    "Height (mm)",
                    min_value=100,
                    max_value=200,
                    step=10,
                    help="Overall height of the lampshade.",
                    key="Height",
                )

                frame_hole_diameter_now = float(st.session_state.get("Inner_frame_hole_diameter", 0))
                min_bottom_radius = max(10, int((frame_hole_diameter_now / 2.0) + 2.0))
                if float(st.session_state.get("Radius_bottom", 0)) < float(min_bottom_radius):
                    st.session_state["Radius_bottom"] = int(min_bottom_radius)

                radius_bottom = st.number_input(
                    "Bottom radius (mm)",
                    min_value=int(min_bottom_radius),
                    max_value=80,
                    step=1,
                    help="Base radius at the bottom. Minimum is automatically constrained to fit the inner frame.",
                    key="Radius_bottom",
                )
                radius_middle = st.number_input(
                    "Middle radius (mm)",
                    min_value=10,
                    max_value=80,
                    step=1,
                    help="Base radius at the adjustable middle height.",
                    key="Radius_middle",
                )
                radius_top = st.number_input(
                    "Top radius (mm)",
                    min_value=10,
                    max_value=80,
                    step=1,
                    help="Base radius at the top.",
                    key="Radius_top",
                )
                radius_middle_z = st.slider(
                    "Middle radius position (% of height)",
                    min_value=5,
                    max_value=95,
                    step=1,
                    help="Where along the height the middle radius occurs.",
                    key="Radius_middle_z",
                )

                inner_frame_hole_diameter = st.number_input(
                    "Frame hole diameter (mm)",
                    min_value=0,
                    max_value=200,
                    step=1,
                    help="Diameter of the inner hole in the printed frame/ring.",
                    key="Inner_frame_hole_diameter",
                )
                inner_frame_height = st.slider(
                    "Inner frame height (mm)",
                    min_value=0,
                    max_value=10,
                    step=1,
                    help="0 disables the inner frame.",
                    key="Inner_frame_height",
                )

            with st.expander("Shape", expanded=True):
                st.markdown("**Star tips**")
                star_tips = st.slider(
                    "Number of star tips",
                    min_value=0,
                    max_value=8,
                    step=1,
                    help="0 disables the star pattern (round shade).",
                    key="Star_tips",
                )
                tip_length = st.slider(
                    "Star tip length",
                    min_value=10,
                    max_value=30,
                    step=2,
                    help="How long each star point extends outwards (mm).",
                    key="Tip_length",
                )
                twist_turns = st.slider(
                    "Twist (turns bottom→top)",
                    min_value=-2.0,
                    max_value=2.0,
                    step=0.05,
                    help="Applies twist to the shell. The inner frame stays untwisted.",
                    key="Twist_turns",
                )

                st.markdown("**Bulge**")
                secondary_bulge_count = st.slider(
                    "Bulge count",
                    min_value=0,
                    max_value=6,
                    step=1,
                    help="0 disables secondary bulges.",
                    key="Secondary_bulge_count",
                )
                main_bulge = st.slider(
                    "Main bulge amplitude (mm)",
                    min_value=0.0,
                    max_value=25.0,
                    step=2.5,
                    help="Controls the overall bulbous shape.",
                    key="Main_bulge",
                )
                secondary_bulges = st.slider(
                    "Secondary bulge amplitude (mm)",
                    min_value=0.0,
                    max_value=20.0,
                    step=2.5,
                    help="Controls the smaller bulges along the height.",
                    key="Secondary_bulges",
                )

            with st.expander("Zigzags", expanded=True):
                zigzag_min = st.slider(
                    "Zigzag depth (min)",
                    min_value=0.0,
                    max_value=6.0,
                    step=0.25,
                    help="Minimum zigzag depth around the circumference.",
                    key="zag_min",
                )
                zigzag_max = st.slider(
                    "Zigzag depth (max)",
                    min_value=0.0,
                    max_value=10.0,
                    step=0.25,
                    help="Maximum zigzag depth (usually at star tips).",
                    key="zag_max",
                )
                zigzag_freq_factor = st.slider(
                    "Zigzag frequency multiplier",
                    min_value=0.25,
                    max_value=3.0,
                    step=0.05,
                    help="Higher = more zigzags around the perimeter.",
                    key="zigzag_freq_factor",
                )
                zigzag_radius_factor = st.slider(
                    "Zigzag amplitude multiplier",
                    min_value=0.0,
                    max_value=3.0,
                    step=0.05,
                    help="Scales how strongly zigzags affect radius.",
                    key="zigzag_radius_factor",
                )
                zigzag_rounding_radius = st.slider(
                    "Zigzag rounding (0–10)",
                    min_value=0,
                    max_value=10,
                    step=1,
                    help="0 = sharp/triangular; 10 = smooth/rounded.",
                    key="zigzag_rounding_radius",
                )

            with st.expander("Advanced", expanded=False):
                eh = st.number_input(
                    "Layer height EH (mm)",
                    min_value=0.05,
                    max_value=2.0,
                    step=0.05,
                    help="Base layer height used by the design (will be scaled for the plot modes).",
                    key="EH",
                )
                ew = st.number_input(
                    "Line width EW (mm)",
                    min_value=0.1,
                    max_value=2.0,
                    step=0.05,
                    help="Designed extrusion width.",
                    key="EW",
                )
                initial_print_speed = st.number_input(
                    "Initial print speed (mm/min)",
                    min_value=10,
                    max_value=5000,
                    step=10,
                    key="initial_print_speed",
                )
                main_print_speed = st.number_input(
                    "Main print speed (mm/min)",
                    min_value=10,
                    max_value=10000,
                    step=10,
                    key="main_print_speed",
                )
                speedchange_layers = st.number_input(
                    "Speed ramp layers",
                    min_value=0,
                    max_value=50,
                    step=1,
                    help="Number of layers used to ramp from initial to main speed.",
                    key="speedchange_layers",
                )
                initial_z_factor = st.number_input(
                    "First-layer Z factor",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.05,
                    help="Scales the first layer Z (lower = more squish).",
                    key="initial_z_factor",
                )

                inner_frame_wave_amplitude = st.number_input(
                    "Inner frame wave amplitude (mm)",
                    min_value=0.0,
                    max_value=200.0,
                    step=0.5,
                    help="Amplitude of the wavy inner frame lines.",
                    key="Inner_frame_wave_amplitude",
                )
                centre_xy = st.number_input(
                    "Centre position XY (mm)",
                    min_value=0,
                    max_value=500,
                    step=1,
                    help="Where the shade is centered on the build plate.",
                    key="Centre_XY",
                )
        else:
            with st.expander("Body", expanded=True):
                rt_inner_rad = st.number_input(
                    "Inner radius (mm)",
                    min_value=10.0,
                    max_value=30.0,
                    step=0.5,
                    help="Base radius that other features morph outwards from.",
                    key="RT_inner_rad",
                )
                rt_height = st.slider(
                    "Height (mm)",
                    min_value=20,
                    max_value=80,
                    step=5,
                    help="Height of the part.",
                    key="RT_height",
                )

            with st.expander("Shape", expanded=True):
                rt_skew_percent = st.slider(
                    "Twist (%)",
                    min_value=-100.0,
                    max_value=100.0,
                    step=1.0,
                    help="100% is one full rotation anti-clockwise over the height.",
                    key="RT_skew_percent",
                )
                rt_star_tips = st.slider(
                    "Star tips",
                    min_value=0,
                    max_value=10,
                    step=1,
                    help="Number of star points around the perimeter.",
                    key="RT_star_tips",
                )
                rt_tip_length = st.slider(
                    "Star tip length (mm)",
                    min_value=-20.0,
                    max_value=20.0,
                    step=0.5,
                    help="How far each star point extends outward.",
                    key="RT_tip_length",
                )
                rt_bulge = st.slider(
                    "Bulge (mm)",
                    min_value=-20.0,
                    max_value=20.0,
                    step=0.5,
                    help="Adds a smooth bulge over the height.",
                    key="RT_bulge",
                )

            with st.expander("Advanced", expanded=False):
                rt_nozzle_dia = st.number_input(
                    "Nozzle Diameter (mm)",
                    min_value=0.3,
                    max_value=1.2,
                    step=0.05,
                    help="Nozzle diameter used to derive layer height and line width.",
                    key="RT_nozzle_dia",
                )
                rt_ripples_per_layer = st.slider(
                    "Ripples Per Layer",
                    min_value=20,
                    max_value=100,
                    step=1,
                    help="How many ripples occur around one layer.",
                    key="RT_ripples_per_layer",
                )
                rt_rip_depth = st.slider(
                    "Ripple Depth (mm)",
                    min_value=0.0,
                    max_value=5.0,
                    step=0.1,
                    help="Amplitude of the ripple effect.",
                    key="RT_rip_depth",
                )
                rt_shape_factor = st.slider(
                    "Star Tip Pointiness",
                    min_value=0.25,
                    max_value=5.0,
                    step=0.05,
                    help="Higher values make tips sharper/more pointy.",
                    key="RT_shape_factor",
                )

        with st.expander("Controls", expanded=True):
            output = st.selectbox(
                "Output mode",
                ["Simple Plot", "Detailed Plot", "GCode"],
                help="Choose whether to preview the design or download GCode.",
                key="Output",
            )
            viewer_mode = st.selectbox(
                "Viewer detail",
                ["Fast viewer", "Normal viewer", "High detail"],
                help="Fast viewer down-samples the preview for speed (does not change the exported GCode).",
                key="Viewer",
            )
            annotations = st.checkbox("Show annotations", help="Show/hide notes in the preview.", key="Annotations")
            viewer_point_stride, viewer_layer_stride = _viewer_presets(viewer_mode)

        with st.expander("Printer", expanded=True):
            printer_name = st.selectbox(
                "Printer profile",
                ["generic", "ultimaker2plus", "prusa_i3", "ender_3", "cr_10", "bambulab_x1", "toolchanger_T"],
                help="Affects startup/end GCode and conventions.",
                key="Printer_name",
            )
            nozzle_temp = st.number_input(
                "Nozzle temperature (°C)", min_value=0, max_value=400, step=1, key="Nozzle_temp"
            )
            bed_temp = st.number_input("Bed temperature (°C)", min_value=0, max_value=150, step=1, key="Bed_temp")
            fan_percent = st.number_input("Part cooling fan (%)", min_value=0, max_value=100, step=1, key="Fan_percent")
            material_flow_percent = st.number_input(
                "Flow multiplier (%)", min_value=0, max_value=200, step=1, key="Material_flow_percent"
            )
            print_speed_percent = st.number_input(
                "Speed multiplier (%)", min_value=10, max_value=400, step=5, key="Print_speed_percent"
            )
            design_name = st.text_input(
                "Output name",
                help="Used as the downloaded GCode filename prefix.",
                key="Design_name",
            )

        if design == "Lampshade":
            st.divider()
            with st.expander("Presets", expanded=False):
                preset_name = st.text_input(
                    "Preset file name",
                    value=str(st.session_state.get("Design_name", "fc_lampshade")),
                    help="Used only for the downloaded preset filename.",
                    key="_preset_name",
                )
                preset_json = json.dumps(_preset_payload(), indent=2, sort_keys=True)
                st.download_button(
                    "Download preset (.json)",
                    data=preset_json,
                    file_name=f"{preset_name or 'lampshade'}_preset.json",
                    mime="application/json",
                    use_container_width=True,
                )
                uploaded = st.file_uploader("Upload preset (.json)", type=["json"], key="_preset_upload")
                if uploaded is not None:
                    try:
                        raw = uploaded.getvalue()
                        token = hashlib.sha256(raw).hexdigest()
                        if st.session_state.get("_preset_upload_token") != token:
                            loaded = json.loads(raw.decode("utf-8"))
                            _apply_preset(loaded)

                            # Force a regen so the preview matches the newly loaded settings.
                            st.session_state["last_params"] = None
                            st.session_state["last_result"] = None

                            st.session_state["_preset_upload_token"] = token
                            st.success("Preset loaded")
                    except Exception as e:
                        st.error(f"Invalid preset: {e}")
                else:
                    st.session_state.pop("_preset_upload_token", None)

    if "last_params" not in st.session_state:
        st.session_state.last_params = None
    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    if design == "Lampshade":
        params: LampshadeParams | RippleTextureParams = LampshadeParams(
            Output=output,
            Annotations=annotations,
            Printer_name=printer_name,
            Design_name=design_name,
            Nozzle_temp=int(nozzle_temp),
            Bed_temp=int(bed_temp),
            Fan_percent=int(fan_percent),
            Material_flow_percent=int(material_flow_percent),
            Print_speed_percent=int(print_speed_percent),
            Height=int(height),
            Nominal_radius=int(radius_middle),
            Radius_bottom=int(radius_bottom),
            Radius_middle=int(radius_middle),
            Radius_top=int(radius_top),
            Radius_middle_z_fraction=float(radius_middle_z) / 100.0,
            Tip_length=int(tip_length),
            Star_tips=int(star_tips),
            Main_bulge=float(main_bulge),
            Secondary_bulges=float(secondary_bulges),
            Secondary_bulge_count=int(secondary_bulge_count),
            Twist_turns=float(twist_turns),
            Inner_frame_hole_diameter=int(inner_frame_hole_diameter),
            Inner_frame_height=int(inner_frame_height),
            Inner_frame_wave_amplitude=float(inner_frame_wave_amplitude),
            Centre_XY=int(centre_xy),
            zag_min=float(zigzag_min),
            zag_max=float(zigzag_max),
            zigzag_freq_factor=float(zigzag_freq_factor),
            zigzag_radius_factor=float(zigzag_radius_factor),
            zigzag_rounding_radius=int(zigzag_rounding_radius),
            EH=float(eh),
            EW=float(ew),
            initial_print_speed=int(initial_print_speed),
            main_print_speed=int(main_print_speed),
            speedchange_layers=int(speedchange_layers),
            initial_z_factor=float(initial_z_factor),
            viewer_point_stride=int(viewer_point_stride),
            viewer_layer_stride=int(viewer_layer_stride),
        )
    else:
        params = RippleTextureParams(
            Output=output,
            Viewer=viewer_mode,
            Annotations=annotations,
            Printer_name=printer_name,
            Nozzle_temp=int(nozzle_temp),
            Bed_temp=int(bed_temp),
            print_speed=500.0,
            Fan_percent=int(fan_percent),
            Material_flow_percent=int(material_flow_percent),
            Print_speed_percent=int(print_speed_percent),
            Design_name=design_name,
            inner_rad=float(st.session_state.get("RT_inner_rad", 15.0)),
            height=float(st.session_state.get("RT_height", 40)),
            skew_percent=float(st.session_state.get("RT_skew_percent", 10.0)),
            star_tips=int(st.session_state.get("RT_star_tips", 4)),
            tip_length=float(st.session_state.get("RT_tip_length", 5.0)),
            bulge=float(st.session_state.get("RT_bulge", 2.0)),
            nozzle_dia=float(st.session_state.get("RT_nozzle_dia", 0.4)),
            ripples_per_layer=int(st.session_state.get("RT_ripples_per_layer", 50)),
            rip_depth=float(st.session_state.get("RT_rip_depth", 1.0)),
            shape_factor=float(st.session_state.get("RT_shape_factor", 1.5)),
            ripple_segs=2,
            first_layer_E_factor=0.4,
            centre_x=50.0,
            centre_y=50.0,
            viewer_point_stride=int(viewer_point_stride),
            viewer_layer_stride=int(viewer_layer_stride),
        )

    # Gate regeneration behind button to keep the app responsive.
    # First run: auto-generate once.
    params_key = json.dumps({"design": design, "params": asdict(params)}, sort_keys=True)
    if st.session_state.last_params is None:
        generate = True

    if generate:
        with st.spinner("Generating..."):
            try:
                if design == "Lampshade":
                    steps, plot_controls, gcode_controls = build_lampshade_steps(params)  # type: ignore[arg-type]
                else:
                    steps, plot_controls, gcode_controls = build_ripple_texture_steps(params)  # type: ignore[arg-type]
                st.session_state.last_params = params_key

                if params.Output in ["Simple Plot", "Detailed Plot"]:
                    plot_controls.raw_data = True
                    plot_data = fc.transform(steps, "plot", plot_controls)
                    fig = plotdata_to_figure(plot_data, plot_controls)
                    st.session_state.last_result = {"type": "plot", "fig": fig}
                else:
                    gcode_str = fc.transform(steps, "gcode", gcode_controls)
                    st.session_state.last_result = {"type": "gcode", "gcode": gcode_str}
            except Exception as e:
                st.session_state.last_result = {"type": "error", "error": repr(e)}

    result = st.session_state.last_result
    if result is None:
        st.info("Click Generate / Update")
    elif result["type"] == "error":
        st.error(result["error"])
    elif result["type"] == "plot":
        st.plotly_chart(result["fig"], use_container_width=True)
        if design == "Lampshade":
            st.divider()

            st.title("How to")

            images_dir = Path(__file__).parent
            image_names = [
                "Frameheight.png",
                "Framehole.png",
                "profile radius.png",
                "startiplength.png",
                "startips.png",
                "topradius.png",
                "Zigzag.png",
            ]
            image_titles = {
                "Frameheight.png": "Frame height",
                "Framehole.png": "Frame Hole",
                "profile radius.png": "Body Radius",
                "startiplength.png": "Star tip length",
                "startips.png": "Star tips",
                "topradius.png": "Top radius",
                "Zigzag.png": "ZigZag",
            }
            image_descriptions = {
                "Frameheight.png": "**Frame height**\n\nControls the height of the printed inner frame/ring at the base. Set to 0 to disable the inner frame.",
                "Framehole.png": "**Frame Hole**\n\nSets the diameter of the central hole. This should be able to fit your bulb socket.",
                "profile radius.png": "**Body Radius**\n\nShows how bottom / middle / top radii define the overall silhouette from bottom to top, with the middle radius occurring at the chosen height percentage.",
                "startiplength.png": "**Star tip length**\n\nControls how far each star point extends outward. Higher values make sharper, more pronounced tips.",
                "startips.png": "**Star tips**\n\nControls how many star points are around the perimeter. Set to 0 for a round lampshade.",
                "topradius.png": "**Top radius**\n\nControls the size of the opening at the top. Larger values widen the lampshade at the top.",
                "Zigzag.png": "**ZigZag**\n\nAdds a periodic in-and-out ripple around the perimeter. Depth controls how strong it is; frequency controls how many zigzags occur around the circumference.",
            }
            images = [(name, images_dir / name) for name in image_names if (images_dir / name).exists()]
            if images:
                image_data_uris = [
                    f"data:image/png;base64,{base64.b64encode(p.read_bytes()).decode('utf-8')}" for _, p in images
                ]

                clicked = clickable_images(
                    image_data_uris,
                    titles=[image_titles.get(name, name) for name, _ in images],
                    div_style={
                        "display": "flex",
                        "justify-content": "center",
                        "gap": "12px",
                        "flex-wrap": "nowrap",
                        "width": "100%",
                    },
                    img_style={"height": "120px", "cursor": "pointer"},
                )

                if clicked is not None and int(clicked) >= 0:
                    image_name, image_path = images[int(clicked)]
                    dialog_title = image_titles.get(image_name, image_name)

                    @st.dialog(dialog_title)
                    def _show_image_dialog() -> None:
                        st.image(str(image_path), use_container_width=True)
                        desc = image_descriptions.get(image_name)
                        if desc:
                            st.markdown(desc)

                    _show_image_dialog()
    elif result["type"] == "gcode":
        st.download_button(
            "Download GCode",
            data=result["gcode"],
            file_name=f"{params.Design_name}.gcode",
            mime="text/plain",
        )
        st.text_area("GCode preview", value=result["gcode"][:20000], height=400)

    return params


_ = _build_params_from_ui()
