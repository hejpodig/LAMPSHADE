import json
import sys
from dataclasses import asdict
from pathlib import Path

import streamlit as st

# Ensure local FullControl sources are importable when running from repo root.
# This repo layout is:
#   <repo>/fullcontrol/fullcontrol  (package)
#   <repo>/fullcontrol/lab          (package)
_REPO_ROOT = Path(__file__).resolve().parents[1]
_FULLCONTROL_SRC_ROOT = _REPO_ROOT / "fullcontrol"
if str(_FULLCONTROL_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_FULLCONTROL_SRC_ROOT))

import fullcontrol as fc

from streamlit_app.lampshade_model import LampshadeParams, build_lampshade_steps
from streamlit_app.plotting import plotdata_to_figure


st.set_page_config(page_title="FullControl Lampshade", layout="wide")


def _viewer_presets(viewer_mode: str) -> tuple[int, int]:
    if viewer_mode == "Fast viewer":
        return 5, 4
    if viewer_mode == "High detail":
        return 1, 1
    return 2, 2


def _build_params_from_ui() -> LampshadeParams:
    col_controls, col_preview = st.columns([1, 2], gap="large")

    with col_controls:
        st.markdown("### Controls")

        output = st.selectbox("Output", ["Simple Plot", "Detailed Plot", "GCode"], index=1)
        viewer_mode = st.selectbox("Viewer", ["Fast viewer", "Normal viewer", "High detail"], index=1)
        annotations = st.checkbox("Annotations", value=True)
        viewer_point_stride, viewer_layer_stride = _viewer_presets(viewer_mode)

        st.divider()
        st.markdown("### Printer")
        printer_name = st.selectbox(
            "Printer",
            ["generic", "ultimaker2plus", "prusa_i3", "ender_3", "cr_10", "bambulab_x1", "toolchanger_T"],
            index=0,
        )
        nozzle_temp = st.number_input("Nozzle °C", min_value=0, max_value=400, value=220, step=1)
        bed_temp = st.number_input("Bed °C", min_value=0, max_value=150, value=40, step=1)
        fan_percent = st.number_input("Fan %", min_value=0, max_value=100, value=100, step=1)
        material_flow_percent = st.number_input("Flow %", min_value=0, max_value=200, value=100, step=1)
        print_speed_percent = st.number_input("Speed %", min_value=10, max_value=400, value=200, step=5)
        design_name = st.text_input("Name", value="fc_lampshade")

        st.divider()
        st.markdown("### Geometry")
        height = st.slider("Height", min_value=100, max_value=200, value=150, step=10)
        nominal_radius = st.slider("Radius", min_value=20, max_value=50, value=34, step=1)
        tip_length = st.slider("Tip len", min_value=10, max_value=30, value=20, step=2)
        star_tips = st.slider("Star tips", min_value=0, max_value=8, value=6, step=1)
        main_bulge = st.slider("Main bulge", min_value=0.0, max_value=25.0, value=22.5, step=2.5)
        secondary_bulges = st.slider("2nd bulge", min_value=0.0, max_value=20.0, value=15.0, step=2.5)
        secondary_bulge_count = st.slider("Sec bulges", min_value=0, max_value=6, value=2, step=1)
        twist_turns = st.slider("Twist", min_value=-2.0, max_value=2.0, value=0.0, step=0.05)

        inner_frame_hole_diameter = st.number_input("Frame hole", min_value=0, max_value=200, value=30, step=1)
        inner_frame_height = st.slider("Frame ht", min_value=0, max_value=10, value=3, step=1)
        inner_frame_wave_amplitude = st.number_input(
            "Frame amp", min_value=0.0, max_value=200.0, value=17.5, step=0.5
        )
        centre_xy = st.number_input("Centre XY", min_value=0, max_value=500, value=104, step=1)

        st.divider()
        st.markdown("### Zigzags")
        zigzag_min = st.slider("Zigzag min", min_value=0.0, max_value=6.0, value=1.0, step=0.25)
        zigzag_max = st.slider("Zigzag max", min_value=0.0, max_value=10.0, value=5.0, step=0.25)
        zigzag_freq_factor = st.slider("Zigzag freq", min_value=0.25, max_value=3.0, value=1.0, step=0.05)
        zigzag_radius_factor = st.slider("Zigzag radius", min_value=0.0, max_value=3.0, value=1.0, step=0.05)
        zigzag_rounding_radius = st.slider("Zigzag round", min_value=0, max_value=10, value=0, step=1)

        st.divider()
        with st.expander("Advanced", expanded=False):
            eh = st.number_input("Layer height (EH)", min_value=0.05, max_value=2.0, value=0.2, step=0.05)
            ew = st.number_input("Line width (EW)", min_value=0.1, max_value=2.0, value=0.5, step=0.05)
            initial_print_speed = st.number_input(
                "Initial speed", min_value=10, max_value=5000, value=500, step=10
            )
            main_print_speed = st.number_input("Main speed", min_value=10, max_value=10000, value=1500, step=10)
            speedchange_layers = st.number_input("Speedchange layers", min_value=0, max_value=50, value=5, step=1)
            initial_z_factor = st.number_input("Initial z factor", min_value=0.0, max_value=1.0, value=0.7, step=0.05)

        generate = st.button("Generate / Update", type="primary")

    with col_preview:
        st.markdown("### Preview")
        if "last_params" not in st.session_state:
            st.session_state.last_params = None
        if "last_result" not in st.session_state:
            st.session_state.last_result = None

    params = LampshadeParams(
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
        Nominal_radius=int(nominal_radius),
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

    # Gate regeneration behind button to keep the app responsive.
    # First run: auto-generate once.
    params_key = json.dumps(asdict(params), sort_keys=True)
    if st.session_state.last_params is None:
        generate = True

    if generate:
        with col_preview:
            with st.spinner("Generating..."):
                try:
                    steps, plot_controls, gcode_controls = build_lampshade_steps(params)
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

    with col_preview:
        result = st.session_state.last_result
        if result is None:
            st.info("Click Generate / Update")
        elif result["type"] == "error":
            st.error(result["error"])
        elif result["type"] == "plot":
            st.plotly_chart(result["fig"], use_container_width=True)
        elif result["type"] == "gcode":
            st.download_button(
                "Download GCode",
                data=result["gcode"],
                file_name=f"{params.Design_name}.gcode",
                mime="text/plain",
            )
            st.text_area("GCode preview", value=result["gcode"][:20000], height=400)

    return params


st.title("FullControl Lampshade")
st.caption("Interactive lampshade generator + preview + GCode export")

_ = _build_params_from_ui()
