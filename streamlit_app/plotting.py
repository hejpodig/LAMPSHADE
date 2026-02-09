from __future__ import annotations

import plotly.graph_objects as go

from fullcontrol.visualize.plot_data import PlotData
from fullcontrol.visualize.controls import PlotControls


def plotdata_to_figure(data: PlotData, controls: PlotControls) -> go.Figure:
    """Create a Plotly Figure from FullControl PlotData.

    Mirrors fullcontrol.visualize.plotly.plot(), but returns a Figure.
    """

    fig = go.Figure()

    max_width = 0.0
    for path in data.paths:
        colors_now = [f"rgb({c[0]*255:.2f}, {c[1]*255:.2f}, {c[2]*255:.2f})" for c in path.colors]
        linewidth_now = controls.line_width * (2 if path.extruder.on else 0.5)
        max_width = max(max_width, linewidth_now)

        if (not controls.hide_travel) or path.extruder.on:
            fig.add_trace(
                go.Scatter3d(
                    mode="lines",
                    x=path.xvals,
                    y=path.yvals,
                    z=path.zvals,
                    showlegend=False,
                    line=dict(width=linewidth_now, color=colors_now),
                )
            )

    bounding_box_size = max(
        data.bounding_box.maxx - data.bounding_box.minx,
        data.bounding_box.maxy - data.bounding_box.miny,
        data.bounding_box.maxz - min(0, data.bounding_box.minz),
    )
    bounding_box_size += 0.002
    bounding_box_size += max_width

    annotations = []
    if controls.hide_annotations is False and not controls.neat_for_publishing:
        annotations_pts = []
        for annotation in data.annotations:
            x, y, z = (annotation[axis] for axis in "xyz")
            annotations_pts.append((x, y, z))
            annotations.append(dict(showarrow=False, x=x, y=y, z=z, text=annotation["label"], yshift=10))

        if annotations_pts:
            xs, ys, zs = zip(*annotations_pts)
            fig.add_trace(
                go.Scatter3d(
                    mode="markers",
                    x=xs,
                    y=ys,
                    z=zs,
                    showlegend=False,
                    marker=dict(size=2, color="red"),
                )
            )

    relative_centre_z = 0.5 * data.bounding_box.rangez / bounding_box_size
    camera_centre_z = -0.5 + relative_centre_z
    camera = dict(
        eye=dict(x=-0.5 / controls.zoom, y=-1 / controls.zoom, z=-0.5 + 0.5 / controls.zoom),
        center=dict(x=0, y=0, z=camera_centre_z),
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="black",
        scene_aspectmode="cube",
        scene=dict(
            annotations=annotations,
            xaxis=dict(
                backgroundcolor="black",
                nticks=10,
                range=[data.bounding_box.midx - bounding_box_size / 2, data.bounding_box.midx + bounding_box_size / 2],
            ),
            yaxis=dict(
                backgroundcolor="black",
                nticks=10,
                range=[data.bounding_box.midy - bounding_box_size / 2, data.bounding_box.midy + bounding_box_size / 2],
            ),
            zaxis=dict(backgroundcolor="black", nticks=10, range=[min(0, data.bounding_box.minz), bounding_box_size]),
        ),
        scene_camera=camera,
        width=800,
        height=500,
        margin=dict(l=10, r=10, b=10, t=10, pad=4),
    )

    if controls.hide_axes or controls.neat_for_publishing:
        for axis in ["xaxis", "yaxis", "zaxis"]:
            fig.update_layout(scene={axis: dict(showgrid=False, zeroline=False, visible=False)})
    if controls.neat_for_publishing:
        fig.update_layout(width=500, height=500)

    return fig
