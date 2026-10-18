"""Interactive HTML timeline of a replayed storm, one site at a time."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

RAIN, PROB = "#2a78d6", "#0b0b0b"
NOTIFY_C, INTERVENE_C, SPILL_C = "#eda100", "#e34948", "#eb6834"
MUTED, GRID = "#52514e", "#e6e5e0"
HOUR_MS = 3_600_000


def _y_range(site, thresholds):
    """Room for the probability line and both threshold lines."""
    top = max(site["probs"].max() if len(site["probs"]) else 0, thresholds["intervene"])
    return [0, 100 * top * 1.15]


def _site_traces(rows, probs, alerts):
    """Rain bars, probability line and alert markers for one site.

    alerts: list of dicts with timestamp, p and level.
    """
    risk = probs.reindex(rows.index)
    colours = [INTERVENE_C if a["level"] == "INTERVENE" else NOTIFY_C for a in alerts]
    return [
        go.Bar(
            x=rows.index,
            y=rows["rain_1h"],
            marker_color=RAIN,
            # bars are drawn ending at their hour, as rain is the hour's total
            width=HOUR_MS * 0.85,
            offset=-HOUR_MS * 0.85,
            name="rain (mm/h)",
            hovertemplate="%{x|%d %b %H:%M}<br>rain %{y:.1f} mm<extra></extra>",
        ),
        go.Scatter(
            x=risk.index,
            y=risk * 100,
            mode="lines",
            line={"color": PROB, "width": 2},
            name="spill chance in next 6 h",
            connectgaps=False,
            hovertemplate="%{x|%d %b %H:%M}<br>%{y:.1f}% chance<extra></extra>",
        ),
        go.Scatter(
            x=[a["timestamp"] for a in alerts],
            y=[100 * a["p"] for a in alerts],
            mode="markers",
            marker={
                "size": 11,
                "color": colours,
                "line": {"color": "white", "width": 2},
            },
            name="alert raised",
            hovertemplate="%{x|%d %b %H:%M}<br>alert at %{y:.1f}%<extra></extra>",
        ),
    ]


def _shapes(spills, thresholds, t0, t1):
    shapes = []
    for start, end in spills:
        # some groundwater-fed spills last weeks; only draw the replayed part
        shapes.append(
            {
                "type": "rect",
                "xref": "x2",
                "yref": "y2 domain",
                "x0": max(start, t0),
                "x1": min(end, t1),
                "y0": 0,
                "y1": 1,
                "fillcolor": SPILL_C,
                "opacity": 0.18,
                "line": {"width": 0},
                "layer": "below",
            }
        )
    for name, colour in [("notify", NOTIFY_C), ("intervene", INTERVENE_C)]:
        y = thresholds[name] * 100
        shapes.append(
            {
                "type": "line",
                "xref": "x2",
                "yref": "y2",
                "x0": t0,
                "x1": t1,
                "y0": y,
                "y1": y,
                "line": {"color": colour, "width": 1.5, "dash": "dash"},
            }
        )
    return shapes


def build(site_data, thresholds, t0, t1, title):
    """site_data: list of dicts with name, rows, probs, alerts, spills."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.3, 0.7], vertical_spacing=0.06
    )
    per_site = 3
    buttons = []
    for i, site in enumerate(site_data):
        for j, trace in enumerate(
            _site_traces(site["rows"], site["probs"], site["alerts"])
        ):
            trace.visible = i == 0
            trace.showlegend = i == 0
            fig.add_trace(trace, row=1 if j == 0 else 2, col=1)
        visible = [k // per_site == i for k in range(per_site * len(site_data))]
        buttons.append(
            {
                "label": site["name"],
                "method": "update",
                "args": [
                    {"visible": visible, "showlegend": visible},
                    {
                        "shapes": _shapes(site["spills"], thresholds, t0, t1),
                        "title.text": f"{title}<br><sup>{site['name']}</sup>",
                        "yaxis2.range": _y_range(site, thresholds),
                    },
                ],
            }
        )
    fig.update_layout(
        title={"text": f"{title}<br><sup>{site_data[0]['name']}</sup>", "x": 0.01},
        shapes=_shapes(site_data[0]["spills"], thresholds, t0, t1),
        updatemenus=[
            {
                "buttons": buttons,
                "x": 1,
                "xanchor": "right",
                "y": 1.13,
                "yanchor": "top",
                "showactive": True,
            }
        ],
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
        legend={"orientation": "h", "y": -0.1},
        margin={"t": 90, "r": 20, "l": 60, "b": 120},
        bargap=0,
        font={"family": "system-ui, sans-serif", "color": PROB},
        annotations=[
            {
                "text": "dashed lines: NOTIFY (amber) and INTERVENE (red) thresholds; "
                "orange bands: the overflow was really spilling",
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": -0.2,
                "showarrow": False,
                "font": {"size": 11, "color": MUTED},
                "xanchor": "left",
            }
        ],
    )
    fig.update_yaxes(title_text="rain (mm/h)", row=1, col=1, gridcolor=GRID)
    fig.update_yaxes(
        title_text="spill chance (%)",
        row=2,
        col=1,
        gridcolor=GRID,
        range=_y_range(site_data[0], thresholds),
    )
    fig.update_xaxes(gridcolor=GRID, range=[t0, t1])
    return fig
