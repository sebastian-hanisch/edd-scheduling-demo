"""Plotly-Abbildungen der EDD-Demo: Gantt-artiges Balkendiagramm, Fälligkeits-/Fertigstellungsvergleich, Sweeps,
Timing. Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen."""

import numpy as np
import plotly.graph_objects as go

EDD_COLOR = "#4c78a8"
SPT_COLOR = "#e45756"
RANDOM_COLOR = "#7f7f7f"
DUE_COLOR = "#54a24b"
SHIFT_COLOR = "#f58518"


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.15), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_schedule(p, order, completion, upto=None, color=EDD_COLOR):
    """`completion`: die TATSÄCHLICHEN Fertigstellungszeiten (aus `evaluate_order`/`evaluate_order_with_shifts`) -
    auf dem Werkstatt/Logistik-Vehikel enthalten sie Lücken durch Schichtgrenzen, sichtbar als Leerraum."""
    order = np.asarray(order)
    upto = len(order) if upto is None else upto
    starts = np.asarray(completion) - p[order]
    fig = go.Figure()
    for i in range(upto):
        j = order[i]
        fig.add_trace(go.Bar(x=[float(p[j])], y=["Maschine"], base=[float(starts[i])], orientation="h",
                              marker=dict(color=color, line=dict(width=1, color="white")),
                              name=f"Auftrag {j}", hovertemplate=f"Auftrag {j}<br>Dauer {p[j]}<extra></extra>", showlegend=False))
    fig.update_xaxes(title_text="Zeit")
    fig.update_yaxes(showticklabels=False)
    return _base(fig, 180)


def build_completion_vs_due(completion, due, order):
    x = np.arange(1, len(order) + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=completion, mode="lines+markers", line=dict(color=EDD_COLOR, width=2.5), name="Fertigstellung"))
    fig.add_trace(go.Scatter(x=x, y=due[order], mode="markers", marker=dict(color=DUE_COLOR, size=8, symbol="diamond"), name="Fälligkeit"))
    fig.update_xaxes(title_text="Position in der Reihenfolge")
    fig.update_yaxes(title_text="Zeit")
    return _base(fig, 320)


def build_sweep(rows, param_label, value_key="value", y_keys=(("gap_spt", "EDD gegen SPT", SPT_COLOR), ("gap_random", "EDD gegen Zufall", RANDOM_COLOR))):
    xs = [r[value_key] for r in rows]
    fig = go.Figure()
    for key, name, color in y_keys:
        fig.add_trace(go.Scatter(x=xs, y=[r[key] for r in rows], mode="lines+markers", line=dict(color=color, width=2.5), name=name))
    fig.update_xaxes(title_text=param_label)
    fig.update_yaxes(title_text="Lmax-Abstand zu EDD (Minuten)")
    fig.update_layout(legend=dict(orientation="h", y=-0.3))
    return _base(fig, 360)


def build_timing(rows):
    xs = [r["value"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[r["brute_force_seconds"] * 1000 for r in rows], mode="lines+markers", line=dict(color=SPT_COLOR, width=2.5), name="Brute-Force (O(n!))"))
    fig.add_trace(go.Scatter(x=xs, y=[r["edd_seconds"] * 1000 for r in rows], mode="lines+markers", line=dict(color=EDD_COLOR, width=2.5), name="EDD (O(n log n))"))
    fig.update_xaxes(title_text="Aufträge")
    fig.update_yaxes(title_text="Rechenzeit (ms)", type="log")
    fig.update_layout(legend=dict(orientation="h", y=-0.3))
    return _base(fig, 340)


def build_shift_gap(rows):
    xs = [r["value"] for r in rows]
    upper = [r["gap_max"] for r in rows]
    lower = [r["gap_min"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs + xs[::-1], y=upper + lower[::-1], fill="toself", fillcolor="rgba(245,133,24,0.15)", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=[r["gap_mean"] for r in rows], mode="lines+markers", line=dict(color=SHIFT_COLOR, width=2.5), name="EDD über dem echten Optimum (Lmax-Differenz)"))
    fig.update_xaxes(title_text="Schichtlänge (Minuten)", autorange="reversed")
    fig.update_yaxes(title_text="Abstand zum Optimum mit Schichten (Minuten)")
    return _base(fig, 340)
