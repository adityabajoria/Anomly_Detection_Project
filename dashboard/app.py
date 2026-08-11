import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, State, no_update
import dash_bootstrap_components as dbc

RESULTS_DIR = PROJECT_ROOT / "results"
SCORES_DIR = RESULTS_DIR / "scores"

MODEL_LABELS = {
    "zscore": "Z-Score",
    "pca": "PCA",
    "iforest": "Isolation Forest",
    "lstm_ae": "LSTM Autoencoder",
}

CARD_STYLE = {
    "backgroundColor": "#111827",
    "border": "1px solid #1f2937",
    "borderRadius": "16px",
    "padding": "20px",
    "height": "100%",
}

SIDEBAR_STYLE = {
    "backgroundColor": "#020617",
    "minHeight": "100vh",
    "padding": "28px",
    "borderRight": "1px solid #1f2937",
}

MAIN_STYLE = {"backgroundColor": "#030712", "minHeight": "100vh", "padding": "32px"}
MUTED = {"color": "#9ca3af"}


def machine_sort_key(name):
    nums = re.findall(r"\d+", name)
    return tuple(int(n) for n in nums) if nums else (0,)


def available_machines():
    return sorted((p.stem for p in RESULTS_DIR.glob("*.json") if p.stem != "summary"),
                  key=machine_sort_key)


def load_results(machine_id):
    with open(RESULTS_DIR / f"{machine_id}.json") as f:
        return json.load(f)


def load_scores(machine_id, model):
    data = np.load(SCORES_DIR / f"{machine_id}_{model}.npz")
    return data["scores"], data["labels"], float(data["threshold"])


def find_segments(binary):
    segs, in_seg, start = [], False, 0
    for i, v in enumerate(binary):
        if v == 1 and not in_seg:
            in_seg, start = True, i
        if in_seg and (v == 0 or i == len(binary) - 1):
            segs.append((start, i if v == 0 else i + 1))
            in_seg = False
    return segs


def build_timeline(scores, labels, threshold):
    fig = go.Figure()
    for start, end in find_segments(labels):
        fig.add_vrect(x0=start, x1=end, fillcolor="#ef4444", opacity=0.18, line_width=0)
    fig.add_trace(go.Scattergl(
        y=scores, mode="lines", name="anomaly score",
        line={"color": "#38bdf8", "width": 1},
    ))
    fig.add_hline(y=threshold, line_dash="dash", line_color="#f59e0b",
                  annotation_text="threshold", annotation_font_color="#f59e0b")
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 50, "r": 20, "t": 20, "b": 40},
        height=360,
        xaxis_title="timestep",
        yaxis_title="anomaly score",
        showlegend=False,
    )
    return fig


def build_replay_timeline(scores, labels, threshold, frame):
    fig = go.Figure()
    for start, end in find_segments(labels):
        fig.add_vrect(x0=start, x1=end, fillcolor="#ef4444", opacity=0.18, line_width=0)
    visible = scores[:frame]
    fig.add_trace(go.Scattergl(
        y=visible, mode="lines", name="anomaly score",
        line={"color": "#38bdf8", "width": 1},
    ))
    alarm_idx = np.where(visible >= threshold)[0]
    if len(alarm_idx) > 0:
        fig.add_trace(go.Scattergl(
            x=alarm_idx, y=visible[alarm_idx], mode="markers",
            marker={"color": "#ef4444", "size": 6}, name="alarm",
        ))
    fig.add_hline(y=threshold, line_dash="dash", line_color="#f59e0b")
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 50, "r": 20, "t": 20, "b": 40}, height=360,
        xaxis={"title": "timestep", "range": [0, len(scores)]},
        yaxis={"title": "anomaly score", "range": [0, float(scores.max()) * 1.05]},
        showlegend=False,
    )
    return fig


def kpi_card(title, value_id):
    return dbc.Col(dbc.Card(dbc.CardBody([
        html.H6(title, style=MUTED),
        html.H3(id=value_id),
    ]), style=CARD_STYLE), width=3)


def sidebar():
    machines = available_machines()
    return dbc.Col(
        [
            html.H2("Sentinel", style={"fontWeight": "700"}),
            html.P("Machine health monitoring", style={**MUTED, "fontSize": "14px"}),
            html.Hr(style={"borderColor": "#1f2937"}),
            html.Label("Machine", style={"fontWeight": "600"}),
            dcc.Dropdown(
                id="machine-dropdown",
                options=[{"label": m, "value": m} for m in machines],
                value=machines[0] if machines else None,
                clearable=False,
                style={"color": "#111827"},
            ),
            html.Br(),
            html.Label("Model", style={"fontWeight": "600"}),
            dcc.Dropdown(id="model-dropdown", clearable=False, style={"color": "#111827"}),
            html.Div(
                [
                    html.H6("About", style=MUTED),
                    html.P(
                        "Sentinel benchmarks anomaly detectors on multivariate "
                        "server telemetry. Results are precomputed by "
                        "run_experiments.py.",
                        style={"fontSize": "13px", **MUTED},
                    ),
                ],
                style={"marginTop": "40px"},
            ),
        ],
        width=2,
        style=SIDEBAR_STYLE,
    )


def main_column():
    return dbc.Col(
        [
            html.H1("Sentinel", style={"fontWeight": "800", "marginBottom": "4px"}),
            html.P("Anomaly detection for multivariate server telemetry.",
                   style={**MUTED, "marginBottom": "24px"}),
            dbc.Row(
                [
                    kpi_card("Honest F1", "kpi-honest-f1"),
                    kpi_card("Point-Adjusted F1", "kpi-adjusted-f1"),
                    kpi_card("PR-AUC", "kpi-pr-auc"),
                    kpi_card("Throughput (pts/s)", "kpi-throughput"),
                ],
                className="mb-4 g-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.H4("Anomaly Score Timeline", style={"fontWeight": "700"}),
                                html.P("Score per timestep. Shaded regions are true "
                                       "anomalies; dashed line is the decision threshold.",
                                       style=MUTED),
                                html.Div([
                                    dbc.Button("Play", id="play-btn", color="primary",
                                               size="sm", className="me-3"),
                                    html.Span(id="replay-progress", style=MUTED),
                                ], style={"marginBottom": "12px"}),
                                dcc.Interval(id="replay-interval", interval=80, disabled=True),
                                dcc.Store(id="replay-frame", data=0),
                                dcc.Graph(id="timeline-graph", config={"displayModeBar": False}),
                            ],
                            style=CARD_STYLE,
                        ),
                        width=8,
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                html.H4("Detection Summary", style={"fontWeight": "700"}),
                                html.Div(id="status-panel"),
                            ],
                            style=CARD_STYLE,
                        ),
                        width=4,
                    ),
                ],
                className="mb-4 g-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.H4("Model Metrics", style={"fontWeight": "700"}),
                                html.Div(id="metrics-table"),
                            ],
                            style=CARD_STYLE,
                        ),
                        width=6,
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                html.H4("Top Alert Regions", style={"fontWeight": "700"}),
                                html.Div(id="alerts-panel"),
                            ],
                            style=CARD_STYLE,
                        ),
                        width=6,
                    ),
                ],
                className="g-3",
            ),
        ],
        width=10,
        style=MAIN_STYLE,
    )


app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.layout = html.Div(
    style={"backgroundColor": "#030712", "color": "#f9fafb",
           "fontFamily": "Inter, sans-serif"},
    children=[dbc.Row([sidebar(), main_column()], className="g-0")],
)


@app.callback(
    Output("model-dropdown", "options"),
    Output("model-dropdown", "value"),
    Input("machine-dropdown", "value"),
)
def update_model_options(machine_id):
    if machine_id is None:
        return [], None
    models = list(load_results(machine_id).keys())
    options = [{"label": MODEL_LABELS.get(m, m), "value": m} for m in models]
    return options, models[0]


@app.callback(
    Output("replay-interval", "disabled"),
    Output("play-btn", "children"),
    Input("play-btn", "n_clicks"),
    State("replay-interval", "disabled"),
    prevent_initial_call=True,
)
def toggle_play(n_clicks, currently_disabled):
    now_playing = currently_disabled
    label = "Pause" if now_playing else "Play"
    return (not now_playing), label


@app.callback(
    Output("timeline-graph", "figure", allow_duplicate=True),
    Output("replay-frame", "data"),
    Output("replay-progress", "children"),
    Input("replay-interval", "n_intervals"),
    State("replay-frame", "data"),
    State("machine-dropdown", "value"),
    State("model-dropdown", "value"),
    prevent_initial_call=True,
)
def advance_replay(n_intervals, frame, machine_id, model):
    if not machine_id or not model:
        return no_update, no_update, no_update
    scores, labels, threshold = load_scores(machine_id, model)
    frame = (frame or 0) + 40
    if frame >= len(scores):
        frame = len(scores)
    fig = build_replay_timeline(scores, labels, threshold, frame)
    progress = f"t = {frame:,} / {len(scores):,}"
    return fig, frame, progress


@app.callback(
    Output("timeline-graph", "figure"),
    Output("kpi-honest-f1", "children"),
    Output("kpi-adjusted-f1", "children"),
    Output("kpi-pr-auc", "children"),
    Output("kpi-throughput", "children"),
    Output("metrics-table", "children"),
    Output("status-panel", "children"),
    Output("alerts-panel", "children"),
    Input("machine-dropdown", "value"),
    Input("model-dropdown", "value"),
)
def update_view(machine_id, model):
    if not machine_id or not model:
        empty = go.Figure()
        return empty, "-", "-", "-", "-", None, None, None
    result = load_results(machine_id)[model]
    try:
        scores, labels, threshold = load_scores(machine_id, model)
    except FileNotFoundError:
        scores = None
    if scores is None:
        fig = go.Figure()
    else:
        fig = build_timeline(scores, labels, threshold)
    honest, adjusted = result["honest"], result["point_adjusted"]
    table = dbc.Table(
        [
            html.Thead(html.Tr([html.Th(""), html.Th("Honest"), html.Th("Point-Adjusted")])),
            html.Tbody([
                html.Tr([html.Td(name),
                         html.Td(f"{honest[key]:.3f}"),
                         html.Td(f"{adjusted[key]:.3f}")])
                for name, key in [("Precision", "precision"),
                                  ("Recall", "recall"),
                                  ("F1", "f1")]
            ]),
        ],
        bordered=False, color="dark", size="sm", className="mb-0",
    )
    pred = (scores >= threshold).astype(int)
    true_segs = find_segments(labels)
    pred_segs = find_segments(pred)
    caught = sum(1 for s, e in true_segs if pred[s:e].any())
    status = html.Div([
        html.Div(f"{caught} / {len(true_segs)}",
                 style={"fontSize": "42px", "fontWeight": "800", "color": "#22c55e"}),
        html.P("true anomaly segments detected", style=MUTED),
    ])
    ranked = sorted(pred_segs, key=lambda s: float(scores[s[0]:s[1]].max()), reverse=True)[:5]
    alerts = html.Ul(
        [html.Li(f"t = {s:,} - {e:,}  peak score {scores[s:e].max():.2f}")
         for s, e in ranked],
        style={"paddingLeft": "20px"},
    ) if ranked else html.P("No alerts.", style=MUTED)
    return (fig, f"{honest['f1']:.3f}", f"{adjusted['f1']:.3f}",
            f"{result['pr_auc']:.3f}", f"{result['throughput_pts_per_sec']:,}",
            table, status, alerts)


if __name__ == "__main__":
    app.run(debug=True)