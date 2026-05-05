"""
Shared helpers for Catch Theft: training-data profiles, Plotly visuals, and PDF export.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import fpdf
import networkx as nx
import pandas as pd
from fpdf import FPDF
import plotly.graph_objects as go

from middleware import ThreatDetectionAgent


def pdf_safe_text(value: str | None) -> str:
    """Normalize Unicode for PDF core fonts (Latin-1 safe).

    Args:
        value: Arbitrary user or model text; ``None`` treated as empty.

    Returns:
        A string encodable for standard PDF Helvetica output.
    """
    if value is None:
        return ""
    t = str(value)
    for a, b in (
        ("\u2014", "-"),
        ("\u2013", "-"),
        ("\u2019", "'"),
        ("\u2018", "'"),
        ("\u2032", "'"),
        ("\u2026", "..."),
        ("\u200b", ""),
        ("\ufeff", ""),
        ("\u26a0\ufe0f", "[!] "),
        ("\u26a0", "[!] "),
    ):
        t = t.replace(a, b)
    return t.encode("latin-1", errors="replace").decode("latin-1")


def report_address_suffix(address: str) -> str:
    """Return last four alphanumeric characters of *address* (uppercased) for filenames.

    Args:
        address: Ethereum-style or arbitrary address string.

    Returns:
        A short suffix, or ``"0000"`` if insufficient alphanumeric characters exist.
    """
    alnum = "".join(c for c in (address or "").strip() if c.isalnum())
    if len(alnum) >= 4:
        return alnum[-4:].upper()
    return alnum.upper() if alnum else "0000"


def profile_from_dataset(
    csv_path: Path,
    flag: Literal[0, 1],
    feature_names: tuple[str, ...],
) -> dict[str, float]:
    """Load one labeled row from the project CSV and project it onto *feature_names*.

    Args:
        csv_path: Path to ``transaction_dataset.csv`` (or compatible schema).
        flag: ``0`` for legitimate rows, ``1`` for fraud-labeled rows.
        feature_names: Column order expected by the trained estimator.

    Returns:
        Feature name to float mapping; if *csv_path* is missing, zeros for every name.

    Raises:
        KeyError: If required columns are absent after cleaning (unexpected for the shipped dataset).
    """
    if not csv_path.is_file():
        return {n: 0.0 for n in feature_names}

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.replace(r"\s+", " ", regex=True)
    drop_cols = [
        c
        for c in df.columns
        if c.lower() in ("index", "address", "unnamed: 0") or c == "Unnamed: 0"
    ]
    df = df.drop(columns=drop_cols, errors="ignore")
    y = df["FLAG"]
    x = df.drop(columns=["FLAG"])
    erc20_text = [
        c
        for c in x.columns
        if "erc20" in c.lower() and not pd.api.types.is_numeric_dtype(x[c])
    ]
    if erc20_text:
        x = x.drop(columns=erc20_text)
    x = x.select_dtypes(include=["number"])
    x = x[list(feature_names)]
    row = x.loc[y == flag].iloc[0]
    return {k: float(row[k]) for k in feature_names}


def short_feature_label(name: str, max_len: int = 24) -> str:
    """Truncate a long feature name for compact chart axis labels.

    Args:
        name: Full feature column label.
        max_len: Maximum grapheme length before ellipsis.

    Returns:
        Original *name* or a shortened variant with a trailing ellipsis marker.
    """
    return name if len(name) <= max_len else name[: max_len - 1] + "…"


def radar_dimension_names(
    xai: list[dict[str, Any]] | None,
    agent: ThreatDetectionAgent,
    limit: int = 8,
) -> list[str]:
    """Select polar-chart axes: XAI-ranked features first, then global importances.

    Args:
        xai: Optional list of XAI dicts with a ``\"name\"`` key.
        agent: Agent providing :meth:`~middleware.ThreatDetectionAgent.get_global_feature_importances`.
        limit: Maximum number of axis names to return.

    Returns:
        Ordered feature names (length at most *limit*).
    """
    keys: list[str] = []
    if xai:
        for item in xai:
            n = str(item.get("name", ""))
            if n and n not in keys:
                keys.append(n)
    for fname, _imp in agent.get_global_feature_importances():
        if fname not in keys:
            keys.append(fname)
        if len(keys) >= limit:
            break
    return keys[:limit]


def radar_series(
    keys: list[str], current: dict[str, float], baseline: dict[str, float]
) -> tuple[list[float], list[float]]:
    """Normalize current vs baseline magnitudes per axis to comparable [0, 1] ranges.

    Args:
        keys: Feature names defining radar axes.
        current: Observed feature values for the transaction under review.
        baseline: Reference benign profile values.

    Returns:
        Two parallel lists ``(current_normalized, baseline_normalized)``.
    """
    cur: list[float] = []
    bas: list[float] = []
    for k in keys:
        c = float(current.get(k, 0) or 0)
        b = float(baseline.get(k, 0) or 0)
        m = max(abs(c), abs(b), 1e-12)
        cur.append(float(abs(c) / m))
        bas.append(float(abs(b) / m))
    return cur, bas


def _network_center_label(target_address: str) -> str:
    """Build a short graph label for the focal wallet node.

    Args:
        target_address: Full on-chain address from the UI.

    Returns:
        Abbreviated label suitable for NetworkX node text.
    """
    a = (target_address or "").strip()
    if len(a) > 18:
        return f"{a[:8]}…{a[-6:]}"
    return a or "Target wallet"


def generate_network_graph(target_address: str, is_threat: bool) -> go.Figure:
    """Build a synthetic local transaction-path graph for demo storytelling.

    Uses ``networkx.spring_layout`` and Plotly scatter traces. Not derived from live
    chain analytics.

    Args:
        target_address: Wallet under review (graph center).
        is_threat: If ``True``, draw a denser high-risk motif; otherwise a benign star.

    Returns:
        A :class:`plotly.graph_objects.Figure` using the ``plotly_dark`` template.
    """
    g = nx.Graph()
    center = _network_center_label(target_address)
    g.add_node(center, node_role="center", threat=is_threat)

    if not is_threat:
        leaves = [
            "Binance Hot Wallet",
            "Coinbase",
            "User Wallet",
            "User Wallet 2",
            "User Wallet 3",
        ]
        for name in leaves:
            g.add_node(name, node_role="benign", threat=False)
            g.add_edge(center, name)
    else:
        mix = "Tornado Cash (Mixer)"
        phish = "Known Phishing Contract"
        dark = "Darkweb Entity"
        relay = "Suspicious Relay"
        peel = "Peel Chain Node"
        dust = "Dust / Hop Account"
        for n in (mix, phish, dark, relay, peel, dust):
            g.add_node(n, node_role="risk", threat=True)
        g.add_edge(center, relay)
        g.add_edge(center, peel)
        g.add_edge(relay, mix)
        g.add_edge(relay, phish)
        g.add_edge(peel, dark)
        g.add_edge(peel, dust)
        g.add_edge(dust, mix)
        g.add_edge(mix, phish)
        g.add_edge(phish, relay)
        g.add_edge(dark, mix)
        g.add_edge(center, dust)

    pos = nx.spring_layout(g, seed=42, k=0.9 if is_threat else 0.55, iterations=80)

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    ec = "#94a3b8" if is_threat else "#64748b"
    for u, v_e in g.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v_e]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line={"width": 1.8, "color": ec},
        hoverinfo="skip",
        mode="lines",
        showlegend=False,
    )

    node_x = [pos[n][0] for n in g.nodes()]
    node_y = [pos[n][1] for n in g.nodes()]
    colors: list[str] = []
    sizes: list[int] = []
    for n in g.nodes():
        role = g.nodes[n].get("node_role", "")
        thr = bool(g.nodes[n].get("threat", is_threat))
        if role == "center":
            colors.append("#f97316" if thr else "#2563eb")
            sizes.append(28)
        elif thr or role == "risk":
            if "Tornado" in n:
                colors.append("#dc2626")
            elif "Phishing" in n:
                colors.append("#eab308")
            elif "Darkweb" in n:
                colors.append("#b45309")
            else:
                colors.append("#9f1239")
            sizes.append(22)
        else:
            if "Binance" in n:
                colors.append("#16a34a")
            elif "Coinbase" in n:
                colors.append("#15803d")
            else:
                colors.append("#38bdf8")
            sizes.append(20)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=list(g.nodes()),
        textposition="top center",
        textfont={"size": 10, "color": "#e5e5e5"},
        hovertext=list(g.nodes()),
        hoverinfo="text",
        marker={
            "size": sizes,
            "color": colors,
            "line": {"width": 1.2, "color": "rgba(255,255,255,0.35)"},
        },
        showlegend=False,
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        template="plotly_dark",
        title={
            "text": (
                "Synthetic local network — high-risk exposure"
                if is_threat
                else "Synthetic local network — venue / retail flow"
            ),
            "font": {"size": 14, "color": "#e5e5e5"},
        },
        paper_bgcolor="#121212",
        plot_bgcolor="#121212",
        xaxis={"visible": False, "scaleanchor": "y", "scaleratio": 1},
        yaxis={"visible": False},
        margin={"l": 24, "r": 24, "t": 56, "b": 24},
        showlegend=False,
        height=520,
        hovermode="closest",
    )
    return fig


def generate_pdf_report(
    address: str,
    verdict: str,
    confidence: float,
    latency: float | int,
    xai_features: list[dict[str, Any]] | None,
    llm_report: str | None,
) -> bytes:
    """Render a compliance-style multi-section threat report as PDF bytes.

    Args:
        address: Wallet address under review.
        verdict: Display verdict string from the screening pipeline.
        confidence: Argmax class confidence as a percentage (0-100 scale).
        latency: End-to-end scan latency in milliseconds.
        xai_features: Ranked feature explanations, if any.
        llm_report: Optional narrative from the LLM layer.

    Returns:
        Raw PDF bytes suitable for HTTP download or attachment.
    """
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(18, 18, 18)

    font_dir = Path(fpdf.__file__).resolve().parent / "font"
    body_font = "Helvetica"
    try:
        dejavu_r = font_dir / "DejaVuSans.ttf"
        dejavu_b = font_dir / "DejaVuSans-Bold.ttf"
        if dejavu_r.is_file():
            pdf.add_font("DejaVu", "", str(dejavu_r))
            if dejavu_b.is_file():
                pdf.add_font("DejaVu", "B", str(dejavu_b))
            body_font = "DejaVu"
    except (OSError, ValueError):
        body_font = "Helvetica"

    pdf.add_page()

    pdf.set_font(body_font, "B", 20)
    pdf.set_text_color(24, 24, 24)
    pdf.cell(
        0,
        12,
        pdf_safe_text("CATCH THEFT THREAT INTELLIGENCE REPORT"),
        align="C",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.set_draw_color(200, 160, 40)
    pdf.set_line_width(0.4)
    pdf.line(18, pdf.get_y(), 192, pdf.get_y())
    pdf.ln(6)

    pdf.set_font(body_font, "", 9)
    pdf.set_text_color(80, 80, 80)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    pdf.cell(
        0, 5, pdf_safe_text(f"Document generated: {generated}"), new_x="LMARGIN", new_y="NEXT"
    )
    pdf.cell(
        0,
        5,
        pdf_safe_text(
            "Classification: CONFIDENTIAL - Internal Security & Compliance Review"
        ),
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(6)

    pdf.set_font(body_font, "B", 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, pdf_safe_text("1. TRANSACTION DETAILS"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font(body_font, "", 10)
    pdf.set_text_color(30, 30, 30)
    pdf.cell(
        0, 6, pdf_safe_text(f"Wallet address: {address or 'N/A'}"), new_x="LMARGIN", new_y="NEXT"
    )
    pdf.cell(
        0,
        6,
        pdf_safe_text(f"Processing latency: {float(latency):,.1f} ms"),
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.cell(
        0,
        6,
        pdf_safe_text(f"Risk / model confidence (argmax class): {float(confidence):.2f}%"),
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.cell(0, 6, pdf_safe_text(f"Final verdict: {verdict}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font(body_font, "B", 12)
    pdf.cell(
        0,
        8,
        pdf_safe_text("2. AI SECURITY ADVISOR (LAYER 3)"),
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.set_font(body_font, "", 10)
    advisory = (llm_report or "").strip() or "No LLM advisory was generated for this scan."
    pdf.multi_cell(0, 5, pdf_safe_text(advisory), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_font(body_font, "B", 12)
    pdf.cell(
        0,
        8,
        pdf_safe_text("3. CRITICAL FLAGS (XAI - TOP CONTRIBUTORS)"),
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.set_font(body_font, "", 10)
    if xai_features:
        for i, feat in enumerate(xai_features, start=1):
            name = str(feat.get("name", ""))
            val = feat.get("value")
            imp = feat.get("importance")
            score = feat.get("contribution_score")
            line = (
                f"{i}. Feature: {name}\n"
                f"   Observed value: {val} | Global importance: {imp} "
                f"| Contribution score: {score}"
            )
            pdf.multi_cell(0, 5, pdf_safe_text(line), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
    else:
        pdf.multi_cell(
            0,
            5,
            pdf_safe_text(
                "No ML-derived critical feature ranking for this outcome "
                "(e.g. verdict driven solely by rule-based blacklist, or benign classification)."
            ),
            new_x="LMARGIN",
            new_y="NEXT",
        )

    pdf.ln(6)
    pdf.set_font(body_font, "I", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(
        0,
        4,
        pdf_safe_text(
            "Disclaimer: This report is generated for operational security review. "
            "It does not constitute legal or investment advice. "
            "Retain in accordance with your organization's records policy."
        ),
        new_x="LMARGIN",
        new_y="NEXT",
    )

    raw = pdf.output(dest="S")
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw)
    return str(raw).encode("latin-1", errors="replace")


def synthetic_roc_curve_figure() -> go.Figure:
    """Return an illustrative ROC curve (PoC static points, ~AUC 0.985) for dashboards.

    Returns:
        Plotly figure with model trace and diagonal chance line.
    """
    fpr = [
        0.0,
        0.01,
        0.02,
        0.04,
        0.06,
        0.1,
        0.15,
        0.22,
        0.32,
        0.45,
        0.6,
        0.78,
        1.0,
    ]
    tpr = [
        0.0,
        0.78,
        0.88,
        0.92,
        0.94,
        0.955,
        0.965,
        0.972,
        0.978,
        0.981,
        0.983,
        0.986,
        1.0,
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name="Random Forest (illustrative)",
            line={"color": "#F0B90B", "width": 3},
            fill="tozeroy",
            fillcolor="rgba(240,185,11,0.12)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Chance",
            line={"color": "#6b7280", "width": 2, "dash": "dash"},
        )
    )
    fig.update_layout(
        title="Receiver Operating Characteristic (synthetic validation curve)",
        template="plotly_dark",
        paper_bgcolor="#121212",
        plot_bgcolor="#1a1a1a",
        font_color="#e5e5e5",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis={"range": [-0.02, 1.02], "gridcolor": "#333"},
        yaxis={"range": [-0.02, 1.02], "gridcolor": "#333"},
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.22, "x": 0},
        height=420,
        margin={"l": 48, "r": 24, "t": 56, "b": 72},
    )
    return fig


__all__ = [
    "generate_network_graph",
    "generate_pdf_report",
    "pdf_safe_text",
    "profile_from_dataset",
    "radar_dimension_names",
    "radar_series",
    "report_address_suffix",
    "short_feature_label",
    "synthetic_roc_curve_figure",
]
