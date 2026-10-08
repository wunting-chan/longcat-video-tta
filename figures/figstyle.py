"""House figure style for Artificial Individuality, after DeepSeek's technical reports (docs/FIGURE_STYLE.md).

Measured from DeepSeek-V3 (arXiv 2412.19437), DeepSeek-R1 (2501.12948), Janus-Pro (2501.17811), DeepSeek-VL2
(2412.10302) and DeepSeek-OCR (2510.18234). Use one template per figure type:

    import figstyle as fs
    fs.use("bar")                       # or "line", "scatter", "heatmap", "image"
    fig, ax = plt.subplots(figsize=fs.size("full", 0.45))
    fs.grouped_bars(ax, groups, series, values, lead="DINO + rule", ...)
    fs.top_legend(fig, ax)
    fs.save(fig, "out/fig_name")       # writes .pdf and .png; refuses unlabeled axes

Every axis states what it measures and its unit; every colour, line style, marker and band is named in the
legend or the caption. ``fs.save`` checks for axis labels and fails loudly if one is missing.
"""
from __future__ import annotations

import os
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle  # noqa: E402

# --------------------------------------------------------------------------- palette
BLUE = "#4D6BFE"        # lead series: our model / the condition the figure is about (DeepSeek blue, white hatching)
LIGHT_BLUE = "#AAC1FF"  # closest counterpart (previous version, ablation, plain DINO)
GRAY = "#BDBDBD"        # other baselines
LIGHT_GRAY = "#D4D4D4"
TAN = "#E8D2A0"
CREAM = "#F5EBD2"
HUMAN = "#E0E0E0"       # reference populations (human experts, chance)
COMPANIONS = [LIGHT_BLUE, GRAY, LIGHT_GRAY, TAN, CREAM]
TEXT = "#262626"
GRID_BAR = "#E8E8E8"
SPINE_BAR = "#CCCCCC"
GRID_LINE = "#C8C8C8"
HIGHLIGHT_GREEN = "#1AAE6F"   # key phrase in a qualitative answer (Janus-Pro / VL2)
STAR_RED = "#E8222D"          # our points in a scatter (red star, black edge)
REFERENCE = "#2CA02C"         # dashed reference line (human level, chance, baseline)
LINE_SERIES = ["#1F3FD1", "#E8222D", "#2CA02C", "#7F3FBF", "#F28E2B", "#555555"]   # R1-style primary lines
MARKERS = ["o", "s", "D", "^", "v", "P"]


def blues(n):
    """Sequential ramp for ordered categories (R1 difficulty levels): light -> dark."""
    return [plt.get_cmap("Blues")(x) for x in np.linspace(0.35, 0.95, n)]


HEATMAP_CMAP = "YlOrRd"      # V3 expert-load heatmap; diverging data: "RdBu_r" centred at 0

# --------------------------------------------------------------------------- sizes (inches)
WIDTHS = {"full": 6.27, "half": 3.05, "slide": 4.25}   # A4, 1-in margins; half column; 128 x 96 mm beamer frame


def size(width="full", aspect=0.5):
    w = WIDTHS[width]
    return (w, w * aspect)


# --------------------------------------------------------------------------- templates
_BASE = {"pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 150, "text.color": TEXT,
         "axes.labelcolor": TEXT, "xtick.color": TEXT, "ytick.color": TEXT, "axes.unicode_minus": True}

TEMPLATES = {
    # V3 / R1 benchmark bars: serif, bold axis titles, faint dashed horizontal grid, light spines all round
    "bar": {"font.family": "DejaVu Serif", "font.size": 7.5, "axes.labelweight": "bold", "axes.labelsize": 7.5,
            "axes.edgecolor": SPINE_BAR, "axes.linewidth": 0.6, "axes.grid": True, "axes.grid.axis": "y",
            "grid.color": GRID_BAR, "grid.linestyle": "--", "grid.linewidth": 0.5, "axes.axisbelow": True,
            "xtick.major.size": 0, "ytick.major.size": 2, "ytick.major.width": 0.5, "hatch.color": "white",
            "hatch.linewidth": 0.6, "legend.fontsize": 7, "legend.frameon": True, "legend.edgecolor": "#E5E5E5",
            "legend.fancybox": False, "axes.titlesize": 8, "axes.titleweight": "bold"},
    # R1 training dynamics / ablations: sans, black thin axes, light grid, markers on every point
    "line": {"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": "black", "axes.linewidth": 0.6,
             "axes.grid": True, "grid.color": GRID_LINE, "grid.linestyle": "--", "grid.linewidth": 0.5,
             "axes.axisbelow": True, "lines.linewidth": 1.3, "lines.markersize": 4.5, "xtick.major.size": 2.5,
             "ytick.major.size": 2.5, "legend.fontsize": 7, "legend.framealpha": 0.9, "legend.edgecolor": "#CCCCCC",
             "axes.titlesize": 8, "axes.titleweight": "normal"},
    # Janus-Pro / VL2 Figure 1 scatter: sans, bold axis titles, labelled points, family lines dashed
    "scatter": {"font.family": "DejaVu Sans", "font.size": 7.5, "axes.labelweight": "bold", "axes.edgecolor": "black",
                "axes.linewidth": 0.6, "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.5,
                "axes.axisbelow": True, "legend.fontsize": 6.5, "legend.framealpha": 0.9, "legend.edgecolor": "#CCCCCC"},
    # V3 expert-load heatmap: sans, no grid, row labels large, horizontal colour bar underneath
    "heatmap": {"font.family": "DejaVu Sans", "font.size": 7.5, "axes.grid": False, "axes.linewidth": 0,
                "xtick.major.size": 0, "ytick.major.size": 0, "axes.titlesize": 8},
    # Janus-Pro qualitative grids: serif headers and captions, no axes
    "image": {"font.family": "DejaVu Serif", "font.size": 7.5, "axes.grid": False},
}


def use(kind):
    """Reset rcParams and apply one template."""
    plt.rcParams.update(matplotlib.rcParamsDefault)
    plt.rcParams.update(_BASE)
    plt.rcParams.update(TEMPLATES[kind])


# --------------------------------------------------------------------------- bars
def grouped_bars(ax, groups, series, values, errors=None, lead=None, colors=None, metric_notes=None,
                 decimals=1, label_values=True, width=0.8, ylim=None):
    """V3 Figure 1: one cluster per group, one bar per series; the lead series is BLUE with white hatching and its
    value labels are bold. ``values[s][g]``; ``errors[s][g]`` (optional) draws thin black whiskers.
    ``metric_notes[g]`` is the small italic metric under each group name, e.g. "(Pass@1)"."""
    n = len(series); w = width / n; x = np.arange(len(groups))
    colors = colors or {}
    companion = iter(COMPANIONS)
    handles = []
    for i, s in enumerate(series):
        c = colors.get(s) or (BLUE if s == lead else next(companion))
        v = np.asarray(values[s], float); xs = x - width / 2 + w * (i + 0.5)
        bars = ax.bar(xs, v, w * 0.92, color=c, hatch="///" if s == lead else None, edgecolor="white" if s == lead else c,
                      linewidth=0, zorder=2, label=s)
        if errors is not None and s in errors:
            ax.errorbar(xs, v, yerr=errors[s], fmt="none", ecolor="#404040", elinewidth=0.6, capsize=1.5, zorder=3)
        if label_values:
            top = v + (np.asarray(errors[s]) if errors is not None and s in errors else 0)
            for xi, vi, ti in zip(xs, v, top):
                ax.annotate(f"{vi:.{decimals}f}", (xi, ti), xytext=(0, 1.5), textcoords="offset points", ha="center",
                            va="bottom", fontsize=6.6 if s == lead else 5.6, fontweight="bold" if s == lead else "normal")
        handles.append(bars)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}" for g in groups], fontweight="bold")
    if metric_notes:
        for xi, note in zip(x, metric_notes):
            ax.annotate(note, (xi, 0), xycoords=("data", "axes fraction"), xytext=(0, -17), textcoords="offset points",
                        ha="center", va="top", fontsize=5.8, style="italic")
    if ylim:
        ax.set_ylim(*ylim)
    return handles


def top_legend(fig_or_ax, ax=None, ncol=None, handles=None, labels=None, y=1.0):
    """One row across the top of the plot, framed (V3/R1)."""
    ax = ax or fig_or_ax
    h, l = (handles, labels) if handles is not None else ax.get_legend_handles_labels()
    return ax.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol or len(h), handlelength=1.6,
                     columnspacing=1.4, borderpad=0.35, prop={"weight": "bold", "size": plt.rcParams["legend.fontsize"]})


# --------------------------------------------------------------------------- lines
def line_series(ax, x, ys, labels, colors=None, markers=None, bands=None, dashed=None):
    """R1-style lines: primary colours, a marker on every point, optional light band (same colour, alpha 0.18)."""
    colors = colors or LINE_SERIES; markers = markers or MARKERS; dashed = dashed or set()
    for i, (y, lab) in enumerate(zip(ys, labels)):
        c = colors[i % len(colors)]
        ax.plot(x, y, color=c, marker=markers[i % len(markers)], linestyle="--" if lab in dashed else "-", label=lab, zorder=3)
        if bands is not None and bands[i] is not None:
            lo, hi = bands[i]; ax.fill_between(x, lo, hi, color=c, alpha=0.18, linewidth=0, zorder=2)


def reference_line(ax, y, label, color=REFERENCE):
    """Dashed horizontal reference (R1 'human participants', our 'chance' or 'no history effect')."""
    ax.axhline(y, color=color, linestyle="--", linewidth=1.0, label=label, zorder=1)


# --------------------------------------------------------------------------- scatter
def scatter_points(ax, x, y, names, lead=(), families=None, colors=None, label_offsets=None):
    """Janus-Pro / VL2 Figure 1: lead points as red stars with black edges, others as coloured dots; every point is
    labelled; points of one family are joined by a thin dashed line (``families`` = {family: [names...]})."""
    colors = colors or {}; label_offsets = label_offsets or {}
    pos = {n: (xi, yi) for n, xi, yi in zip(names, x, y)}
    if families:
        for k, (fam, members) in enumerate(families.items()):
            pts = np.array([pos[m] for m in members if m in pos])
            ax.plot(pts[:, 0], pts[:, 1], "--", color=colors.get(fam, LINE_SERIES[k % len(LINE_SERIES)]), linewidth=0.8,
                    alpha=0.8, label=fam, zorder=1)
    for n, (xi, yi) in pos.items():
        if n in lead:
            ax.scatter([xi], [yi], marker="*", s=150, color=STAR_RED, edgecolor="black", linewidth=0.6, zorder=4)
        else:
            ax.scatter([xi], [yi], s=22, color=colors.get(n, "#F28E2B"), edgecolor="black", linewidth=0.4, zorder=3)
        dx, dy = label_offsets.get(n, (0, 6))
        ax.annotate(n, (xi, yi), xytext=(dx, dy), textcoords="offset points", ha="center", fontsize=6.3)


# --------------------------------------------------------------------------- heatmap
def heatmap(ax, M, row_labels, col_labels=None, cmap=HEATMAP_CMAP, vmin=None, vmax=None, title=None):
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=7.5)
    if col_labels is not None:
        ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, fontsize=5.5)
    if title:
        ax.set_title(title)
    return im


def bottom_colorbar(fig, im, label, rect=(0.3, 0.06, 0.45, 0.025)):
    """Horizontal colour bar under the panels with its label to the left (V3 Figure 9)."""
    cax = fig.add_axes(rect); cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cax._figstyle_exempt = True
    cb.outline.set_linewidth(0.5); cax.tick_params(labelsize=6.5)
    cax.annotate(label, (0, 0.5), xycoords="axes fraction", xytext=(-6, 0), textcoords="offset points", ha="right",
                 va="center", fontsize=7.5)
    return cb


# --------------------------------------------------------------------------- images
def image_cell(ax, img, caption=None, header=None, frame=None, caption_width=26):
    """One image tile: no axes; header above (method/condition, serif), caption below (class names or prompt).
    ``frame`` draws a thin coloured border (BLUE marks the lead condition)."""
    ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(frame is not None); s.set_edgecolor(frame or "white"); s.set_linewidth(1.6)
    if header:
        ax.set_title(header, fontsize=7.5, pad=3)
    if caption:
        lines = [textwrap.fill(line, caption_width, break_on_hyphens=False) for line in caption.split("\n")]
        ax.set_xlabel("\n".join(lines), fontsize=6.5, labelpad=2)
    ax._figstyle_image = True


def title_tab(fig, rect, title, pad=0.004):
    """VL2 / Janus-Pro capability box: thin black outline with a black title tab at the top-left."""
    x, y, w, h = rect
    fig.patches.append(Rectangle((x, y), w, h, transform=fig.transFigure, fill=False, edgecolor="black", linewidth=0.7))
    t = fig.text(x + pad, y + h, f" {title} ", ha="left", va="center", fontsize=6.5, color="white",
                 family="DejaVu Sans", bbox=dict(boxstyle="square,pad=0.25", facecolor="black", edgecolor="black"))
    return t


def prompt_bubble(ax, text, xy=(0.5, -0.08)):
    """Light-blue rounded bubble for a query or prompt under an image (VL2 / Janus-Pro)."""
    ax.annotate(text, xy, xycoords="axes fraction", ha="center", va="top", fontsize=6.5, family="DejaVu Sans",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#E9F0FF", edgecolor="none"))


def panel_letter(ax, letter, x=-0.12, y=1.04):
    """Sub-figure tag "(a)" placed above-left; the sub-caption itself goes in the LaTeX caption."""
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")


# --------------------------------------------------------------------------- output
def check_labels(fig):
    """Every data axis needs an x and y label (images and colour bars are exempt)."""
    missing = []
    for ax in fig.axes:
        if getattr(ax, "_figstyle_image", False) or getattr(ax, "_figstyle_exempt", False) or not ax.get_visible():
            continue
        if ax.get_label() == "<colorbar>" or (ax.get_xlabel() == "" and ax.get_ylabel() == "" and not ax.has_data()):
            continue
        shared_y = any(o.get_ylabel().strip() for o in ax.get_shared_y_axes().get_siblings(ax) if o is not ax)
        shared_x = any(o.get_xlabel().strip() for o in ax.get_shared_x_axes().get_siblings(ax) if o is not ax)
        if not (ax.get_ylabel().strip() or shared_y) or not (ax.get_xlabel().strip() or shared_x or ax.get_xticklabels()):
            missing.append(ax.get_title() or repr(ax))
    if missing:
        raise ValueError(f"axes without a full label: {missing}")


def exempt(ax):
    """Mark an axis that intentionally has no labels (a schematic or a network drawing)."""
    ax._figstyle_exempt = True


def save(fig, path, check=True):
    """Write ``path.pdf`` (vector, fonts embedded) and ``path.png`` (300 dpi)."""
    if check:
        check_labels(fig)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path + ".pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(path + ".png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)


def caption(n, lead_sentence, body):
    """LaTeX caption in the DeepSeek form: 'Figure n | **Lead sentence.** How to read it.'"""
    return f"\\caption*{{Figure {n} $|$ \\textbf{{{lead_sentence}}} {body}}}"
