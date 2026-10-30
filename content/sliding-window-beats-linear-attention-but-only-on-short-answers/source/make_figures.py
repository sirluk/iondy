"""Figures for the post "Sliding window beats linear attention, but only on short answers".

Run from anywhere (needs matplotlib):
    python -I make_figures.py

Writes every PNG the post shows into ../images, 1440 px wide on a white background.

Decode speeds for SWA and Bifocal in all mixing layers come from data/speed_all_layers.csv (which
also holds their versions in Qwen3.5's 24 + 8 layout). The GDN + attention and full-attention
speeds are written out in fig_decode_throughput. All: one H100, batch 1, Qwen3.5-4B-shaped models
with synthetic weights, vLLM. Quality numbers come from data/post_numbers.json, which
analysis/post_numbers.py writes.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

SOURCE = Path(__file__).resolve().parent
DATA = SOURCE / "data"
OUT = SOURCE.parent / "images"

# Colors follow the entity through the whole post.
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
EMPTY = "#f1f0ec"
REST = "#dddcd5"
HYBRID = "#2a78d6"  # Bifocal (ours)
SWA = "#eb6834"  # sliding window with sinks
GDN = "#1baf7a"  # Qwen3.5's own layout: 3 Gated DeltaNet layers : 1 full attention layer
FULL = "#898781"  # full softmax attention

# Names readers see. Bifocal = sliding window + linear mixer (our mLSTM) in parallel in every
# layer: sharp near, compressed far. The text introduces both terms once.
NAME_HYBRID = "Bifocal (ours)"
NAME_GDN = "GDN + attention"
NAME_FULL = "Full attention"

mpl.rcParams.update(
    {
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "text.color": INK,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2,
        "axes.facecolor": SURFACE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": BASELINE,
        "ytick.color": BASELINE,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "figure.facecolor": SURFACE,
        "savefig.dpi": 200,
        "savefig.facecolor": SURFACE,
    }
)


def header(fig, title, subtitle, top_in=0.12, gap_in=0.3):
    h = fig.get_figheight()
    if title:
        fig.text(0.03, 1 - top_in / h, title, fontsize=13, color=INK, va="top")
        sub_y = 1 - (top_in + gap_in) / h
    else:
        sub_y = 1 - top_in / h
    fig.text(0.03, sub_y, subtitle, fontsize=9.5, color=INK_2, va="top", linespacing=1.35)


def hairline_grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def data_units_per_inch(ax):
    bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    return (x1 - x0) / bbox.width, abs(y1 - y0) / bbox.height


def bar_h(ax, y, left, width, height, color, round_end=True, radius_in=0.035):
    """Horizontal bar, rounded at its data end and square at the baseline."""
    ux, uy = data_units_per_inch(ax)
    r = radius_in * ux
    if not round_end or width <= 2 * r:
        ax.add_patch(Rectangle((left, y - height / 2), width, height, color=color, lw=0))
        return
    ax.add_patch(
        FancyBboxPatch(
            (left, y - height / 2),
            width,
            height,
            boxstyle=f"round,pad=0,rounding_size={r}",
            mutation_aspect=uy / ux,
            color=color,
            lw=0,
        )
    )
    ax.add_patch(Rectangle((left, y - height / 2), 2 * r, height, color=color, lw=0))


def value_labels(ax, x, y, value, note, pad_pt=6):
    """Value at the bar tip in primary ink, a note right after it in secondary ink."""
    t = ax.annotate(value, (x, y), xytext=(pad_pt, 0), textcoords="offset points", va="center", fontsize=10, color=INK)
    ax.figure.canvas.draw()
    right = t.get_window_extent().x1
    x_note = ax.transData.inverted().transform((right, 0))[0]
    ax.annotate(note, (x_note, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=9, color=INK_2)


def cell(ax, x, y, color, alpha=1.0):
    ax.add_patch(Rectangle((x + 0.07, y + 0.07), 0.86, 0.86, color=color, alpha=alpha, lw=0))


def fmt_gain(g):
    g = round(g + (1e-9 if g >= 0 else -1e-9), 1)
    return "0.0" if g == 0 else f"{g:+.1f}".replace("-", "−")


def fmt_tokens(n):
    return f"{n / 1000:.1f}K" if n >= 1000 else f"{n:.0f}"


# ----------------------------------------------------------------------------------
# Concept: what each layer stores while generating
# ----------------------------------------------------------------------------------

def fig_memory():
    """What each layer type keeps in memory while generating."""
    rows = [
        ("Full attention", "Keeps every token.\nGrows with sequence length.", "full"),
        ("Sliding window + sink tokens", "The first tokens + the\nlast W tokens. Fixed size.", "swa"),
        ("Sliding window + learned sink", "A learned sink + the\nlast W tokens. Fixed size.", "swa_learned"),
        ("Linear mixer (mLSTM)", "One matrix state.\nFixed size.", "lin"),
        (NAME_HYBRID, "Learned sink + last W tokens\n+ one matrix state. Fixed size.", "hyb"),
    ]
    n_full, w, d = 26, 7, 3
    fig, ax = plt.subplots(figsize=(7.2, 3.85))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.02)
    header(
        fig,
        None,
        "Dot: learned sink (scalar). Squares: stored tokens (vectors). Block: the linear mixer's state (matrix).",
    )
    ax.set_xlim(-15, n_full + 4)
    ax.set_ylim(-4.4 * (len(rows) - 1) - 2.4, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    for k, (title, note, kind) in enumerate(rows):
        y = -4.4 * k  # row center
        ax.text(-14.5, y + 0.15, title, ha="left", va="bottom", fontsize=9.5, color=INK)
        ax.text(-14.5, y - 0.15, note, ha="left", va="top", fontsize=8.5, color=INK_2, linespacing=1.3)
        if kind == "full":
            for i in range(n_full):
                cell(ax, i, y - 0.5, FULL)
            ax.annotate("", xy=(n_full + 3, y), xytext=(n_full + 0.3, y),
                        arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))
        if kind == "swa":
            # StreamingLLM-style sinks: the first tokens stay in the cache next to the window.
            for i in range(2):
                cell(ax, i, y - 0.5, SWA, 0.35)
            ax.text(1, y + 0.6, "sink tokens", ha="center", va="bottom", fontsize=8, color=INK_2)
        if kind in ("swa_learned", "hyb"):
            # gpt-oss-style sink: one learned scalar per head, not a cached token.
            ax.add_patch(Circle((1, y), 0.38, color=SWA, alpha=0.5, lw=0))
            ax.text(1, y + 0.6, "learned sink", ha="center", va="bottom", fontsize=8, color=INK_2)
        if kind in ("swa", "swa_learned", "hyb"):
            for i in range(w):
                cell(ax, 2.6 + i, y - 0.5, SWA)
        if kind in ("lin", "hyb"):
            x0 = 0 if kind == "lin" else 2.6 + w + 1
            for i in range(d):
                for j in range(d):
                    cell(ax, x0 + i, y - d / 2 + j, HYBRID)
    fig.savefig(OUT / "fig_memory.png")
    plt.close(fig)


# ----------------------------------------------------------------------------------
# Speed: decode throughput against prompt length
# ----------------------------------------------------------------------------------

SPEED_ALL = DATA / "speed_all_layers.csv"
CTX = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]


def decode_speeds(model):
    """Median decode tokens/s of one model at each prompt length in CTX."""
    import csv

    rows = {int(r["prompt_length"]): float(r["decode_tokens_s_median"])
            for r in csv.DictReader(open(SPEED_ALL)) if r["model"] == model}
    return [rows[c] for c in CTX]


def fig_decode_throughput():
    """Decode speed against prompt length: SWA and Bifocal in all attention and GDN layers, at the same cache."""
    lines = [
        # name, color, tokens/s
        (NAME_SWA_LEARNED, SWA, decode_speeds("swa576_all")),
        (NAME_HYBRID, HYBRID, decode_speeds("fused_kv320_all")),
        # Measured earlier on another H100 with the same software versions, as the post's Speed
        # notes say. A later run of GDN + attention on the GPU of the two lines above gave 182.9
        # and 145.8 tokens/s at 8K and 128K.
        (f"{NAME_GDN} (default)", GDN, [186.8, 184.8, 184.4, 183.0, 179.4, 173.7, 162.9, 146.0]),
        (NAME_FULL, FULL, [200.0, 196.6, 193.2, 186.1, 172.0, 153.2, 125.2, 92.0]),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    fig.subplots_adjust(left=0.1, right=0.97, top=0.81, bottom=0.14)
    header(
        fig,
        None,
        "Qwen3.5-4B shape, one H100, batch 1. SWA or Bifocal in all attention and GDN layers, same cache.",
    )
    for name, color, ys in lines:
        ax.plot(CTX, ys, color=color, lw=2, solid_capstyle="round", solid_joinstyle="round", zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(CTX)
    ax.set_xticklabels([f"{c // 1024}K" for c in CTX])
    ax.minorticks_off()
    ax.set_xlim(CTX[0] / 1.15, CTX[-1] * 1.15)
    ax.set_ylim(0, 225)
    ax.set_yticks([0, 50, 100, 150, 200])
    ax.set_xlabel("Prompt length (tokens)")
    ax.set_ylabel("Decode throughput (tokens/s)")
    hairline_grid(ax)
    legend = [lines[i] for i in (2, 1, 0, 3)]  # the default layout first; drawing order stays as above
    fig.legend(
        [Line2D([], [], color=c, lw=2) for _, c, _ in legend],
        [name for name, _, _ in legend],
        loc="upper left",
        bbox_to_anchor=(0.025, 0.90),
        ncol=4,
        frameon=False,
        fontsize=9,
        handlelength=1.6,
        columnspacing=1.4,
    )
    fig.savefig(OUT / "fig_decode_throughput.png")
    plt.close(fig)


# ----------------------------------------------------------------------------------
# Quality, matched memory: four Qwen3-4B-Instruct students with 576 tokens of cache per layer.
# Numbers come from data/post_numbers.json (analysis/post_numbers.py).
# Main comparison: Bifocal (448 window + state) vs. SWA with a learned sink (576 window).
# ----------------------------------------------------------------------------------

NUMBERS = DATA / "post_numbers.json"
NAME_SWA_LEARNED = "SWA, learned sink"
NAME_SWA_SINK = "SWA + sink tokens"
NAME_Q64 = "Bifocal, window 64"
SCORE = "#52514e"  # one neutral series
MATCHED = [
    # key, name, window label, memory composition: (sink tokens, window tokens, state tokens)
    ("swa576", NAME_SWA_LEARNED, "576 window", (0, 576, 0)),
    ("swa572", NAME_SWA_SINK, "572 window", (4, 572, 0)),
    ("kv448", NAME_HYBRID, "448 window", (0, 448, 128)),
    ("query64", NAME_Q64, "64", (0, 64, 512)),
]
BENCH_NAMES = {
    "gsm8k": "GSM8K", "ifeval": "IFEval", "humaneval": "HumanEval", "aime2024": "AIME 2024", "aime2025": "AIME 2025",
    "hmmt_202502": "HMMT Feb 2025", "hmmt_202511": "HMMT Nov 2025",
}


def load_numbers():
    import json

    return json.load(open(NUMBERS))


def fig_results_matched():
    """Per-benchmark scores of SWA with a learned sink vs. Bifocal at 576 tokens of cache per layer."""
    n = load_numbers()
    benches = sorted(BENCH_NAMES, key=lambda b: n["median_tokens"][b])
    rows = [(BENCH_NAMES[b], n["scores"][f"kv448/{b}"], n["scores"][f"swa576/{b}"], fmt_tokens(n["median_tokens"][b]))
            for b in benches]
    n_short = sum(n["median_tokens"][b] < 1000 for b in benches)
    rows.append(("Math average", n["scores"]["kv448/math"], n["scores"]["swa576/math"], ""))
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    fig.subplots_adjust(left=0.25, right=0.9, top=0.76, bottom=0.12)
    header(
        fig,
        None,
        "Two linearized Qwen3-4B-Instruct students with the same memory per attention layer.",
    )
    fig.text(0.03, 1 - 0.35 / fig.get_figheight(), "* MAL: median answer length in tokens, both models.",
             fontsize=8, color=INK_2, va="top")
    ys = list(range(len(rows)))[::-1]
    head_y = len(rows) - 0.4
    for y, (name, hyb, swa, mal) in zip(ys, rows):
        # The same small vertical offset in every row, so nearly equal scores never hide each other.
        dy = 0.14
        ax.scatter(swa, y + dy, s=70, color=SWA, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.scatter(hyb, y - dy, s=70, color=HYBRID, edgecolor=SURFACE, linewidth=2, zorder=4)
        for text, x_pt, color in ((name, -42, INK), (mal, -10, INK_2)):
            ax.annotate(text, (0, y), xycoords=("axes fraction", "data"), xytext=(x_pt, 0),
                        textcoords="offset points", ha="right", va="center", fontsize=9.5, color=color)
        ax.annotate(fmt_gain(hyb - swa), (1, y), xycoords=("axes fraction", "data"), xytext=(10, 0),
                    textcoords="offset points", va="center", fontsize=9.5, color=INK)
    for text, x, x_pt, ha in (("Benchmark", 0, -42, "right"), ("MAL*", 0, -10, "right"), ("Gain", 1, 10, "left")):
        ax.annotate(text, (x, head_y), xycoords=("axes fraction", "data"), xytext=(x_pt, 0),
                    textcoords="offset points", ha=ha, va="center", fontsize=9, color=INK_2)
    ax.set_yticks([])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("Score (%)")
    ax.spines["left"].set_visible(False)
    hairline_grid(ax, "x")
    # Short answers above the first line, long math answers between the lines, the average below.
    ax.axhline(len(rows) - n_short - 0.5, color=GRID, lw=0.8)
    ax.axhline(0.5, color=GRID, lw=0.8)
    fig.legend(
        [Line2D([], [], marker="o", ls="", color=SWA, markersize=8),
         Line2D([], [], marker="o", ls="", color=HYBRID, markersize=8)],
        [f"{NAME_SWA_LEARNED}: 576 window", f"{NAME_HYBRID}: 448 window + state"],
        loc="upper left",
        bbox_to_anchor=(0.025, 0.86),
        ncol=2,
        frameon=False,
        fontsize=9,
        handletextpad=0.3,
        columnspacing=1.4,
    )
    fig.savefig(OUT / "fig_results_matched.png")
    plt.close(fig)


def fig_budget():
    """The same 576 tokens of cache per layer, spent four ways, and what each scores on math."""
    n = load_numbers()
    fig, (ax_m, ax_s) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.45, 1]})
    fig.subplots_adjust(left=0.22, right=0.97, top=0.78, bottom=0.19, wspace=0.12)
    header(
        fig,
        None,
        "Left: cache per layer in tokens (an FP32 mLSTM state = 128 tokens per KV head). Right: AIME + HMMT.",
    )
    ys = list(range(len(MATCHED)))[::-1]
    ax_m.set_xlim(0, 600)
    ax_s.set_xlim(0, 21)
    for ax in (ax_m, ax_s):
        ax.set_ylim(-0.6, len(MATCHED) - 0.4)
    gap = data_units_per_inch(ax_m)[0] * 0.02
    for y, (key, name, win_label, (sink, window, state)) in zip(ys, MATCHED):
        # 4 sink tokens would be ~2 px wide in the post, so they are drawn 12 tokens wide and the
        # window next to them 8 tokens narrower; the total stays at 576.
        shown_sink = max(sink, 12) if sink else 0
        segments = ((shown_sink, None), (window - (shown_sink - sink), win_label), (state, f"{state} state"))
        colors = (SWA, SWA, HYBRID)
        left = 0
        for (width, label), color in zip(segments, colors):
            if not width:
                continue
            last = left + width >= sink + window + state
            ax_m.add_patch(Rectangle((left, y - 0.25), width - (0 if last else gap), 0.5, color=color, lw=0))
            if label and width >= 60:
                ax_m.text(left + width / 2, y, label, ha="center", va="center", fontsize=8.5, color=SURFACE)
            left += width
        score = n["scores"][f"{key}/math"]
        bar_h(ax_s, y, 0, score, 0.5, SCORE)
        ax_s.annotate(f"{score:.1f}%", (score, y), xytext=(5, 0), textcoords="offset points", va="center", fontsize=10, color=INK)
    ax_m.set_yticks(ys)
    ax_m.set_yticklabels([m[1] for m in MATCHED], fontsize=10, color=INK)
    ax_m.tick_params(axis="y", length=0)
    ax_m.set_xticks([0, 192, 384, 576])
    ax_m.set_xlabel("Cache per layer (tokens)")
    ax_s.set_yticks([])
    ax_s.set_xticks([0, 10, 20])
    ax_s.set_xlabel("Math score (%)")
    for ax in (ax_m, ax_s):
        ax.spines["left"].set_visible(False)
        hairline_grid(ax, "x")
    y_sink = next(y for y, m in zip(ys, MATCHED) if m[3][0])
    ax_m.annotate("4 sink tokens", (2, y_sink + 0.25), xytext=(0, 3), textcoords="offset points",
                  ha="left", va="bottom", fontsize=8, color=INK_2)
    fig.savefig(OUT / "fig_budget.png")
    plt.close(fig)


def fig_long_answers():
    """Math pass@1 on problems with typically short vs. typically long answers."""
    n = load_numbers()["problem_split"]
    arms = [("swa576", NAME_SWA_LEARNED, SWA), ("kv448", NAME_HYBRID, HYBRID)]
    short, long_ = n["kv448-swa576/short"], n["kv448-swa576/long"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.45))
    fig.subplots_adjust(left=0.2, right=0.95, top=0.68, bottom=0.22, wspace=0.45)
    header(
        fig,
        None,
        "Math pass@1. Problems grouped by their median answer length over both models (AIME and HMMT).",
    )
    for ax, (split, title, xmax, ticks) in zip(axes, (
        (short, f"{short['n']} problems with answers under 4K tokens", 75, [0, 20, 40, 60]),
        (long_, f"{long_['n']} problems with answers over 4K tokens", 5.6, [0, 1, 2, 3, 4, 5]),
    )):
        ax.set_xlim(0, xmax)
        ax.set_ylim(-0.6, len(arms) - 0.4)
        ys = list(range(len(arms)))[::-1]
        for y, (key, _, color) in zip(ys, arms):
            bar_h(ax, y, 0, split[key], 0.5, color)
            ax.annotate(f"{split[key]:.1f}%", (split[key], y), xytext=(5, 0), textcoords="offset points",
                        va="center", fontsize=10, color=INK)
        ax.set_yticks(ys)
        ax.set_yticklabels([a[1] for a in arms] if split is short else [], fontsize=10, color=INK)
        ax.tick_params(axis="y", length=0)
        ax.set_xticks(ticks)
        ax.set_title(title, loc="left", fontsize=9.5, color=INK, pad=8)
        ax.set_xlabel("Pass@1 (%)")
        ax.spines["left"].set_visible(False)
        hairline_grid(ax, "x")
    fig.savefig(OUT / "fig_long_answers.png")
    plt.close(fig)


def fig_window_tape():
    """The SWA student's window against one median-length math answer."""
    n = load_numbers()
    window = 576
    rollout = round(n["math"]["median_tokens_main"], -2)
    prompt = 150  # illustrative length of a competition-math question (75 to 184 tokens here)
    total = prompt + rollout
    fig, ax = plt.subplots(figsize=(7.2, 1.9))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.82, bottom=0.05)
    header(
        fig,
        None,
        f"A sliding window sees {window / rollout:.0%} of a typical math answer.",
    )
    ax.set_xlim(-total * 0.01, total * 1.01)
    ax.set_ylim(-1.2, 1.25)
    ax.axis("off")
    gap = data_units_per_inch(ax)[0] * 0.02
    bar_h(ax, 0.3, 0, prompt, 0.55, INK_2, round_end=False)
    bar_h(ax, 0.3, prompt + gap, total - prompt - window - 2 * gap, 0.55, REST, round_end=False)
    bar_h(ax, 0.3, total - window, window, 0.55, SWA, round_end=False)
    ax.text(total / 2, 0.3, "The model's own reasoning", ha="center", va="center", fontsize=9, color=INK)
    question = ax.text(0, 0.72, "Question", ha="left", va="bottom", fontsize=9, color=INK)
    fig.canvas.draw()
    x_note = ax.transData.inverted().transform((question.get_window_extent().x1, 0))[0]
    ax.annotate("· gone from the window after a few hundred tokens of reasoning", (x_note, 0.72), xytext=(4, 0),
                textcoords="offset points", ha="left", va="bottom", fontsize=9, color=INK_2)
    ax.text(total, 0.72, "Window", ha="right", va="bottom", fontsize=9, color=INK)
    ax.plot([0, total], [-0.4, -0.4], color=HYBRID, lw=2, solid_capstyle="round")
    ax.text(0, -0.62, "Linear mixer state: a compressed summary of every token so far, question included",
            ha="left", va="top", fontsize=9, color=INK_2)
    fig.savefig(OUT / "fig_window_tape.png")
    plt.close(fig)


def fig_state_heads():
    """One key-value head and its 4 query heads: one shared state vs. one state per query head."""
    n = load_numbers()["scores"]
    c = 0.7  # side of one square
    d = 3 * c  # side of one state block
    fig, ax = plt.subplots(figsize=(7.2, 2.95))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.80, bottom=0.02)
    header(
        fig,
        None,
        "One key-value head of Qwen3-4B, shared by 4 query heads. Both layouts hold 576 tokens' worth of cache.",
    )
    ax.set_xlim(-0.3, 32.6)
    ax.set_ylim(-2.3, 8.0)
    ax.set_aspect("equal")
    ax.axis("off")
    y_mid, y_q = 2.0, 6.0

    def window(x0, n_cells):
        for i in range(n_cells):
            ax.add_patch(Rectangle((x0 + i * c + 0.05, y_mid - c / 2 + 0.05), c - 0.1, c - 0.1, color=SWA, lw=0))
        return x0 + n_cells * c / 2  # center

    def state(x0):
        for i in range(3):
            for j in range(3):
                ax.add_patch(Rectangle((x0 + i * c + 0.05, y_mid - d / 2 + j * c + 0.05), c - 0.1, c - 0.1, color=HYBRID, lw=0))
        return x0 + d / 2  # center

    def query(x, label):
        ax.add_patch(FancyBboxPatch((x - 0.6, y_q - 0.4), 1.2, 0.8, boxstyle="round,pad=0,rounding_size=0.15", color=INK_2, lw=0))
        ax.text(x, y_q, label, ha="center", va="center", fontsize=8.5, color=SURFACE)

    def read(x_from, x_to, y_to, color):
        ax.plot([x_from, x_to], [y_q - 0.4, y_to], color=color, lw=1.2, solid_capstyle="round", zorder=0)

    panels = [
        (0.0, "One state per key-value head", 7, 1, f"{NAME_HYBRID}: 448-token window + 1 state",
         f"Math score: {n['kv448/math']:.1f}%"),
        (17.3, "One state per query head", 3, 4, f"{NAME_Q64}: 64-token window + 4 states",
         f"Math score: {n['query64/math']:.1f}%"),
    ]
    for x0, title, n_win, n_states, caption, result in panels:
        ax.text(x0 + 0.3, 7.4, title, ha="left", va="center", fontsize=9.5, color=INK)
        win_c = window(x0 + 0.3, n_win)
        s_x = x0 + 0.3 + n_win * c + 0.7
        centers = [state(s_x + k * (d + 0.6)) for k in range(n_states)]
        qx = centers if n_states == 4 else [x0 + 1.2 + k * 1.9 for k in range(4)]
        for k, x in enumerate(qx):
            read(x, win_c, y_mid + c / 2, BASELINE)
            read(x, centers[k if n_states == 4 else 0], y_mid + d / 2, HYBRID)
            query(x, f"q{k + 1}")
        ax.text(win_c, y_mid - d / 2 - 0.15, "window", ha="center", va="top", fontsize=8, color=INK_2)
        ax.text((centers[0] + centers[-1]) / 2, y_mid - d / 2 - 0.15, "state" if n_states == 1 else "states",
                ha="center", va="top", fontsize=8, color=INK_2)
        ax.text(x0 + 0.3, -0.95, caption, ha="left", va="center", fontsize=9, color=INK)
        ax.text(x0 + 0.3, -1.75, result, ha="left", va="center", fontsize=9, color=INK_2)
    fig.legend(
        [Line2D([], [], color=BASELINE, lw=1.5), Line2D([], [], color=HYBRID, lw=1.5)],
        ["reads the window", "reads a state"],
        loc="upper left",
        bbox_to_anchor=(0.025, 0.89),
        ncol=2,
        frameon=False,
        fontsize=8.5,
        handlelength=1.6,
        columnspacing=1.2,
    )
    fig.savefig(OUT / "fig_state_heads.png")
    plt.close(fig)


def results_table(fname, title, subtitle, columns, rows, note=None):
    """A plain table as an image, in the figures' style. The best value in each column is bold."""
    row_h, top, col_w = 0.34, 0.85, 0.9
    head_h = 0.5 if any("\n" in c for c in columns) else 0.36
    bottom = 0.45 if note else 0.15
    h = top + head_h + len(rows) * row_h + bottom
    fig = plt.figure(figsize=(7.2, h))
    header(fig, title, subtitle)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 7.2)
    ax.set_ylim(0, h)
    ax.axis("off")
    x0, first = 0.03 * 7.2, 2.3  # label column, start of the numbers
    xs = [first + (k + 1) * col_w for k in range(len(columns))]  # right edges; numbers are right-aligned
    x1 = xs[-1]
    # Matplotlib reads only the regular face of Helvetica Neue's .ttc; Arial Bold has matching digits.
    bold = {"family": "Arial", "weight": "bold"}
    y_head = h - top - head_h / 2
    for x, label in zip(xs, columns):
        ax.text(x, y_head, label, ha="right", va="center", fontsize=9, color=INK_2, linespacing=1.3)
    rule_y = h - top - head_h
    ax.plot([x0, x1], [rule_y, rule_y], color=BASELINE, lw=0.8)
    best = [max(v for _, vals in rows if (v := vals[k]) is not None) for k in range(len(columns))]
    for r, (name, vals) in enumerate(rows):
        y = rule_y - (r + 0.5) * row_h
        ax.text(x0, y, name, ha="left", va="center", fontsize=9.5, color=INK)
        for x, v, b in zip(xs, vals, best):
            is_best = v is not None and round(v, 1) == round(b, 1)
            ax.text(x, y, "—" if v is None else f"{v:.1f}", ha="right", va="center", fontsize=9.5,
                    color=INK_2 if v is None else INK, **(bold if is_best else {}))
    end_y = rule_y - len(rows) * row_h
    ax.plot([x0, x1], [end_y, end_y], color=BASELINE, lw=0.8)
    if note:
        ax.text(x0, end_y - 0.12, note, ha="left", va="top", fontsize=8.5, color=INK_2)
    fig.savefig(OUT / fname)
    plt.close(fig)


def fig_tables():
    """All scores of the four matched-memory students: short-answer benchmarks and math."""
    s = load_numbers()["scores"]
    short, math = ["gsm8k", "ifeval", "humaneval"], ["aime2024", "aime2025", "hmmt_202502", "hmmt_202511"]
    results_table(
        "table_short.png",
        "Short-answer benchmarks",
        "Scores in %, greedy decoding. GSM8K: 5-shot, exact match. IFEval: strict prompt accuracy. HumanEval: pass@1.",
        ["GSM8K", "IFEval", "HumanEval"],
        [(name, [None if key == "swa572" else s[f"{key}/{b}"] for b in short]) for key, name, *_ in MATCHED],
        note="SWA + sink tokens: not reported, see \"What we left out\" below.",
    )
    results_table(
        "table_math.png",
        "Math benchmarks",
        "Pass@1 (%), 30 problems each, 16 samples per problem, up to 31,744 new tokens.",
        ["AIME\n2024", "AIME\n2025", "HMMT\nFeb 2025", "HMMT\nNov 2025", "Math\naverage"],
        [(name, [s[f"{key}/{b}"] for b in math] + [s[f"{key}/math"]]) for key, name, *_ in MATCHED],
    )


if __name__ == "__main__":
    fig_memory()
    fig_state_heads()
    fig_decode_throughput()
    fig_results_matched()
    fig_budget()
    fig_long_answers()
    fig_window_tape()
    fig_tables()
