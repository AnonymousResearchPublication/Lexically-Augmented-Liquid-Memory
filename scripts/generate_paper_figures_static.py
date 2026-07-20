from __future__ import annotations

import argparse
import html
import math
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal envs.
    plt = None


HORIZONS = [100, 500, 1000, 5000]
HORIZON_LABELS = ["100", "500", "1K", "5K"]
SAMPLE_SIZES = [300, 225, 150, 150]

AGENT_ORDER = [
    "LALM",
    "Lexical-only",
    "RAG",
    "Pure Liquid",
    "Window",
    "Vanilla",
]

AGENT_STYLES = {
    "LALM": {"color": "#1f77b4", "marker": "o", "linewidth": 2.5},
    "Lexical-only": {"color": "#ff7f0e", "marker": "s", "linewidth": 2.1},
    "RAG": {"color": "#2ca02c", "marker": "^", "linewidth": 2.0},
    "Pure Liquid": {"color": "#9467bd", "marker": "D", "linewidth": 1.4, "alpha": 0.78},
    "Window": {"color": "#8c564b", "marker": "v", "linewidth": 1.4, "alpha": 0.78},
    "Vanilla": {"color": "#7f7f7f", "marker": "x", "linewidth": 1.2, "alpha": 0.78},
}

SYNTHETIC = {
    7: {
        "Vanilla": ([0.00, 0.00, 0.00, 0.00], [0.00, 0.00, 0.00, 0.00]),
        "Window": ([18.33, 0.00, 0.00, 0.00], [4.04, 0.00, 0.00, 0.00]),
        "RAG": ([66.33, 63.11, 58.67, 58.67], [1.15, 5.39, 6.11, 5.03]),
        "Pure Liquid": ([15.00, 14.22, 16.00, 16.00], [2.65, 0.77, 0.00, 0.00]),
        "Lexical-only": ([72.67, 66.67, 67.33, 66.00], [1.15, 3.53, 4.16, 2.00]),
        "LALM": ([73.33, 69.33, 72.00, 72.00], [2.08, 5.33, 2.00, 0.00]),
    },
    13: {
        "Vanilla": ([0.00, 0.00, 0.00, 0.00], [0.00, 0.00, 0.00, 0.00]),
        "Window": ([18.33, 0.00, 0.00, 0.00], [4.04, 0.00, 0.00, 0.00]),
        "RAG": ([66.33, 63.11, 58.67, 58.67], [1.15, 5.39, 6.11, 5.03]),
        "Pure Liquid": ([11.33, 10.22, 3.33, 0.00], [0.58, 0.77, 2.31, 0.00]),
        "Lexical-only": ([72.67, 66.67, 67.33, 66.00], [1.15, 3.53, 4.16, 2.00]),
        "LALM": ([80.67, 74.22, 74.67, 76.67], [1.15, 2.78, 3.06, 3.06]),
    },
}

MEMORY_SCALING_BYTES = {
    "RAG": [160_396, 802_352, 1_604_804, 8_028_400],
    "LALM": [1_318_912, 1_318_912, 1_318_912, 1_318_912],
}


def configure_matplotlib() -> None:
    if plt is None:
        return
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "figure.dpi": 120,
    })


def plot_synthetic_crossover(output: Path) -> None:
    if plt is None:
        plot_synthetic_crossover_svg(output.with_suffix(".svg"))
        return

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharey=True)

    for ax, seed in zip(axes, [7, 13]):
        for label in AGENT_ORDER:
            means, stds = SYNTHETIC[seed][label]
            style = AGENT_STYLES[label]
            ax.errorbar(
                HORIZONS,
                means,
                yerr=stds,
                label=label,
                capsize=3,
                markersize=5,
                **style,
            )
        for horizon, n in zip(HORIZONS, SAMPLE_SIZES):
            ax.annotate(
                f"n={n}",
                xy=(horizon, 3.0),
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#444444",
            )
        ax.set_title(f"Training seed {seed}")
        ax.set_xlabel("Conversation horizon (turns)")
        ax.set_xscale("log")
        ax.set_xticks(HORIZONS)
        ax.set_xticklabels(HORIZON_LABELS)
        ax.set_ylim(-3, 86)
        ax.grid(True, axis="y", alpha=0.25)

    axes[0].set_ylabel("Exact-value accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.5, -0.03),
    )
    fig.suptitle("Synthetic exact-value accuracy across horizons", y=0.99, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_memory_scaling(output: Path) -> None:
    if plt is None:
        plot_memory_scaling_svg(output.with_suffix(".svg"))
        return

    fig, ax = plt.subplots(figsize=(6.4, 4.1))
    ax.plot(
        HORIZONS,
        MEMORY_SCALING_BYTES["RAG"],
        marker="o",
        linewidth=2.3,
        color="#2ca02c",
        label="RAG",
    )
    ax.plot(
        HORIZONS,
        MEMORY_SCALING_BYTES["LALM"],
        marker="s",
        linewidth=2.3,
        color="#1f77b4",
        label="LALM",
    )
    ax.set_title("Logical user-memory payload scaling")
    ax.set_xlabel("Conversation horizon (turns)")
    ax.set_ylabel("Logical user-memory payload (bytes)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(HORIZONS)
    ax.set_xticklabels(HORIZON_LABELS)
    ax.grid(True, which="both", axis="y", alpha=0.25)
    ax.legend(frameon=False)

    ax.annotate(
        "LALM fixed at 1,318,912 bytes",
        xy=(1000, 1_318_912),
        xytext=(550, 2_600_000),
        arrowprops={"arrowstyle": "->", "color": "#1f77b4", "lw": 1.0},
        color="#1f77b4",
        fontsize=8.5,
    )
    ax.annotate(
        "RAG grows with turns",
        xy=(5000, 8_028_400),
        xytext=(1250, 7_000_000),
        arrowprops={"arrowstyle": "->", "color": "#2ca02c", "lw": 1.0},
        color="#2ca02c",
        fontsize=8.5,
    )

    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def svg_text(x: float, y: float, text: str, size: int = 14, weight: str = "normal",
             anchor: str = "middle", color: str = "#111111") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Times New Roman, Times, serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
        f'fill="{color}">{html.escape(text)}</text>'
    )


def svg_line(x1: float, y1: float, x2: float, y2: float, color: str = "#222222",
             width: float = 1.2, dash: str | None = None) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{width}"{dash_attr}/>'
    )


def svg_polyline(points: list[tuple[float, float]], color: str, width: float = 2.4,
                 dash: str | None = None) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return (
        f'<polyline points="{point_text}" fill="none" stroke="{color}" '
        f'stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"{dash_attr}/>'
    )


def svg_marker(x: float, y: float, color: str, marker: str) -> str:
    if marker == "s":
        return f'<rect x="{x - 4:.1f}" y="{y - 4:.1f}" width="8" height="8" fill="{color}"/>'
    if marker == "^":
        return (
            f'<polygon points="{x:.1f},{y - 5:.1f} {x - 5:.1f},{y + 5:.1f} '
            f'{x + 5:.1f},{y + 5:.1f}" fill="{color}"/>'
        )
    if marker == "D":
        return (
            f'<polygon points="{x:.1f},{y - 5:.1f} {x - 5:.1f},{y:.1f} '
            f'{x:.1f},{y + 5:.1f} {x + 5:.1f},{y:.1f}" fill="{color}"/>'
        )
    if marker == "v":
        return (
            f'<polygon points="{x:.1f},{y + 5:.1f} {x - 5:.1f},{y - 5:.1f} '
            f'{x + 5:.1f},{y - 5:.1f}" fill="{color}"/>'
        )
    if marker == "x":
        return svg_line(x - 4, y - 4, x + 4, y + 4, color, 1.8) + svg_line(x - 4, y + 4, x + 4, y - 4, color, 1.8)
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}"/>'


def log_x_mapper(left: float, width: float):
    log_min = math.log10(min(HORIZONS))
    log_max = math.log10(max(HORIZONS))

    def mapper(value: float) -> float:
        return left + (math.log10(value) - log_min) / (log_max - log_min) * width

    return mapper


def linear_y_mapper(top: float, height: float, ymin: float, ymax: float):
    def mapper(value: float) -> float:
        return top + (ymax - value) / (ymax - ymin) * height

    return mapper


def log_y_mapper(top: float, height: float, ymin: float, ymax: float):
    log_min = math.log10(ymin)
    log_max = math.log10(ymax)

    def mapper(value: float) -> float:
        return top + (log_max - math.log10(value)) / (log_max - log_min) * height

    return mapper


def plot_synthetic_crossover_svg(output: Path) -> None:
    width, height = 1260, 580
    panel_top, panel_height = 88, 340
    panel_width = 455
    lefts = [82, 665]
    y_map = linear_y_mapper(panel_top, panel_height, -3, 86)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 34, "Synthetic exact-value accuracy across horizons", 24, "bold"),
    ]

    for left, seed in zip(lefts, [7, 13]):
        x_map = log_x_mapper(left, panel_width)
        right = left + panel_width
        bottom = panel_top + panel_height
        parts.extend([
            svg_text(left + panel_width / 2, 66, f"Training seed {seed}", 18, "bold"),
            svg_line(left, panel_top, left, bottom),
            svg_line(left, bottom, right, bottom),
        ])
        for tick in [0, 20, 40, 60, 80]:
            y = y_map(tick)
            parts.append(svg_line(left, y, right, y, "#dddddd", 0.8))
            parts.append(svg_text(left - 12, y + 4, str(tick), 11, anchor="end", color="#333333"))
        for horizon, label, n in zip(HORIZONS, HORIZON_LABELS, SAMPLE_SIZES):
            x = x_map(horizon)
            parts.append(svg_line(x, bottom, x, bottom + 5, "#222222", 1.0))
            parts.append(svg_text(x, bottom + 22, label, 12))
            parts.append(svg_text(x, y_map(3.0), f"n={n}", 10, color="#555555"))

        for label in AGENT_ORDER:
            means, stds = SYNTHETIC[seed][label]
            style = AGENT_STYLES[label]
            color = style["color"]
            points = [(x_map(h), y_map(m)) for h, m in zip(HORIZONS, means)]
            parts.append(svg_polyline(points, color, style.get("linewidth", 2.0)))
            for (x, y), mean, std in zip(points, means, stds):
                if std:
                    y0, y1 = y_map(mean - std), y_map(mean + std)
                    parts.append(svg_line(x, y0, x, y1, color, 1.0))
                    parts.append(svg_line(x - 4, y0, x + 4, y0, color, 1.0))
                    parts.append(svg_line(x - 4, y1, x + 4, y1, color, 1.0))
                parts.append(svg_marker(x, y, color, style["marker"]))

        parts.append(svg_text(left + panel_width / 2, height - 92, "Conversation horizon (turns)", 13))

    parts.append(svg_text(23, panel_top + panel_height / 2, "Exact-value accuracy (%)", 13, anchor="middle"))
    legend_x, legend_y = 168, 505
    for idx, label in enumerate(AGENT_ORDER):
        col = idx % 3
        row = idx // 3
        x = legend_x + col * 330
        y = legend_y + row * 28
        color = AGENT_STYLES[label]["color"]
        parts.append(svg_line(x, y - 4, x + 30, y - 4, color, 2.2))
        parts.append(svg_marker(x + 15, y - 4, color, AGENT_STYLES[label]["marker"]))
        parts.append(svg_text(x + 42, y, label, 13, anchor="start"))

    parts.append("</svg>")
    output.write_text("\n".join(parts), encoding="utf-8")


def plot_memory_scaling_svg(output: Path) -> None:
    width, height = 760, 500
    left, top, plot_width, plot_height = 95, 65, 560, 315
    bottom = top + plot_height
    right = left + plot_width
    x_map = log_x_mapper(left, plot_width)
    y_map = log_y_mapper(top, plot_height, 100_000, 10_000_000)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 34, "Logical user-memory payload scaling", 24, "bold"),
        svg_line(left, top, left, bottom),
        svg_line(left, bottom, right, bottom),
    ]
    for tick in [100_000, 300_000, 1_000_000, 3_000_000, 10_000_000]:
        y = y_map(tick)
        parts.append(svg_line(left, y, right, y, "#dddddd", 0.8))
        label = f"{tick // 1_000_000}M" if tick >= 1_000_000 else f"{tick // 1000}K"
        parts.append(svg_text(left - 12, y + 4, label, 11, anchor="end", color="#333333"))
    for horizon, label in zip(HORIZONS, HORIZON_LABELS):
        x = x_map(horizon)
        parts.append(svg_line(x, bottom, x, bottom + 5, "#222222", 1.0))
        parts.append(svg_text(x, bottom + 23, label, 12))

    series = [
        ("LALM", "#1f77b4", "s"),
        ("RAG", "#2ca02c", "o"),
    ]
    for label, color, marker in series:
        points = [(x_map(h), y_map(v)) for h, v in zip(HORIZONS, MEMORY_SCALING_BYTES[label])]
        parts.append(svg_polyline(points, color, 2.8))
        for x, y in points:
            parts.append(svg_marker(x, y, color, marker))

    parts.extend([
        svg_text(left + plot_width / 2, height - 54, "Conversation horizon (turns)", 14),
        svg_text(25, top + plot_height / 2, "Logical user-memory payload (bytes)", 13),
        svg_line(430, 130, 520, 130, "#1f77b4", 2.5),
        svg_marker(475, 130, "#1f77b4", "s"),
        svg_text(535, 135, "LALM fixed: 1,318,912 bytes", 13, anchor="start", color="#1f77b4"),
        svg_line(430, 158, 520, 158, "#2ca02c", 2.5),
        svg_marker(475, 158, "#2ca02c", "o"),
        svg_text(535, 163, "RAG grows with turns", 13, anchor="start", color="#2ca02c"),
    ])
    parts.append("</svg>")
    output.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the paper's hardcoded synthetic seed 7/13 figure and "
            "memory-scaling figure. This does not read result directories."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=Path("Figures"))
    args = parser.parse_args()

    configure_matplotlib()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    synthetic = args.output_dir / "synthetic_crossover.png"
    memory = args.output_dir / "memory_scaling.png"
    plot_synthetic_crossover(synthetic)
    plot_memory_scaling(memory)
    print(f"Wrote {synthetic}")
    print(f"Wrote {memory}")


if __name__ == "__main__":
    main()
