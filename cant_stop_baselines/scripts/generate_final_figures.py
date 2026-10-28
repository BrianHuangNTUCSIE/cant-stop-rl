"""Generate dependency-free SVG figures for the final report evidence."""
from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "final_results"


def esc(value: str) -> str:
    return (value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def chart(title: str, y_label: str, series: list[tuple[str, list[float], str]],
          categories: list[str], maximum: float, filename: str,
          intervals: list[list[tuple[float, float]]] | None = None) -> None:
    width, height = 920, 540
    left, right, top, bottom = 90, 35, 70, 95
    plot_w, plot_h = width - left - right, height - top - bottom
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<style>text{font-family:Arial,sans-serif;fill:#172033} .grid{stroke:#d9dee8} .axis{stroke:#344054} .label{font-size:14px} .small{font-size:12px} .title{font-size:20px;font-weight:bold}</style>',
             f'<text x="{width/2}" y="32" text-anchor="middle" class="title">{esc(title)}</text>']
    for tick in range(0, int(maximum) + 1, 10):
        y = top + plot_h - (tick / maximum) * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" class="small">{tick}</text>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" class="axis"/>')
    parts.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{width-right}" y2="{top+plot_h}" class="axis"/>')
    group_w = plot_w / len(categories)
    bar_w = min(48, group_w / (len(series) + 1))
    colors = [s[2] for s in series]
    for i, category in enumerate(categories):
        center = left + group_w * (i + 0.5)
        parts.append(f'<text x="{center:.1f}" y="{top+plot_h+27}" text-anchor="middle" class="label">{esc(category)}</text>')
        for j, (_, values, color) in enumerate(series):
            value = values[i]
            x = center + (j - (len(series)-1)/2) * bar_w - bar_w * .42
            y = top + plot_h - (value / maximum) * plot_h
            h = top + plot_h - y
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w*.84:.1f}" height="{h:.1f}" fill="{color}"/>')
            parts.append(f'<text x="{x+bar_w*.42:.1f}" y="{y-7:.1f}" text-anchor="middle" class="small">{value:.1f}</text>')
            if intervals:
                lo, hi = intervals[j][i]
                ylo = top + plot_h - (lo / maximum) * plot_h
                yhi = top + plot_h - (hi / maximum) * plot_h
                cx = x + bar_w*.42
                parts.append(f'<line x1="{cx:.1f}" y1="{yhi:.1f}" x2="{cx:.1f}" y2="{ylo:.1f}" stroke="#172033" stroke-width="2"/>')
                parts.append(f'<line x1="{cx-5:.1f}" y1="{yhi:.1f}" x2="{cx+5:.1f}" y2="{yhi:.1f}" stroke="#172033"/>')
                parts.append(f'<line x1="{cx-5:.1f}" y1="{ylo:.1f}" x2="{cx+5:.1f}" y2="{ylo:.1f}" stroke="#172033"/>')
    for j, (name, _, color) in enumerate(series):
        x = left + j * 190
        parts.append(f'<rect x="{x}" y="{height-38}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text x="{x+20}" y="{height-26}" class="small">{esc(name)}</text>')
    parts.append(f'<text x="18" y="{top+plot_h/2}" transform="rotate(-90 18 {top+plot_h/2})" text-anchor="middle" class="label">{esc(y_label)}</text>')
    parts.append('</svg>')
    (OUT / filename).write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    chart("PPO advantage over retained baselines", "Percentage points", [
        ("PPO minus DDQN", [3.2, 2.7], "#2563eb"),
        ("PPO minus Ridge", [4.5, 4.3], "#16a34a"),
    ], ["Rule28-smart", "MC"], 8,
          "primary_model_advantage.svg",
          [[(1.2, 5.3), (0.6, 5.0)], [(2.2, 6.7), (1.9, 6.6)]])
    chart("Learned pairing diagnostic", "Win rate (%)", [
        ("PPO-451 + GA", [64.6, 55.4], "#2563eb"),
        ("Pairing-only PPO", [50.4, 44.5], "#dc2626"),
    ], ["Rule28-smart", "MC"], 70, "learned_pairing_negative.svg")
    chart("vmerckle budget sweep vs Rule28-smart", "Win rate (%)", [
        ("Flat Monte Carlo", [0.3, 3.3, 10.3], "#7c3aed"),
        ("Root-only UCB1", [0.8, 4.3, 11.5], "#ea580c"),
    ], ["16 rollouts", "64 rollouts", "256 rollouts"], 15, "vmerckle_budget_sweep.svg")


if __name__ == "__main__":
    main()
