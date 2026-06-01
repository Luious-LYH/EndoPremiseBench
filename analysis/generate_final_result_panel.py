"""Generate the primary Figure 2 result panel.

The figure reads the paper-facing result bundle and includes only complete
6,000-item primary evaluations. Closed reference rows are included when the
coverage table marks their formal paper-facing assets as complete.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "results" / "a_group_supplement_20260524" / "paper_assets_20260525"
FIGURES = ROOT / "figures"

DISPLAY = {
    "gpt5_5": "GPT-5.5",
    "claude_opus_4_7": "Claude-Opus-4.7",
    "grok_4_20_multi_agent_xhigh": "Grok-4.20",
    "simula_qwen25_kvasir": "Qwen2.5-VL-7B-Kvasir-ft",
    "simula_medgemma_kvasir": "MedGemma-4B-Kvasir-ft",
    "qwen3_vl_8b": "Qwen3-VL-8B",
    "qwen3vl4b": "Qwen3-VL-4B",
    "qwen25_vl_7b": "Qwen2.5-VL-7B",
    "qwen25_vl_3b": "Qwen2.5-VL-3B",
    "minicpmv4": "MiniCPM-V-4",
    "internvl25_8b": "InternVL2.5-8B",
    "internvl25_4b": "InternVL2.5-4B",
    "lingshu_7b": "Lingshu-7B",
    "medgemma_4b": "MedGemma-4B",
    "llava_med_v15_mistral_7b": "LLaVA-Med",
}

FAMILY = {
    "gpt5_5": "closed",
    "claude_opus_4_7": "closed",
    "grok_4_20_multi_agent_xhigh": "closed",
    "simula_qwen25_kvasir": "endo",
    "simula_medgemma_kvasir": "endo",
    "lingshu_7b": "medical",
    "medgemma_4b": "medical",
    "llava_med_v15_mistral_7b": "medical",
}

STYLE = {
    "open": {"label": "general open-weight", "color": "#386FA4", "marker": "o"},
    "medical": {"label": "medical open-weight", "color": "#C84E45", "marker": "^"},
    "endo": {"label": "Kvasir-VQA-x1 LoRA", "color": "#2F8F68", "marker": "D"},
    "closed": {"label": "closed reference", "color": "#7A60A8", "marker": "s"},
    "policy": {"label": "reference policy", "color": "#8D8D8D", "marker": "X"},
}

REFERENCE_POLICIES = [
    {"name": "Always N/A", "acc_tp": 0.0, "acc_fp": 100.0},
    {"name": "Never refuse", "acc_tp": 33.2, "acc_fp": 0.0},
    {"name": "Label oracle", "acc_tp": 33.9, "acc_fp": 100.0},
]

ANCHOR_LABEL_POS = {
    "simula_qwen25_kvasir": (37.0, 85.0),
    "claude_opus_4_7": (7.6, 70.3),
    "grok_4_20_multi_agent_xhigh": (47.8, 62.6),
    "gpt5_5": (58.2, 35.3),
    "qwen3_vl_8b": (17.5, 53.5),
    "medgemma_4b": (41.0, 8.4),
}

ANCHOR_LABEL_TEXT = {
    "simula_qwen25_kvasir": "Qwen2.5-VL-7B\nKvasir-ft",
}

# Match the one-decimal presentation in paper/tables/main_results.tex.
SR_DISPLAY = {
    "gpt5_5": "56.2",
    "claude_opus_4_7": "32.6",
    "grok_4_20_multi_agent_xhigh": "42.5",
    "simula_qwen25_kvasir": "25.1",
    "simula_medgemma_kvasir": "91.5",
    "qwen3_vl_8b": "51.7",
    "qwen3vl4b": "53.9",
    "qwen25_vl_7b": "58.2",
    "qwen25_vl_3b": "61.0",
    "minicpmv4": "56.6",
    "internvl25_8b": "55.9",
    "internvl25_4b": "84.4",
    "lingshu_7b": "60.6",
    "medgemma_4b": "94.8",
    "llava_med_v15_mistral_7b": "83.4",
}


def load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with (ASSETS / "main_model_summary.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(float(row["n"])) != 6000:
                continue
            model = row["model"]
            rows.append(
                {
                    "model": model,
                    "name": DISPLAY.get(model, model),
                    "family": FAMILY.get(model, "open"),
                    "acc_tp": float(row["Acc_TP"]) * 100.0,
                    "acc_fp": float(row["Acc_FP"]) * 100.0,
                    "sr": float(row["SR"]) * 100.0,
                }
            )
    return sorted(rows, key=lambda item: float(item["acc_fp"]), reverse=True)


def main() -> None:
    rows = load_rows()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.3,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.2,
        }
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.05, 3.55),
        gridspec_kw={"width_ratios": [1.07, 0.95], "wspace": 0.34},
    )

    ax = axes[0]
    ax.set_facecolor("#FBFBFC")
    ax.add_patch(Rectangle((50, 70), 50, 30, facecolor="#E7F3ED", edgecolor="none", zorder=0))
    ax.text(
        80,
        85,
        "endoscopic target\nutility + restraint",
        ha="center",
        va="center",
        fontsize=7.2,
        color="#2A6F55",
    )
    ax.plot([0, 100], [0, 100], color="#D3D3D3", linewidth=0.75, linestyle="--", zorder=1)

    for level in [25, 40, 55]:
        xs = []
        ys = []
        for x in [i / 2.0 for i in range(1, 201)]:
            if 2 * x <= level:
                continue
            y = (level * x) / (2 * x - level)
            if 0 <= y <= 100:
                xs.append(x)
                ys.append(y)
        ax.plot(xs, ys, color="#E6E6E6", linewidth=0.45, zorder=1)

    for ref in REFERENCE_POLICIES:
        ax.scatter(
            ref["acc_tp"],
            ref["acc_fp"],
            s=44,
            marker=STYLE["policy"]["marker"],
            facecolor=STYLE["policy"]["color"],
            edgecolor="white",
            linewidth=0.62,
            alpha=0.72,
            zorder=2,
        )
    ax.annotate(
        "Label oracle",
        (33.9, 100.0),
        xytext=(43, 96),
        textcoords="data",
        fontsize=6.8,
        arrowprops={"arrowstyle": "-", "color": "#7A7A7A", "linewidth": 0.42},
    )

    for row in rows:
        family = str(row["family"])
        is_anchor = row["model"] in ANCHOR_LABEL_POS
        ax.scatter(
            row["acc_tp"],
            row["acc_fp"],
            s=80 if is_anchor else 48,
            marker=STYLE[family]["marker"],
            facecolor=STYLE[family]["color"],
            edgecolor="white",
            linewidth=0.82,
            alpha=0.96 if is_anchor else 0.56,
            zorder=4 if is_anchor else 3,
        )
        if is_anchor:
            label_x, label_y = ANCHOR_LABEL_POS[str(row["model"])]
            ax.annotate(
                ANCHOR_LABEL_TEXT.get(str(row["model"]), str(row["name"])),
                (row["acc_tp"], row["acc_fp"]),
                xytext=(label_x, label_y),
                textcoords="data",
                fontsize=6.8,
                bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "#C9C9C9", "lw": 0.33, "alpha": 0.93},
                arrowprops={"arrowstyle": "-", "color": "#707070", "linewidth": 0.42},
            )

    ax.set_xlim(-3, 103)
    ax.set_ylim(-3, 105)
    ax.set_xlabel("Supported-question accuracy (%)", fontsize=8.8)
    ax.set_ylabel("False-premise rejection (%)", fontsize=8.8)
    ax.set_title("Utility-Restraint Profile", fontsize=9.9, pad=6)
    ax.tick_params(axis="both", labelsize=7.7)
    ax.grid(True, color="#E3E3E3", linewidth=0.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=entry["marker"],
            color="none",
            markerfacecolor=entry["color"],
            markeredgecolor="white",
            markersize=6.4,
            label=entry["label"],
        )
        for entry in STYLE.values()
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=5.75,
        frameon=True,
        framealpha=0.95,
        borderpad=0.3,
        labelspacing=0.25,
        handletextpad=0.38,
    )

    ax = axes[1]
    bar_rows = sorted(rows, key=lambda item: float(item["sr"]), reverse=True)
    names = [str(row["name"]) for row in bar_rows]
    values = [float(row["sr"]) for row in bar_rows]
    colors = [STYLE[str(row["family"])]["color"] for row in bar_rows]
    bars = ax.barh(range(len(names)), values, color=colors, edgecolor="white", linewidth=0.55, height=0.6)
    ax.invert_yaxis()
    ax.set_xlim(0, 104)
    ax.set_yticks(range(len(names)), names)
    ax.tick_params(axis="y", labelsize=6.6)
    ax.set_xlabel("Unsupported-attribute exposure (%)")
    ax.set_title("Unsupported-Attribute Exposure", pad=6)
    ax.grid(axis="x", color="#E3E3E3", linewidth=0.5)
    ax.axvline(50, linestyle="--", color="#B6B6B6", linewidth=0.72)
    for bar, value, row in zip(bars, values, bar_rows):
        ax.text(
            value + 1.5,
            bar.get_y() + bar.get_height() / 2,
            SR_DISPLAY.get(str(row["model"]), f"{value:.1f}"),
            va="center",
            ha="left",
            fontsize=6.5,
            color="#222222",
        )
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.subplots_adjust(left=0.062, right=0.992, top=0.90, bottom=0.14, wspace=0.34)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig_result_panel.pdf", bbox_inches="tight", pad_inches=0.015)
    fig.savefig(FIGURES / "fig_result_panel.png", dpi=320, bbox_inches="tight", pad_inches=0.015)
    fig.savefig(FIGURES / "fig_full_result_panel_20260525.pdf", bbox_inches="tight", pad_inches=0.015)
    fig.savefig(FIGURES / "fig_full_result_panel_20260525.png", dpi=320, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


if __name__ == "__main__":
    main()

