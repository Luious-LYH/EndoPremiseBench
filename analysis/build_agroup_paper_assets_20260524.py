"""Build paper-facing A-group tables and figures for EndoPremiseBench."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
TABLES = PAPER / "tables"
FIGURES = PAPER / "figures"
RESULTS = ROOT / "results"
AGROUP = RESULTS / "a_group_supplement_20260524"
DATE = "20260524"


MODEL_META = {
    "qwen3vl4b": ("Qwen3-VL-4B", "open general", "anchor"),
    "minicpmv4": ("MiniCPM-V-4", "open general", "anchor"),
    "qwen25_vl_3b": ("Qwen2.5-VL-3B", "open general", "anchor"),
    "internvl25_4b": ("InternVL2.5-4B", "open general", "anchor"),
    "qwen3_vl_8b": ("Qwen3-VL-8B", "open general", "A complete"),
    "qwen25_vl_7b": ("Qwen2.5-VL-7B", "open general", "A complete"),
    "medgemma_4b": ("MedGemma-4B", "open medical", "A complete"),
}


METRIC_PATHS = {
    "qwen3vl4b": RESULTS / "qwen3vl4b_premise_main6000_v2_metrics.json",
    "minicpmv4": RESULTS / "minicpmv4_premise_main6000_v2_metrics.json",
    "qwen25_vl_3b": RESULTS / "qwen25_vl_3b_premise_main6000_v2_metrics.json",
    "internvl25_4b": RESULTS / "internvl25_premise_main6000_v2_metrics.json",
    "qwen3_vl_8b": AGROUP / "metrics" / "qwen3_vl_8b_premise_main6000_metrics.json",
    "qwen25_vl_7b": AGROUP / "metrics" / "qwen25_vl_7b_premise_main6000_metrics.json",
    "medgemma_4b": AGROUP / "metrics" / "medgemma_4b_premise_main6000_metrics.json",
}

SCORED_PATHS = {
    "qwen3vl4b": RESULTS / "qwen3vl4b_premise_main6000_v2_scored.jsonl",
    "minicpmv4": RESULTS / "minicpmv4_premise_main6000_v2_scored.jsonl",
    "qwen25_vl_3b": RESULTS / "qwen25_vl_3b_premise_main6000_v2_scored.jsonl",
    "internvl25_4b": RESULTS / "internvl25_premise_main6000_v2_scored.jsonl",
    "qwen3_vl_8b": AGROUP / "scored" / "qwen3_vl_8b_premise_main6000_scored.jsonl",
    "qwen25_vl_7b": AGROUP / "scored" / "qwen25_vl_7b_premise_main6000_scored.jsonl",
    "medgemma_4b": AGROUP / "scored" / "medgemma_4b_premise_main6000_scored.jsonl",
}


PENDING = [
    ("InternVL2.5-8B", "open general", "running", "899 / 6000 raw rows on remote"),
    ("LLaVA-Med-v1.5-Mistral-7B", "open medical", "running", "1082 / 6000 raw rows on remote"),
    ("Lingshu-7B", "open medical", "not started", "smoke pass only"),
    ("MedGemma-KvasirVQA-x1-ft", "open endoscopy-ft", "running", "815 / 6000 raw rows on remote"),
    ("Qwen2.5-VL-KvasirVQA-x1-ft", "open endoscopy-ft", "not started", "smoke pass only"),
    ("Claude-Opus-4.7", "closed API", "running", "5476 / 6000 raw rows on remote"),
    ("GPT-5.5", "closed API", "running", "818 / 6000 raw rows on remote"),
    ("Gemini-3.1-Pro-High", "closed API", "running", "498 / 6000 raw rows on remote"),
    ("Grok-4.20-MA-xhigh", "closed API", "blocked", "smoke/token failure; shard artifacts not merged"),
]


def pct(x: float) -> float:
    return round(100.0 * float(x), 1)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def table_env(label: str, caption: str, header: list[str], rows: list[list[str]], star: bool = False) -> str:
    env = "table*" if star else "table"
    colspec = "l" * sum(1 for h in header if h in {"Model", "Type", "Status", "Notes", "Run"}) + "r" * sum(
        1 for h in header if h not in {"Model", "Type", "Status", "Notes", "Run"}
    )
    lines = [
        rf"\begin{{{env}}}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\linewidth}{!}{%",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", "}", rf"\end{{{env}}}", ""]
    return "\n".join(lines)


def build_main_tables(metrics: dict[str, dict]) -> None:
    rows = []
    order = ["qwen3_vl_8b", "qwen3vl4b", "qwen25_vl_7b", "qwen25_vl_3b", "minicpmv4", "internvl25_4b", "medgemma_4b"]
    for key in order:
        m = metrics[key]
        name, group, run = MODEL_META[key]
        rows.append(
            {
                "model": name,
                "type": group,
                "run": run,
                "n": int(m["n"]),
                "Acc_TP": pct(m["Acc_TP"]),
                "Acc_FP": pct(m["Acc_FP"]),
                "SR": pct(m["SR"]),
                "ORR": pct(m["ORR"]),
                "PFR": pct(m["PFR"]),
                "Ambig": pct(m.get("ambiguous_rate", 0.0)),
                "HPS": pct(m["HPS"]),
            }
        )
    fields = ["model", "type", "run", "n", "Acc_TP", "Acc_FP", "SR", "ORR", "PFR", "Ambig", "HPS"]
    write_csv(TABLES / f"agroup_main_results_{DATE}.csv", rows, fields)
    tex_rows = [
        [
            r["model"],
            r["type"],
            r["run"],
            str(r["n"]),
            f'{r["Acc_TP"]:.1f}',
            f'{r["Acc_FP"]:.1f}',
            f'{r["SR"]:.1f}',
            f'{r["ORR"]:.1f}',
            f'{r["HPS"]:.1f}',
        ]
        for r in rows
    ]
    tex = table_env(
        "tab:main_results",
        "Main EndoPremiseBench results on completed source-balanced main6000 runs. Values are percentages except $n$. A complete denotes newly completed A-group runs; anchor denotes earlier completed reference runs.",
        ["Model", "Type", "Run", "n", "Acc\\_TP", "Acc\\_FP", "SR", "ORR", "HPS"],
        tex_rows,
        star=True,
    )
    (TABLES / f"agroup_main_results_{DATE}.tex").write_text(tex, encoding="utf-8")

    status_rows = [[m, t, s, n] for m, t, s, n in PENDING]
    status_tex = table_env(
        "tab:pending_matrix",
        "A-group model matrix still running at draft time. Pending rows are shown for planning and must not be interpreted as paper results.",
        ["Model", "Type", "Status", "Notes"],
        status_rows,
        star=True,
    )
    (TABLES / f"agroup_pending_matrix_{DATE}.tex").write_text(status_tex, encoding="utf-8")


def build_delta_tables(metrics: dict[str, dict]) -> None:
    rows = [
        ("Qwen3 scale", "Qwen3-VL-4B", "Qwen3-VL-8B", "42.4", "42.8", "+0.4", "SR 53.9 -> 51.8"),
        ("Qwen2.5 scale", "Qwen2.5-VL-3B", "Qwen2.5-VL-7B", "35.5", "37.5", "+2.0", "SR 61.0 -> 58.2"),
        ("Medical specialization", "Qwen3-VL-8B", "MedGemma-4B", "42.8", "9.0", "-33.8", "SR 51.8 -> 94.8"),
    ]
    write_csv(
        TABLES / f"agroup_model_axis_deltas_{DATE}.csv",
        [
            {"axis": a, "from": b, "to": c, "HPS_from": d, "HPS_to": e, "delta": f, "note": g}
            for a, b, c, d, e, f, g in rows
        ],
        ["axis", "from", "to", "HPS_from", "HPS_to", "delta", "note"],
    )
    tex = table_env(
        "tab:model_axis",
        "Model-axis comparisons supported by completed runs. Scaling yields small HPS changes, while a medical model remains highly exposed under false premises.",
        ["Axis", "From", "To", "HPS\\_from", "HPS\\_to", "$\\Delta$HPS", "Notes"],
        [[a, b, c, d, e, f, g] for a, b, c, d, e, f, g in rows],
        star=True,
    )
    (TABLES / f"agroup_model_axis_deltas_{DATE}.tex").write_text(tex, encoding="utf-8")


def build_baseline_table() -> None:
    src = AGROUP / "analysis" / "trivial_baselines_20260524" / "summary.csv"
    rows = []
    with src.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["baseline"] in {"random_uniform", "majority_per_attribute", "always_NA", "never_NA_random", "premise_oracle_random"}:
                rows.append(
                    [
                        r["baseline"].replace("_", r"\_"),
                        r["n"],
                        f'{pct(r["Acc_TP"]):.1f}',
                        f'{pct(r["Acc_FP"]):.1f}',
                        f'{pct(r["SR"]):.1f}',
                        f'{pct(r["ORR"]):.1f}',
                        f'{pct(r["HPS"]):.1f}',
                    ]
                )
    tex = table_env(
        "tab:trivial_baselines",
        "Deterministic baselines computed from the manifest and options. They show why false-premise rejection alone is insufficient: always selecting N/A achieves perfect false-premise rejection but zero useful true-premise accuracy.",
        ["Baseline", "n", "Acc\\_TP", "Acc\\_FP", "SR", "ORR", "HPS"],
        rows,
    )
    (TABLES / f"agroup_trivial_baselines_{DATE}.tex").write_text(tex, encoding="utf-8")


def metric_from_rows(true_rows: list[dict], false_rows: list[dict]) -> dict[str, float]:
    acc_tp = sum(1 for r in true_rows if r.get("is_correct")) / max(1, len(true_rows))
    acc_fp = sum(1 for r in false_rows if r.get("is_correct")) / max(1, len(false_rows))
    sr = sum(1 for r in false_rows if r.get("parse_status") == "ok" and not r.get("is_correct")) / max(1, len(false_rows))
    orr = sum(1 for r in true_rows if r.get("failure_type") == "Over-Refusal") / max(1, len(true_rows))
    hps = 0.0 if acc_tp + acc_fp == 0 else 2 * acc_tp * acc_fp / (acc_tp + acc_fp)
    return {"Acc_TP": acc_tp, "Acc_FP": acc_fp, "SR": sr, "ORR": orr, "HPS": hps}


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[idx]


def bootstrap_ci_for(path: Path, iterations: int = 1000, seed: int = 20260524) -> dict[str, tuple[float, float, float]]:
    rows = read_jsonl(path)
    true_rows = [r for r in rows if r.get("premise_type") == "true"]
    false_rows = [r for r in rows if r.get("premise_type") == "false"]
    point = metric_from_rows(true_rows, false_rows)
    rng = random.Random(seed)
    samples: dict[str, list[float]] = {k: [] for k in point}
    for _ in range(iterations):
        bt = [true_rows[rng.randrange(len(true_rows))] for _ in true_rows]
        bf = [false_rows[rng.randrange(len(false_rows))] for _ in false_rows]
        m = metric_from_rows(bt, bf)
        for k, v in m.items():
            samples[k].append(v)
    return {k: (point[k], percentile(samples[k], 0.025), percentile(samples[k], 0.975)) for k in point}


def build_ci_table() -> None:
    rows = []
    csv_rows = []
    order = ["qwen3_vl_8b", "qwen3vl4b", "qwen25_vl_7b", "qwen25_vl_3b", "minicpmv4", "internvl25_4b", "medgemma_4b"]
    for key in order:
        cis = bootstrap_ci_for(SCORED_PATHS[key])
        name = MODEL_META[key][0]
        rows.append(
            [
                name,
                f'{pct(cis["Acc_TP"][0]):.1f} [{pct(cis["Acc_TP"][1]):.1f}, {pct(cis["Acc_TP"][2]):.1f}]',
                f'{pct(cis["Acc_FP"][0]):.1f} [{pct(cis["Acc_FP"][1]):.1f}, {pct(cis["Acc_FP"][2]):.1f}]',
                f'{pct(cis["SR"][0]):.1f} [{pct(cis["SR"][1]):.1f}, {pct(cis["SR"][2]):.1f}]',
                f'{pct(cis["HPS"][0]):.1f} [{pct(cis["HPS"][1]):.1f}, {pct(cis["HPS"][2]):.1f}]',
            ]
        )
        csv_row = {"model": name}
        for metric, vals in cis.items():
            csv_row[metric] = vals[0]
            csv_row[f"{metric}_lo"] = vals[1]
            csv_row[f"{metric}_hi"] = vals[2]
        csv_rows.append(csv_row)
    write_csv(
        TABLES / f"agroup_bootstrap_ci_{DATE}.csv",
        csv_rows,
        ["model", "Acc_TP", "Acc_TP_lo", "Acc_TP_hi", "Acc_FP", "Acc_FP_lo", "Acc_FP_hi", "SR", "SR_lo", "SR_hi", "ORR", "ORR_lo", "ORR_hi", "HPS", "HPS_lo", "HPS_hi"],
    )
    tex = table_env(
        "tab:bootstrap_ci",
        "Bootstrap 95\\% confidence intervals for completed main6000 runs. Values are percentages.",
        ["Model", "Acc\\_TP", "Acc\\_FP", "SR", "HPS"],
        rows,
        star=True,
    )
    (TABLES / f"agroup_bootstrap_ci_{DATE}.tex").write_text(tex, encoding="utf-8")


def build_figures(metrics: dict[str, dict]) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    keys = ["qwen3_vl_8b", "qwen3vl4b", "qwen25_vl_7b", "qwen25_vl_3b", "minicpmv4", "internvl25_4b", "medgemma_4b"]
    names = [MODEL_META[k][0].replace("-VL", "\nVL").replace("InternVL", "Intern\nVL") for k in keys]
    acc_tp = [pct(metrics[k]["Acc_TP"]) for k in keys]
    acc_fp = [pct(metrics[k]["Acc_FP"]) for k in keys]
    sr = [pct(metrics[k]["SR"]) for k in keys]
    hps = [pct(metrics[k]["HPS"]) for k in keys]

    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(8.2, 3.0))
    x = range(len(keys))
    ax.scatter(acc_tp, acc_fp, s=[max(30, v * 5) for v in hps], color="#2f6f9f", edgecolor="white", linewidth=1.0)
    for i, name in enumerate(names):
        ax.annotate(name.replace("\n", " "), (acc_tp[i], acc_fp[i]), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlim(25, 42)
    ax.set_ylim(0, 55)
    ax.set_xlabel("True-premise attribute accuracy (Acc_TP, %)")
    ax.set_ylabel("False-premise rejection (Acc_FP, %)")
    ax.grid(True, linewidth=0.35, color="#d0d0d0")
    ax.set_title("Useful answering and false-premise rejection remain jointly difficult")
    fig.tight_layout()
    fig.savefig(FIGURES / f"fig_agroup_frontier_{DATE}.pdf")
    fig.savefig(FIGURES / f"fig_agroup_frontier_{DATE}.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 3.1))
    bars = ax.bar(range(len(keys)), sr, color=["#4c78a8", "#4c78a8", "#72b7b2", "#72b7b2", "#59a14f", "#9c755f", "#e15759"])
    ax.axhline(50, color="#555555", linewidth=0.8, linestyle="--")
    ax.set_xticks(list(range(len(keys))))
    ax.set_xticklabels(names, rotation=0)
    ax.set_ylabel("Unsupported attribute exposure (SR, %)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linewidth=0.35, color="#d0d0d0")
    for b, v in zip(bars, sr):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_title("Completed A-group runs preserve high false-premise exposure")
    fig.tight_layout()
    fig.savefig(FIGURES / f"fig_agroup_exposure_{DATE}.pdf")
    fig.savefig(FIGURES / f"fig_agroup_exposure_{DATE}.png", dpi=220)
    plt.close(fig)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    metrics = {key: read_json(path) for key, path in METRIC_PATHS.items()}
    build_main_tables(metrics)
    build_delta_tables(metrics)
    build_baseline_table()
    build_ci_table()
    build_figures(metrics)
    print(json.dumps({"tables": str(TABLES), "figures": str(FIGURES), "models": len(metrics)}, indent=2))


if __name__ == "__main__":
    main()
