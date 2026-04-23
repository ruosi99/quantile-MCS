import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCENARIOS = [
    {
        "scenario": "Balanced",
        "c_over": 1.0,
        "c_under": 1.0,
        "critical_fractile": 0.50,
        "recommended_strategy": "q0.50",
    },
    {
        "scenario": "Shortage-4x",
        "c_over": 1.0,
        "c_under": 4.0,
        "critical_fractile": 0.80,
        "recommended_strategy": "q0.80",
    },
    {
        "scenario": "Shortage-9x",
        "c_over": 1.0,
        "c_under": 9.0,
        "critical_fractile": 0.90,
        "recommended_strategy": "q0.90",
    },
    {
        "scenario": "Shortage-19x",
        "c_over": 1.0,
        "c_under": 19.0,
        "critical_fractile": 0.95,
        "recommended_strategy": "q0.95",
    },
]

STRATEGY_TO_INDEX = {
    "q0.50": 3,
    "q0.80": 4,
    "q0.90": 5,
    "q0.95": 6,
}

COLORS = {
    "q0.50": "#1f4e79",
    "q0.80": "#4c956c",
    "q0.90": "#f4a259",
    "q0.95": "#d1495b",
}


def evaluate_costs(result_dir: str) -> pd.DataFrame:
    pred_quantiles = np.load(os.path.join(result_dir, "predict_quantiles.npy"))
    labels = np.load(os.path.join(result_dir, "label_list.npy"))

    rows = []
    for scenario in SCENARIOS:
        scenario_costs = {}
        for strategy_name, q_idx in STRATEGY_TO_INDEX.items():
            decision = pred_quantiles[:, :, q_idx]
            over_supply = np.maximum(decision - labels, 0.0)
            under_supply = np.maximum(labels - decision, 0.0)
            cost = scenario["c_over"] * over_supply + scenario["c_under"] * under_supply
            total_cost = float(cost.sum())
            mean_cost = float(cost.mean())
            scenario_costs[strategy_name] = total_cost
            rows.append(
                {
                    "scenario": scenario["scenario"],
                    "c_over": scenario["c_over"],
                    "c_under": scenario["c_under"],
                    "critical_fractile": scenario["critical_fractile"],
                    "recommended_strategy": scenario["recommended_strategy"],
                    "strategy": strategy_name,
                    "total_cost": total_cost,
                    "mean_cost": mean_cost,
                }
            )

        q50_total = scenario_costs["q0.50"]
        best_total = min(scenario_costs.values())
        for row in rows[-len(STRATEGY_TO_INDEX):]:
            row["cost_vs_q50_pct"] = 100.0 * row["total_cost"] / q50_total
            row["improvement_vs_q50_pct"] = 100.0 * (q50_total - row["total_cost"]) / q50_total
            row["is_best"] = row["total_cost"] == best_total

    return pd.DataFrame(rows)


def plot_cost_comparison(df: pd.DataFrame, output_path: str) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6))

    scenarios = [s["scenario"] for s in SCENARIOS]
    strategies = list(STRATEGY_TO_INDEX.keys())
    x = np.arange(len(scenarios))
    width = 0.18

    for offset, strategy in enumerate(strategies):
        subset = (
            df[df["strategy"] == strategy]
            .set_index("scenario")
            .loc[scenarios]
            .reset_index()
        )
        xpos = x + (offset - 1.5) * width
        bars = ax.bar(
            xpos,
            subset["cost_vs_q50_pct"],
            width=width,
            color=COLORS[strategy],
            label=strategy,
            edgecolor="white",
            linewidth=0.8,
        )
        if strategy != "q0.50":
            for bar, value in zip(bars, subset["improvement_vs_q50_pct"]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 1.0,
                    f"{value:+.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    rotation=90,
                )

    ax.axhline(100.0, color="#333333", linestyle="--", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{s['scenario']}\n($c_o={s['c_over']:.0f}, c_u={s['c_under']:.0f}$)" for s in SCENARIOS],
        fontsize=10,
    )
    ax.set_ylabel("Total economic cost relative to q0.50 (%)", fontsize=12)
    ax.set_title("Risk-Aware Economic Evaluation Across Asymmetric Cost Settings", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(title="Decision strategy", ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Risk-aware economic evaluation from saved quantile predictions.")
    parser.add_argument(
        "--result_dir",
        type=str,
        default="canonical_main_results",
        help="Canonical result folder containing predict_quantiles.npy and label_list.npy",
    )
    args = parser.parse_args()

    df = evaluate_costs(args.result_dir)
    csv_path = os.path.join(args.result_dir, "risk_aware_economic_evaluation.csv")
    fig_path = os.path.join(args.result_dir, "risk_aware_economic_evaluation.png")

    df.to_csv(csv_path, index=False)
    plot_cost_comparison(df, fig_path)

    best_rows = df[df["is_best"]].sort_values("scenario")
    print(f"Saved table: {csv_path}")
    print(f"Saved figure: {fig_path}")
    print(best_rows[["scenario", "strategy", "total_cost", "improvement_vs_q50_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
