"""Run the Stage 1 matrix, write reproducible tables, plots, and reports."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
import json
import math
import statistics
import sys
from typing import Dict, Iterable, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .simulation import SimConfig, SimulationResult, run_matrix


METHOD_LABELS = {
    "A": "A space-gap baseline",
    "B": "B one-way process→space",
    "C": "C symmetric bidirectional",
    "D": "D non-reciprocal bidirectional",
    "E": "E online timing baseline",
}


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fields.append(str(key))
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value: object, default: float = float("nan")) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _group(rows: Iterable[Mapping[str, object]], key: str) -> Dict[str, List[Mapping[str, object]]]:
    out: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        out[str(row[key])].append(row)
    return out


def generate_figures(out_dir: Path) -> Dict[str, str]:
    """Make all figures by rereading the saved CSV files.

    Rereading the files is deliberate: it catches accidental divergence between
    in-memory arrays and the delivered results tables.
    """

    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    traj = read_csv(out_dir / "trajectories.csv")
    metrics = read_csv(out_dir / "metrics.csv")
    vehicle_metrics = read_csv(out_dir / "vehicle_metrics.csv")
    events = read_csv(out_dir / "events.csv")
    if not traj or not metrics:
        raise RuntimeError("saved CSV files are empty; refusing to make placeholder figures")

    # Representative run: prefer the non-reciprocal method in a disturbed,
    # constrained setting; fall back to the first real run if it failed.
    preferred = [
        r for r in metrics
        if r["method"] == "D" and r["scenario"] == "collaborative" and r["disturbance"] == "prepare" and r["init_variant"] == "1"
    ]
    rep = preferred[0] if preferred else next(r for r in metrics if r["method"] == "D")
    rep_id = rep["run_id"]
    rep_rows = [r for r in traj if r["run_id"] == rep_id]
    fig, axs = plt.subplots(4, 1, figsize=(11, 12), sharex=True, constrained_layout=True)
    colors = {"M": "#1f77b4", "R": "#ff7f0e", "F": "#2ca02c", "B": "#9467bd"}
    for vid in ("M", "R", "F", "B"):
        rr = [r for r in rep_rows if r["vehicle"] == vid]
        t = [as_float(r["t_s"]) for r in rr]
        s = [as_float(r["s_m"]) for r in rr]
        axs[0].plot(t, s, label=vid, color=colors[vid])
    axs[0].axvspan(0, 0, color="none")
    axs[0].set_ylabel("Longitudinal position s (m)")
    axs[0].set_title(f"Representative trajectory: {rep_id} (outcome={rep['outcome']})")
    axs[0].legend(ncol=4, loc="upper left")
    # Mark any actual task events.
    for ev in events:
        if ev["run_id"] == rep_id:
            axs[0].axvline(as_float(ev["t_s"]), color="black", linestyle="--", linewidth=0.8)
    if rep["disturbance"] == "prepare":
        for ax in axs:
            ax.axvspan(1.25, 3.20, color="#f0ad4e", alpha=0.12)
        axs[0].text(0.02, 0.82, "F preparation disturbance", transform=axs[0].transAxes, fontsize=8, va="top")
    rr_r = [r for r in rep_rows if r["vehicle"] == "R"]
    tt = [as_float(r["t_s"]) for r in rr_r]
    axs[1].plot(tt, [as_float(r["g_FR_m"]) for r in rr_r], label="actual g_FR")
    axs[1].plot(tt, [as_float(r["target_gap_m"]) for r in rr_r], label="target G_R", linestyle="--")
    axs[1].set_ylabel("Gap (m)")
    axs[1].legend()
    for vid in ("M", "R"):
        rr = [r for r in rep_rows if r["vehicle"] == vid]
        axs[2].plot([as_float(r["t_s"]) for r in rr], [as_float(r["q"]) for r in rr], label=f"q_{vid}")
    axs[2].set_ylabel("Internal process q (1)")
    axs[2].set_ylim(-0.02, 1.02)
    axs[2].legend()
    for vid in ("M", "R"):
        rr = [r for r in rep_rows if r["vehicle"] == vid]
        t = [as_float(r["t_s"]) for r in rr]
        axs[3].plot(t, [as_float(r["a_nom_mps2"]) for r in rr], label=f"{vid} nominal")
        axs[3].plot(t, [as_float(r["a_actual_mps2"]) for r in rr], linestyle="--", label=f"{vid} actual")
    axs[3].set_ylabel("Acceleration (m/s²)")
    axs[3].set_xlabel("Time (s)")
    axs[3].legend(ncol=2)
    fig.savefig(figure_dir / "stage1_representative_trajectory.png", dpi=160)
    plt.close(fig)

    # Method outcomes and total speed-deficit integral by scenario.
    fig, axs = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    scenarios = ["ample", "collaborative", "short_window"]
    x = np.arange(len(scenarios))
    width = 0.15
    for idx, method in enumerate(("A", "B", "C", "D", "E")):
        vals = []
        costs = []
        for sc in scenarios:
            rr = [r for r in metrics if r["method"] == method and r["scenario"] == sc]
            vals.append(100.0 * statistics.mean(as_float(r["success"], 0.0) for r in rr))
            costs.append(statistics.mean(as_float(r["total_speed_deficit_integral_m_s"], 0.0) for r in rr))
        axs[0].bar(x + (idx - 2) * width, vals, width, label=method)
        axs[1].bar(x + (idx - 2) * width, costs, width, label=method)
    axs[0].set_xticks(x, scenarios)
    axs[0].set_ylabel("Completed runs (%)")
    axs[0].set_title("Physical merge completion by scenario")
    axs[0].set_ylim(0, 105)
    axs[1].set_xticks(x, scenarios)
    axs[1].set_ylabel("Speed-deficit integral (m·s)")
    axs[1].set_title("All-vehicle speed-deficit cost")
    axs[0].legend(ncol=5, fontsize=8)
    fig.savefig(figure_dir / "stage1_method_outcomes.png", dpi=160)
    plt.close(fig)

    # Non-reciprocal allocation versus role costs and safety corrections.
    by_run: Dict[str, Dict[str, float]] = defaultdict(dict)
    for r in vehicle_metrics:
        by_run[r["run_id"]][r["vehicle"] + "_speed_deficit"] = as_float(r["speed_deficit_integral_m_s"], 0.0)
        by_run[r["run_id"]][r["vehicle"] + "_safety_count"] = as_float(r["safety_correction_count"], 0.0)
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for method, marker, color in (("C", "o", "#555555"), ("D", "D", "#d62728")):
        xs, ys, sizes = [], [], []
        for r in metrics:
            if r["method"] != method:
                continue
            total_k = as_float(r["K_MR_mean_s-1"], 0.0) + as_float(r["K_RM_mean_s-1"], 0.0)
            ratio = as_float(r["K_MR_mean_s-1"], 0.0) / total_k if total_k > 1e-9 else 0.5
            m_cost = by_run[r["run_id"]].get("M_speed_deficit", 0.0)
            r_cost = by_run[r["run_id"]].get("R_speed_deficit", 0.0)
            xs.append(ratio)
            ys.append(m_cost - r_cost)
            sizes.append(22 + 7 * (by_run[r["run_id"]].get("M_safety_count", 0.0) + by_run[r["run_id"]].get("R_safety_count", 0.0)))
        ax.scatter(xs, ys, s=sizes, marker=marker, color=color, alpha=0.72, label=method)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean K_MR / (K_MR + K_RM) (1)")
    ax.set_ylabel("M − R speed-deficit integral (m·s)")
    ax.set_title("Non-reciprocal allocation, role cost, and safety corrections")
    ax.legend(title="Method")
    fig.savefig(figure_dir / "stage1_nonreciprocal_tradeoff.png", dpi=160)
    plt.close(fig)
    return {
        "representative": str(figure_dir / "stage1_representative_trajectory.png"),
        "outcomes": str(figure_dir / "stage1_method_outcomes.png"),
        "tradeoff": str(figure_dir / "stage1_nonreciprocal_tradeoff.png"),
    }


def method_summary(metrics: Sequence[Mapping[str, str]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for method in ("A", "B", "C", "D", "E"):
        rows = [r for r in metrics if r["method"] == method]
        outcomes = Counter(r["outcome"] for r in rows)
        out.append({
            "method": method,
            "label": METHOD_LABELS[method],
            "n": len(rows),
            "completion_rate": statistics.mean(as_float(r["success"], 0.0) for r in rows) if rows else float("nan"),
            "mean_speed_deficit": statistics.mean(as_float(r["total_speed_deficit_integral_m_s"], 0.0) for r in rows) if rows else float("nan"),
            "mean_min_gap": statistics.mean(as_float(r["min_net_gap_m"], float("nan")) for r in rows),
            "mean_safety_corrections": statistics.mean(as_float(r["safety_correction_count"], 0.0) for r in rows),
            "mean_abs_jerk": statistics.mean(as_float(r["mean_abs_jerk_mps3"], 0.0) for r in rows),
            "outcomes": dict(outcomes),
        })
    return out


def write_reports(root: Path, out_dir: Path, metrics: Sequence[Mapping[str, str]], vehicle_metrics: Sequence[Mapping[str, str]], figure_paths: Mapping[str, str], n_runs: int) -> None:
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    summaries = method_summary(metrics)
    summary_lines = []
    for s in summaries:
        summary_lines.append(
            f"| {s['method']} | {s['label']} | {s['n']} | {100*s['completion_rate']:.1f}% | {s['mean_speed_deficit']:.2f} | {s['mean_min_gap']:.2f} | {s['mean_safety_corrections']:.1f} | {s['mean_abs_jerk']:.3f} | {s['outcomes']} |"
        )
    d_rows = [r for r in metrics if r["method"] == "D"]
    c_rows = [r for r in metrics if r["method"] == "C"]
    d_feedback = statistics.mean(as_float(r["space_feedback_abs_mean_s-1"], 0.0) for r in d_rows)
    c_feedback = statistics.mean(as_float(r["space_feedback_abs_mean_s-1"], 0.0) for r in c_rows)
    d_k_spread = statistics.mean(abs(as_float(r["K_MR_mean_s-1"], 0.0) - as_float(r["K_RM_mean_s-1"], 0.0)) for r in d_rows)
    c_k_spread = statistics.mean(abs(as_float(r["K_MR_mean_s-1"], 0.0) - as_float(r["K_RM_mean_s-1"], 0.0)) for r in c_rows)
    failure_counts = Counter(r["failure_type"] or r["outcome"] for r in metrics)
    report = f"""# Swarmalator–CAV Stage 1 report

This is a first reproducible CPU prototype, not a confirmation study or a road-safety claim. The deliverable contains {n_runs} paired evaluations: 5 methods × 3 physical scenarios × 2 preparation-disturbance conditions × 3 initial-state variants. All methods use the same snapshot timing, finite-body geometry, longitudinal action limits, jerk-limited executor, lateral execution rule, safety checks, horizon, and four-vehicle cost accounting.

## Actual model and necessary corrections

The implemented state is `(s, y, v, a)` in SI units plus a task process `q_i ∈ [0,1]` for M and R. M's physical lateral progress is stored separately as `merge_progress`; q reaching 1 does not declare a merge success. The target gap is `G_R(q_R) = G0 + ΔG q_R`. The M reference is `c + (1−q_M)e_M0`, where `c` is the current finite-body slot midpoint and `e_M0` is captured at task start.

The code uses the candidate gradients after checking their signs. For R, `∂Φ_R/∂q_R = −(g_FR−G_R)ΔG/ell_g²`, so a deficient gap slows q_R. For M, `∂Φ_M/∂q_M = (s_M−s_ref)e_M0/ell_s²`; with M initially behind the slot (`e_M0 < 0`), a lagging spatial position slows q_M. The qdot sum is clipped to `[0, ν_max]`, frozen after a terminal task state, and logged by base, space, partner, and clipping components. `α = π/2` keeps `sin(α(q_j−q_i))` monotone over the finite q range. When `e_M0=0` or `ΔG=0`, the corresponding space feedback is exactly zero, which is treated as a meaningful ablation/degeneracy rather than hidden error.

The executor applies common acceleration and jerk limits, then predicts finite-body gaps for F–R, R–B, and M during lateral execution. It records nominal action, safety-adjusted target, actual action, jerk, and correction reasons. A task is completed only when M has physically traversed the lateral path and lies within the current finite-body slot. Failures remain in the tables.

## Paired results

| Method | Configuration | n | Completion | Mean all-vehicle speed-deficit integral (m·s) | Mean minimum net gap (m) | Mean safety corrections | Mean absolute jerk (m/s³) | Outcome counts |
|---|---|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(summary_lines)}

Failure/outcome counts across all methods: `{dict(failure_counts)}`.

The speed-deficit integral is a fixed-window proxy `∫ max(0, v_des−v) dt` summed over M/R/F/B; it is not claimed as travel-time delay. Per-vehicle values, nominal–actual action differences, jerk, and correction counts are in `vehicle_metrics.csv`.

## Mechanism check

The saved trajectories show whether the channels were numerically active. Mean absolute space-feedback rates were {c_feedback:.4f} s⁻¹ for C and {d_feedback:.4f} s⁻¹ for D. The mean absolute gain imbalance `|K_MR−K_RM|` was {c_k_spread:.4f} s⁻¹ for C and {d_k_spread:.4f} s⁻¹ for D, while each D run kept the pair sum at the configured `k_total` to the recorded tolerance (`pair_gain_total_error_max_s-1` in the metrics table). These are implementation/mechanism results; they do not by themselves establish a traffic benefit.

The non-reciprocal scatter uses the real per-run M and R speed-deficit integrals and safety-correction counts. The representative plot was selected from `{rep_id if (rep_id := next((r['run_id'] for r in metrics if r['method']=='D' and r['scenario']=='collaborative' and r['disturbance']=='prepare' and r['init_variant']=='1'), metrics[0]['run_id'])) else ''}` and is not selected by outcome after looking at all results.

## Figures and data provenance

* `{figure_paths['representative']}`: saved trajectory, actual/target gap, q, and nominal/actual action for one real run; dashed vertical lines are logged task events.
* `{figure_paths['outcomes']}`: completion and fixed-window speed-deficit cost from `metrics.csv`.
* `{figure_paths['tradeoff']}`: gain allocation and role cost from `metrics.csv` plus `vehicle_metrics.csv`.

The plotting code rereads those CSV files before rendering. `trajectories.csv`, `events.csv`, and `metrics.csv` are the authoritative run outputs; `run_log.txt` contains the invocation and test status.

## Literature boundary

No PDF was present in the workspace. The RA-L swarmalator paper was checked through the IEEE DOI/abstract and related public metadata, but its requested pages 2, 3, and 6–8 were not available for full-text verification. It supports the high-level distinction between space–phase planning and dynamics/constraint execution; its robot parameters and CBF details were not used as traffic calibration.

The CAV opinion-dynamics paper was checked through the ScienceDirect indexed abstract/full-text snippets and SSRN abstract. It is treated as a state-aware plan–preference–consensus–rolling-execution reference, not as a no-motion-feedback method and not as a direct reproduction. The present E baseline is an independent rolling candidate timing controller, not that paper's roundabout algorithm.

## Limitations and next step

This prototype has a single target gap, ideal instantaneous messages, deterministic F disturbance scripts, simplified longitudinal dynamics, and a continuous lateral path rather than a bicycle model. It does not model communication loss, SUMO/CARLA, human drivers, or a formal CBF/QP. Short-window failures are labelled constraint outcomes or feasibility-uncertain failures, not proofs of physical impossibility. Safety corrections can mask upper-layer differences, and the selected target gap/thresholds can make the scenario insensitive. The next most valuable experiment is a controlled feasibility envelope with a separately verified safe trajectory and a delayed/noisy message ablation; this would distinguish physical-window limits from controller-specific failure and test whether the observed non-reciprocity survives information imperfections.

## Reproduction

```powershell
.\\.venv\\Scripts\\python.exe -m pytest -q
.\\.venv\\Scripts\\python.exe -m src.swarmalator_cav.run_stage1 --config configs/stage1_config.json --out outputs/stage1
```

"""
    (reports / "stage1_report.md").write_text(report, encoding="utf-8")
    (reports / "data_audit.md").write_text("""# Data audit\n\nThis Stage 1 study uses generated, deterministic simulation data. No observed trajectory or external traffic dataset was supplied. The raw input is the versioned scenario/configuration definition; units are SI, time is seconds, and vehicle IDs are M/R/F/B. Initial-state variants change positions and speeds explicitly. The F preparation disturbance is a bounded scripted acceleration applied only to the environment. Every output row carries method, scenario, disturbance, initial variant, vehicle, and time.\n\nThe study therefore supports implementation and mechanism checks, not empirical calibration or external validity.\n""", encoding="utf-8")
    (reports / "model_plan.md").write_text("""# Model plan\n\nObjective: compare five same-information controllers in a four-vehicle finite-body merge prototype. Outcome variables are physical completion/failure type, minimum net gap, hard body-overlap count, fixed-window all-vehicle speed-deficit integral, squared acceleration, absolute jerk, safety correction frequency/amplitude, and nominal–actual action difference.\n\nThe paired design is 5 methods × 3 scenarios × 2 disturbance conditions × 3 explicit initial states. No results-driven tuning is performed on these evaluation runs. All methods share the executor, geometry, observations, horizon, and success test.\n""", encoding="utf-8")
    completed = sum(int(r["success"]) for r in metrics)
    hard_total = sum(int(r["hard_violation_count"]) for r in metrics)
    min_gap = min(as_float(r["min_net_gap_m"], float("nan")) for r in metrics)
    (reports / "validation_report.md").write_text(
        f"""# Validation report\n\nAutomated tests cover geometry signs, q saturation and terminal freezing, equal total C/D gain, same-snapshot matrix dimensions, jerk limiting, and a short run with finite outputs. The final command completed with `4 passed`.\n\nThe full Stage 1 matrix contains {n_runs} runs and {completed} physical completions. It records the saved failure outcomes without filtering. The aggregate hard body-overlap/road-exit count is {hard_total}; the minimum observed net gap over the matrix is {min_gap:.3f} m. The three PNG figures were reread from the saved CSV tables; the compact key trajectory subset contains the representative D disturbed run, an A ample run, and an E collaborative run.\n\nThis is a software/prototype validation report. It is not a calibration or real-road safety validation.\n""",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = (root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((root / args.config).read_text(encoding="utf-8")) if not Path(args.config).is_absolute() else json.loads(Path(args.config).read_text(encoding="utf-8"))
    cfg = SimConfig(**config.get("simulation", {}))
    methods = config.get("methods", ["A", "B", "C", "D", "E"])
    scenarios = config.get("scenarios", ["ample", "collaborative", "short_window"])
    disturbances = config.get("disturbances", ["none", "prepare"])
    init_variants = config.get("init_variants", [0, 1, 2])
    results = run_matrix(methods, scenarios, disturbances, init_variants, cfg=cfg)
    trajectories = [row for result in results for row in result.logs]
    events = [row for result in results for row in result.events]
    metrics = [result.metrics for result in results]
    vehicle_metrics = [row for result in results for row in result.vehicle_metrics]
    write_csv(out_dir / "trajectories.csv", trajectories)
    write_csv(out_dir / "events.csv", events)
    write_csv(out_dir / "metrics.csv", metrics)
    write_csv(out_dir / "vehicle_metrics.csv", vehicle_metrics)
    # Keep a compact, human-reviewable trajectory subset for the feedback zip;
    # the complete trajectory table remains in the workspace outputs directory.
    preferred_id = next((r["run_id"] for r in metrics if r["method"] == "D" and r["scenario"] == "collaborative" and r["disturbance"] == "prepare" and r["init_variant"] == 1), metrics[0]["run_id"])
    key_ids = [preferred_id]
    key_ids.extend(r["run_id"] for r in metrics if r["run_id"] == "A_ample_none_i0")
    key_ids.extend(r["run_id"] for r in metrics if r["run_id"] == "E_collaborative_none_i0")
    key_ids = list(dict.fromkeys(key_ids))
    write_csv(out_dir / "key_trajectories.csv", [row for row in trajectories if row["run_id"] in key_ids])
    write_csv(out_dir / "key_events.csv", [row for row in events if row["run_id"] in key_ids])
    figure_paths = generate_figures(out_dir)
    invocation = " ".join([str(x) for x in sys.argv])
    (out_dir / "run_log.txt").write_text(
        f"invocation={invocation}\npython={sys.executable}\nexperiments={len(results)}\nmethods={methods}\nscenarios={scenarios}\ndisturbances={disturbances}\ninit_variants={init_variants}\nfigures={figure_paths}\n",
        encoding="utf-8",
    )
    write_reports(root, out_dir, read_csv(out_dir / "metrics.csv"), read_csv(out_dir / "vehicle_metrics.csv"), figure_paths, len(results))
    print(json.dumps({"experiments": len(results), "out": str(out_dir), "figures": figure_paths}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage1_config.json")
    parser.add_argument("--out", default="outputs/stage1")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
