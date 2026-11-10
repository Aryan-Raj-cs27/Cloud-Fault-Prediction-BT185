"""Fault-aware scheduling experiments: FARS vs baselines, sanity checks and ablations.

Steps (all thresholds tuned on VALIDATION seeds with the VALIDATION score distribution;
final numbers use disjoint TEST seeds). Score distributions contain only blocks still running
after the observation window (strict early warning, D-013):
  1. tune FARS (tau_low, tau_high) -> configs/scheduler/fars_tuned.yaml + heatmap (ablation d)
  2. FARS vs Random / Round-Robin / First-Fit / Best-Fit / Least-Loaded / Reactive (20 seeds),
     FARS driven by the TEST score distribution
  2b. the same test seeds with FARS driven by the VALIDATION score distribution (D-014)
  3. sanity: FARS with an oracle predictor and with a random predictor
  4. ablations: (a) no anomaly detector, (b) no calibration, (c) other horizons, (e) noise
  5. a recorded replay (FARS vs least-loaded) for the dashboard

Usage:
    python scripts/run_simulation.py [--quick]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from faultaware.eval import plots
from faultaware.eval.stats import fmt_mean_std, paired_tests
from faultaware.eval.tables import write_table
from faultaware.scheduler.fars import FARSConfig
from faultaware.simulator.engine import SimConfig
from faultaware.simulator.experiment import run_many, run_one, tune_fars
from faultaware.utils.config import CONFIG_DIR, load_config, project_path
from faultaware.utils.logging import get_logger

log = get_logger("run_simulation")

KEY_METRICS = [
    "task_interruptions",
    "wasted_compute_min",
    "lost_node_min",
    "sla_violation_rate",
    "mean_response_min",
    "makespan_min",
    "utilisation",
    "migrations",
    "checkpoint_overhead_min",
    "energy_kwh",
]


def summarise(df: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    """Mean ± std per scheduler for the key metrics."""
    rows = []
    for name in order:
        sub = df[df.scheduler == name]
        if sub.empty:
            continue
        rows.append(
            {
                "scheduler": name,
                "seeds": len(sub),
                **{
                    m: fmt_mean_std(sub[m], 3 if m in ("sla_violation_rate", "utilisation") else 1)
                    for m in KEY_METRICS
                },
            }
        )
    return pd.DataFrame(rows)


def significance(df: pd.DataFrame, ref: str, others: list[str]) -> pd.DataFrame:
    """Paired Wilcoxon / t-tests of ``ref`` vs each other scheduler on the same seeds."""
    rows = []
    a = df[df.scheduler == ref].sort_values("seed")
    for other in others:
        b = df[df.scheduler == other].sort_values("seed")
        for metric in (
            "task_interruptions",
            "wasted_compute_min",
            "lost_node_min",
            "sla_violation_rate",
            "mean_response_min",
        ):
            res = paired_tests(a[metric].to_numpy(), b[metric].to_numpy())
            base = b[metric].mean()
            rows.append(
                {
                    "A": ref,
                    "B": other,
                    "metric": metric,
                    "mean_A": a[metric].mean(),
                    "mean_B": base,
                    "rel_change_%": 100 * (a[metric].mean() - base) / base if base else 0.0,
                    **res,
                }
            )
    return pd.DataFrame(rows)


SCORE_SOURCES = {
    "test": "test period, still-running blocks (strict)",
    "val": "validation period, still-running blocks (pessimistic)",
}


def score_source_table(
    fars_by_source: dict[str, pd.DataFrame], main_df: pd.DataFrame, baselines: list[str]
) -> pd.DataFrame:
    """FARS under each risk-score distribution vs the best baseline (D-014).

    Baselines ignore risk scores and the signal has its own random stream, so their runs are
    identical under every score source and the same paired seeds can be reused.
    """
    ll = main_df[main_df.scheduler == "least_loaded"].sort_values("seed")
    best = {
        m: min(main_df[main_df.scheduler == b][m].mean() for b in baselines)
        for m in ("task_interruptions", "lost_node_min")
    }
    rows = []
    for split, df in fars_by_source.items():
        df = df.sort_values("seed")
        row: dict[str, Any] = {"score_source": SCORE_SOURCES[split]}
        for m in ("task_interruptions", "wasted_compute_min", "lost_node_min", "migrations"):
            row[m] = fmt_mean_std(df[m], 1)
        for m in ("task_interruptions", "lost_node_min"):
            row[f"{m}_vs_best_baseline_%"] = round(100 * (df[m].mean() - best[m]) / best[m], 1)
            row[f"{m}_p_vs_least_loaded"] = paired_tests(df[m].to_numpy(), ll[m].to_numpy())[
                "wilcoxon_p"
            ]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="few seeds (smoke test)")
    parser.add_argument("--n-jobs", type=int, default=8)
    args, overrides = parser.parse_known_args()
    cfg = load_config(overrides)
    sim = SimConfig.from_cfg(cfg.simulator)
    scfg = cfg.scheduler
    val_seeds = list(cfg.simulator.seeds.validation)[: 2 if args.quick else None]
    test_seeds = list(cfg.simulator.seeds.test)[: 3 if args.quick else None]
    scores = str(project_path(cfg.simulator.signal.scores_file))
    figs, tables = project_path(cfg.paths.figures_dir), project_path(cfg.paths.tables_dir)
    base = FARSConfig(
        w_free=scfg.weights.free_capacity,
        w_risk=scfg.weights.risk,
        w_migration=scfg.weights.migration_cost,
        smoothing=scfg.risk_smoothing,
        elevated_interval=scfg.elevated_checkpoint_interval_min,
        default_interval=cfg.simulator.mitigation.checkpoint_interval_min,
    )
    persistent = bool(cfg.simulator.signal.persistent)

    def spec(
        split: str,
        variant: str = "fused",
        file: str = scores,
        source: str = "empirical",
        noise: float = 0.0,
    ) -> dict[str, Any]:
        return {
            "source": source,
            "scores_file": file,
            "split": split,
            "variant": variant,
            "noise": noise,
            "persistent": persistent,
        }

    def tune(variant: str = "fused", file: str = scores) -> tuple[FARSConfig, np.ndarray]:
        return tune_fars(
            sim,
            base,
            val_seeds,
            spec("val", variant, file),
            list(scfg.tau_grid_low),
            list(scfg.tau_grid_high),
            args.n_jobs,
        )

    summary: dict[str, Any] = {"sim_config": asdict(sim)}
    # 1. tune
    fars_cfg, grid = tune()
    log.info("tuned FARS: tau_low=%.2f tau_high=%.2f", fars_cfg.tau_low, fars_cfg.tau_high)
    (CONFIG_DIR / "scheduler" / "fars_tuned.yaml").write_text(
        "# Generated by scripts/run_simulation.py (validation seeds) — do not edit.\n"
        + OmegaConf.to_yaml(
            OmegaConf.create({"tau_low": fars_cfg.tau_low, "tau_high": fars_cfg.tau_high})
        ),
        encoding="utf-8",
    )
    plots.heatmap(
        grid,
        list(scfg.tau_grid_low),
        list(scfg.tau_grid_high),
        "tau_low",
        "tau_high",
        "FARS threshold sweep: lost node-minutes (validation)",
        figs / "fars_threshold_sweep",
    )
    summary["fars"] = asdict(fars_cfg)
    summary["threshold_grid"] = {
        "tau_low": list(scfg.tau_grid_low),
        "tau_high": list(scfg.tau_grid_high),
        "lost_node_min": np.where(np.isnan(grid), None, grid).tolist(),
    }

    # 2. main comparison on test seeds
    baselines = list(scfg.baselines)
    order = [*baselines, "fars"]
    main_df = pd.concat(
        [
            run_many(
                sim,
                baselines,
                test_seeds,
                spec("test"),
                None,
                args.n_jobs,
                scfg.reactive_cooldown_min,
            ),
            run_many(sim, ["fars"], test_seeds, spec("test"), fars_cfg, args.n_jobs),
        ],
        ignore_index=True,
    )
    table = summarise(main_df, order)
    write_table(table, tables / "scheduler_comparison")
    sig = significance(main_df, "fars", baselines)
    write_table(sig, tables / "scheduler_significance")
    for metric, label in (
        ("task_interruptions", "Task interruptions (count)"),
        ("wasted_compute_min", "Wasted compute (min)"),
        ("lost_node_min", "Lost node time (min)"),
        ("mean_response_min", "Mean response time (min)"),
    ):
        g = main_df.groupby("scheduler")[metric]
        plots.bar_with_error(
            order,
            [g.mean()[s] for s in order],
            [g.std()[s] for s in order],
            label,
            f"{label}: FARS vs baselines (test seeds, mean ± std)",
            figs / f"scheduler_{metric}",
            highlight="fars",
        )

    # 2b. score-source robustness (D-014): test-period scores are easier than validation-period
    # scores even on still-running blocks, so FARS is also run on the test seeds with the
    # validation distribution (same tau, same workloads and faults).
    fars_val = run_many(sim, ["fars"], test_seeds, spec("val"), fars_cfg, args.n_jobs)
    sources = score_source_table(
        {"test": main_df[main_df.scheduler == "fars"], "val": fars_val}, main_df, baselines
    )
    write_table(sources, tables / "scheduler_score_sources")
    both = pd.concat([main_df[main_df.scheduler != "fars"], fars_val], ignore_index=True)
    write_table(significance(both, "fars", baselines), tables / "scheduler_significance_valscores")

    # 3. sanity checks
    sanity = pd.concat(
        [
            run_many(
                sim, ["fars"], test_seeds, spec("test", source="oracle"), fars_cfg, args.n_jobs
            ).assign(scheduler="fars+oracle"),
            run_many(
                sim, ["fars"], test_seeds, spec("test", source="random"), fars_cfg, args.n_jobs
            ).assign(scheduler="fars+random"),
            main_df[main_df.scheduler.isin(["least_loaded", "fars"])],
        ],
        ignore_index=True,
    )
    write_table(
        summarise(sanity, ["least_loaded", "fars+random", "fars", "fars+oracle"]),
        tables / "simulator_sanity",
    )
    sanity_sig = pd.concat(
        [
            significance(sanity, "fars+oracle", ["least_loaded"]),
            significance(sanity, "fars+random", ["least_loaded"]),
        ]
    )
    write_table(sanity_sig, tables / "simulator_sanity_significance")

    # 4. ablations
    abl_rows = [main_df[main_df.scheduler == "fars"].assign(variant="FARS (full)")]
    ablation_specs = [
        ("(a) no anomaly detector", "supervised_cal", scores),
        ("(b) no calibration", "supervised_raw", scores),
    ]
    dist_dir = project_path(cfg.simulator.signal.scores_file).parent
    for tag in ("w10", "full"):
        f = dist_dir / f"score_distributions_{tag}.npz"
        if f.is_file():
            ablation_specs.append((f"(c) horizon: {tag}", "fused", str(f)))
    tuned_taus = {"FARS (full)": (fars_cfg.tau_low, fars_cfg.tau_high)}
    for label, variant, file in ablation_specs:
        cfg_v, _ = tune(variant, file)
        tuned_taus[label] = (cfg_v.tau_low, cfg_v.tau_high)
        abl_rows.append(
            run_many(
                sim, ["fars"], test_seeds, spec("test", variant, file), cfg_v, args.n_jobs
            ).assign(variant=label)
        )
    abl = pd.concat(abl_rows, ignore_index=True)
    ll = main_df[main_df.scheduler == "least_loaded"]
    abl_table = []
    for label, sub in abl.groupby("variant", sort=False):
        res = paired_tests(
            sub.sort_values("seed")["lost_node_min"].to_numpy(),
            ll.sort_values("seed")["lost_node_min"].to_numpy(),
        )
        abl_table.append(
            {
                "variant": label,
                "tau_low/tau_high": str(tuned_taus[label]),
                **{
                    m: fmt_mean_std(sub[m], 1)
                    for m in (
                        "task_interruptions",
                        "wasted_compute_min",
                        "lost_node_min",
                        "migrations",
                    )
                },
                "vs_least_loaded_lost_p": res["wilcoxon_p"],
            }
        )
    write_table(pd.DataFrame(abl_table), tables / "ablations")

    noise_levels = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    noise_m, noise_s, noise_rows = [], [], []
    for eta in noise_levels:
        df = run_many(sim, ["fars"], test_seeds, spec("test", noise=eta), fars_cfg, args.n_jobs)
        noise_m.append(df.lost_node_min.mean())
        noise_s.append(df.lost_node_min.std())
        noise_rows.append(
            {
                "noise": eta,
                "lost_node_min": fmt_mean_std(df.lost_node_min, 1),
                "task_interruptions": fmt_mean_std(df.task_interruptions, 1),
                "migrations": fmt_mean_std(df.migrations, 1),
            }
        )
    write_table(pd.DataFrame(noise_rows), tables / "ablation_noise")
    ll_mean, ll_std = ll.lost_node_min.mean(), ll.lost_node_min.std()
    plots.line_sweep(
        noise_levels,
        {
            "fars": (noise_m, noise_s),
            "least_loaded": ([ll_mean] * len(noise_levels), [ll_std] * len(noise_levels)),
        },
        "Prediction noise η (fraction of scores replaced by U(0,1))",
        "Lost node time (min)",
        "Noise sensitivity of FARS (test seeds)",
        figs / "ablation_noise",
    )

    # 5. replay trace for the dashboard (seed 0)
    replay = {}
    for name, fc in (("fars", fars_cfg), ("least_loaded", None)):
        row = run_one(sim, name, test_seeds[0], spec("test"), fc, record_trace=True)
        replay[name] = {
            "metrics": {k: v for k, v in row.items() if k != "trace"},
            "trace": row["trace"],
        }
    out_dir = project_path(cfg.paths.reports_dir)
    (out_dir / "simulation_replay.json").write_text(json.dumps(replay), encoding="utf-8")

    summary.update(
        {
            "comparison": table.to_dict(orient="records"),
            "significance": sig.to_dict(orient="records"),
            "score_sources": sources.to_dict(orient="records"),
            "sanity": summarise(
                sanity, ["least_loaded", "fars+random", "fars", "fars+oracle"]
            ).to_dict(orient="records"),
            "ablations": abl_table,
            "noise": noise_rows,
            "seeds": {"validation": val_seeds, "test": test_seeds},
        }
    )
    (out_dir / "simulation_summary.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8"
    )
    main_df.to_csv(tables / "scheduler_runs_raw.csv", index=False)
    log.info("simulation experiments complete")


if __name__ == "__main__":
    main()
