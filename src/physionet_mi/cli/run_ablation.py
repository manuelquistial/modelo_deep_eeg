"""Run full ablation: no_ea/ea x holdout/loso (+ LDA holdout)."""

import argparse
import json
from pathlib import Path

import pandas as pd

from physionet_mi.baseline.lda_fbcsp import run_lda_holdout
from physionet_mi.config import load_config
from physionet_mi.training.holdout import run_holdout
from physionet_mi.training.loso import run_loso
from physionet_mi.utils.logging import setup_logging


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--max-folds", type=int, default=3, help="LOSO folds for dev runs")
    parser.add_argument("--skip-loso", action="store_true")
    parser.add_argument("--skip-lda", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.project_root).resolve() if args.project_root else Path.cwd()
    rows = []

    for use_ea, cfg_name in [(False, "preprocess_no_ea.yaml"), (True, "preprocess_ea.yaml")]:
        cfg_path = root / "configs" / cfg_name
        cfg = load_config(cfg_path, project_root=root)
        tag = "ea" if use_ea else "no_ea"

        m = run_holdout(cfg, f"dl_{tag}_holdout")
        rows.append({"run": f"dl_{tag}_holdout", "use_ea": use_ea, "protocol": "holdout", **m})

        if not args.skip_loso:
            s = run_loso(cfg, f"dl_{tag}_loso", max_folds=args.max_folds)
            rows.append({
                "run": f"dl_{tag}_loso",
                "use_ea": use_ea,
                "protocol": "loso",
                "accuracy": s["accuracy_mean"],
                "balanced_accuracy": s["balanced_accuracy_mean"],
                "macro_f1": s["macro_f1_mean"],
                "kappa": s["kappa_mean"],
                "accuracy_std": s["accuracy_std"],
            })

        if not args.skip_lda and use_ea:
            m_lda = run_lda_holdout(cfg, f"lda_{tag}_holdout")
            rows.append({"run": f"lda_{tag}_holdout", "use_ea": use_ea, "protocol": "holdout", **m_lda})

    df = pd.DataFrame(rows)
    out = root / "outputs" / "ablation_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    (root / "outputs" / "ablation_summary.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print(f"Saved ablation summary to {out}")


if __name__ == "__main__":
    main()
