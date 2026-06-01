"""Train EEGMeModel with LOSO cross-validation."""

import argparse
from pathlib import Path

from physionet_mi.training.loso import main as run_main


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train LOSO EEGMeModel")
    parser.add_argument("--config", type=str, default="configs/preprocess_no_ea.yaml")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--project-root", type=str, default=None)
    args = parser.parse_args(argv)

    root = Path(args.project_root).resolve() if args.project_root else Path.cwd()
    config_path = str(root / args.config)
    run_main(config_path, args.run_name, max_folds=args.max_folds)


if __name__ == "__main__":
    main()
