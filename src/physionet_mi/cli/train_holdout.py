"""Train EEGMeModel with subject hold-out."""

import argparse
from pathlib import Path

from physionet_mi.training.holdout import main as run_main


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train hold-out EEGMeModel")
    parser.add_argument("--config", type=str, default="configs/preprocess_no_ea.yaml")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--project-root", type=str, default=None)
    args = parser.parse_args(argv)

    root = Path(args.project_root).resolve() if args.project_root else Path.cwd()
    config_path = str(root / args.config)
    run_main(config_path, args.run_name)


if __name__ == "__main__":
    main()
