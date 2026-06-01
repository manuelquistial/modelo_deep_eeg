"""Run FBCSP+LDA baseline on hold-out split."""

import argparse
from pathlib import Path

from physionet_mi.baseline.lda_fbcsp import main as run_main


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="LDA+FBCSP baseline hold-out")
    parser.add_argument("--config", type=str, default="configs/preprocess_ea.yaml")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--project-root", type=str, default=None)
    args = parser.parse_args(argv)

    root = Path(args.project_root).resolve() if args.project_root else Path.cwd()
    run_main(str(root / args.config), args.run_name)


if __name__ == "__main__":
    main()
