"""Run CSP+SVM baseline on hold-out split."""

import argparse
from pathlib import Path

from physionet_mi.baseline.csp_svm_classifier import main as run_main


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="CSP+SVM baseline hold-out")
    parser.add_argument("--config", type=str, default="configs/preprocess_ea.yaml")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--project-root", type=str, default=None)
    args = parser.parse_args(argv)

    root = Path(args.project_root).resolve() if args.project_root else Path.cwd()
    run_main(str(root / args.config), args.run_name, project_root=root)


if __name__ == "__main__":
    main()
