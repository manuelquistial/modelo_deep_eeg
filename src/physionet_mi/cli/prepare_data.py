"""Prepare and cache hold-out arrays (PhysioNet or BNCI2014_001 via config)."""

import argparse
from pathlib import Path

from physionet_mi.config import load_config
from physionet_mi.data.cache import build_and_cache_holdout
from physionet_mi.utils.logging import setup_logging


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Prepare dataset cache (PhysioNet or BNCI)")
    parser.add_argument("--config", type=str, default="configs/preprocess_no_ea.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--project-root", type=str, default=None)
    args = parser.parse_args(argv)

    root = Path(args.project_root).resolve() if args.project_root else Path.cwd()
    cfg = load_config(root / args.config, project_root=root)
    build_and_cache_holdout(cfg, force=args.force)


if __name__ == "__main__":
    main()
