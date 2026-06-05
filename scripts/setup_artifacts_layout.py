#!/usr/bin/env python3
"""Create the centralized artifacts/ layout and link legacy folders."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from physionet_mi.paths import (  # noqa: E402
    LEGACY_BASELINE_RUNS,
    LEGACY_CACHE,
    LEGACY_PUBLISHABLE_RUNS,
    LEGACY_RAW,
    LEGACY_SMOKE,
    REL_BASELINE_RUNS,
    REL_CACHE,
    REL_PUBLISHABLE_RUNS,
    REL_RAW,
    REL_SMOKE,
    artifacts_root,
    ensure_artifact_tree,
    find_project_root,
)


def _is_empty_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    for child in path.rglob("*"):
        if child.is_file():
            return False
    return True


def _symlink(target: Path, link: Path, dry_run: bool) -> bool:
    if link.is_symlink():
        print(f"  skip (symlink exists): {link} -> {link.readlink()}")
        return False
    if link.exists():
        if _is_empty_dir(link) and target.exists():
            print(f"  replace empty dir {link} with symlink to {target.name}")
            if not dry_run:
                shutil.rmtree(link)
        else:
            print(f"  skip (exists): {link}")
            return False
    if not target.exists():
        print(f"  skip (missing target): {target}")
        return False
    rel = os.path.relpath(target, link.parent)
    print(f"  link {link} -> {rel}")
    if not dry_run:
        link.symlink_to(rel)
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Setup centralized artifacts/ layout")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    root = find_project_root(ROOT)
    art = artifacts_root(root)
    print(f"Project root: {root}")
    print(f"Artifacts root: {art}")

    links = [
        (root / LEGACY_RAW, art / REL_RAW),
        (root / LEGACY_CACHE, art / REL_CACHE),
        (root / LEGACY_BASELINE_RUNS, art / REL_BASELINE_RUNS),
        (root / LEGACY_PUBLISHABLE_RUNS, art / REL_PUBLISHABLE_RUNS),
        (root / LEGACY_SMOKE, art / REL_SMOKE),
    ]
    print("\nLegacy symlinks:")
    linked = 0
    for target, link in links:
        if _symlink(target, link, args.dry_run):
            linked += 1
    print(f"\nLinked {linked} legacy folder(s).")

    created = ensure_artifact_tree(root)
    if created:
        print(f"Created {len(created)} directories under {art}")
    print("Done.")


if __name__ == "__main__":
    main()
