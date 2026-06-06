"""Run CellSeg1 LoRA training from a YAML config.

This wrapper keeps the cluster notebook simple and avoids relying on the
CellSeg1 repository being installed as a package.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path, help="Path to a cloned Nuisal/cellseg1 repo.")
    parser.add_argument("--config", required=True, type=Path, help="Runtime CellSeg1 YAML config.")
    parser.add_argument("--summary", type=Path, default=None, help="Optional JSON summary output path.")
    args = parser.parse_args()

    repo = args.repo.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    if not repo.exists():
        raise FileNotFoundError(f"CellSeg1 repo not found: {repo}")
    if not config_path.exists():
        raise FileNotFoundError(f"CellSeg1 config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    sys.path.insert(0, str(repo))
    from cellseg1_train import main as train_main

    started = time.time()
    print("CellSeg1 wrapper config:")
    print(
        json.dumps(
            {
                "repo": str(repo),
                "config": str(config_path),
                "train_image_dir": config.get("train_image_dir"),
                "train_mask_dir": config.get("train_mask_dir"),
                "result_pth_path": config.get("result_pth_path"),
                "model_path": config.get("model_path"),
                "epoch_max": config.get("epoch_max"),
                "batch_size": config.get("batch_size"),
                "gradient_accumulation_step": config.get("gradient_accumulation_step"),
                "train_id": config.get("train_id"),
            },
            indent=2,
            default=str,
        )
    )

    train_main(config_path)
    elapsed = time.time() - started
    result_path = Path(config["result_pth_path"])
    summary = {
        "repo": str(repo),
        "config": str(config_path),
        "result_pth_path": str(result_path),
        "result_exists": result_path.exists(),
        "elapsed_seconds": elapsed,
    }
    print("CellSeg1 training summary:")
    print(json.dumps(summary, indent=2, default=str))
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
