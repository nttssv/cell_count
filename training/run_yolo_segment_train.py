"""Run Ultralytics YOLO segmentation training from key=value arguments.

This wrapper exists because `python -m ultralytics` is not supported by the
installed Ultralytics package in some cluster environments.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def parse_value(value: str):
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower == "none":
        return None
    try:
        if value.strip() and all(ch not in value for ch in ".eE"):
            return int(value)
        return float(value)
    except ValueError:
        return value


def parse_key_value_args(argv: list[str]) -> dict:
    args: dict[str, object] = {}
    for item in argv:
        if "=" not in item:
            raise SystemExit(f"Expected key=value argument, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip().replace("-", "_")
        if not key:
            raise SystemExit(f"Empty key in argument: {item}")
        args[key] = parse_value(value)
    return args


def main() -> None:
    args = parse_key_value_args(sys.argv[1:])
    if "model" not in args:
        raise SystemExit("Missing required argument: model=...")
    if "data" not in args:
        raise SystemExit("Missing required argument: data=...")

    model_path = str(args.pop("model"))
    data_yaml = str(args.pop("data"))
    args.setdefault("task", "segment")

    from ultralytics import YOLO

    print("YOLO wrapper config:")
    print(json.dumps({"model": model_path, "data": data_yaml, **args}, indent=2, default=str))

    model = YOLO(model_path)
    results = model.train(data=data_yaml, **args)
    save_dir = getattr(results, "save_dir", None)
    if save_dir:
        print(f"YOLO save_dir: {Path(save_dir)}")


if __name__ == "__main__":
    main()
