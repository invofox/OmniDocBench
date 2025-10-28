#!/usr/bin/env python
import argparse
import subprocess
from pathlib import Path
from typing import Any, Generator

import yaml


def flatten(prefix: str, node: Any) -> Generator[str, None, None]:
    """Recursively flatten YAML structure into file paths"""
    if isinstance(node, list):
        for item in node:
            if isinstance(item, (dict, list)):
                yield from flatten(prefix, item)
            else:
                yield str(Path(prefix) / item)
    elif isinstance(node, dict):
        for key, val in node.items():
            yield from flatten(Path(prefix) / key, val)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage datasets with DVC")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["pull", "push"],
        default="push",
        help="Operation mode: pull or push data files using DVC",
    )
    parser.add_argument(
        "--datasets_specs",
        type=str,
        default="datasets.yaml",
        help="Path to the datasets YAML file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets_file = Path(args.datasets_specs)
    if not datasets_file.is_file():
        raise FileNotFoundError(f"Datasets file not found: {datasets_file}")
    with datasets_file.open() as f:
        datasets = yaml.safe_load(f)

    all_paths = list(flatten("data", datasets))

    base_command = [["dvc", "add"]] if args.mode == "push" else [["dvc", "pull", "--force"], ["dvc", "checkout"]]
    for cmd in base_command:
        complete_command = cmd + all_paths

        print(f"Running: {' '.join(complete_command)}")
        subprocess.run(complete_command, check=True)


if __name__ == "__main__":
    main()
