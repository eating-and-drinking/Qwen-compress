#!/usr/bin/env python3
"""
CLI Entry Point
Usage:
  python run.py --config configs/pipeline.json
  python run.py --config configs/pipeline.json --target 50000
  python run.py --mode merge --output_dir outputs/cot_dataset
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)


def cmd_generate(args):
    from pipeline import run_pipeline

    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)

    # CLI args override config
    if args.target:
        cfg["target_total"] = args.target
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.no_cross_val:
        cfg["enable_cross_validation"] = False

    run_pipeline(cfg)


def cmd_merge(args):
    from formatter import merge_cot_and_direct
    merge_cot_and_direct(args.output_dir, cot_ratio=args.cot_ratio)


def cmd_stats(args):
    """Show basic statistics of generated dataset"""
    import json
    base = Path(args.output_dir)
    for name in ["train_cot", "train_direct", "train_mixed", "val", "test"]:
        path = base / f"{name}.jsonl"
        if path.exists():
            with open(path) as f:
                count = sum(1 for _ in f)
            print(f"  {name:20s}: {count:>8,} items")
    meta = base / "metadata.json"
    if meta.exists():
        print(f"\nMetadata: {meta}")


def main():
    parser = argparse.ArgumentParser(description="CoT Generator CLI")
    sub = parser.add_subparsers(dest="command")

    # generate
    p_gen = sub.add_parser("generate", help="Run generation pipeline")
    p_gen.add_argument("--config", default="configs/pipeline.json", help="Pipeline config file")
    p_gen.add_argument("--target", type=int, help="Override target count")
    p_gen.add_argument("--output_dir", help="Override output directory")
    p_gen.add_argument("--no_cross_val", action="store_true", help="Disable cross validation (faster)")

    # merge
    p_merge = sub.add_parser("merge", help="Merge CoT/Direct training sets")
    p_merge.add_argument("--output_dir", required=True)
    p_merge.add_argument("--cot_ratio", type=float, default=0.7)

    # stats
    p_stats = sub.add_parser("stats", help="Show dataset statistics")
    p_stats.add_argument("--output_dir", required=True)

    # Default behavior: run generate when no subcommand
    args = parser.parse_args()
    if args.command is None:
        args.command = "generate"
        args.config = "configs/pipeline.json"
        args.target = None
        args.output_dir = None
        args.no_cross_val = False

    dispatch = {
        "generate": cmd_generate,
        "merge": cmd_merge,
        "stats": cmd_stats,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
