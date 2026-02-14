#!/usr/bin/env python3
"""
run_pipeline.py

Master pipeline script that runs all stages of the Part 2 analysis:

  1. Load & Identify  — reads part1 JSONL, builds author/paper tables with COI flags
  2. Network Build    — creates co-authorship network with centrality + communities
  3. Statistics       — runs chi-square, Fisher, odds ratio, permutation tests
  4. Visualizations   — generates static PNG plots + interactive HTML charts
  5. Network Viz      — generates interactive co-authorship network HTML

Usage:
  python run_pipeline.py --part1_dir ../part1/study_dump --output_dir ./output

  Then launch the dashboard:
    streamlit run viz/app.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def run_step(description: str, cmd: list[str]):
    """Run a pipeline step and exit on failure."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode != 0:
        print(f"\n[FAILED] {description}")
        sys.exit(result.returncode)
    print(f"\n[OK] {description}")


def main():
    ap = argparse.ArgumentParser(description="Run the full Part 2 analysis pipeline")
    ap.add_argument("--part1_dir", required=True,
                    help="Path to part1 output dir (contains data/pubmed_records.jsonl)")
    ap.add_argument("--output_dir", default="./output",
                    help="Where to write all part2 outputs (default: ./output)")
    ap.add_argument("--include_wos", action="store_true",
                    help="Include Web of Science records if available")
    ap.add_argument("--min_papers", type=int, default=2,
                    help="Min papers for an author to be included in the network (default: 2)")
    ap.add_argument("--max_viz_nodes", type=int, default=400,
                    help="Max nodes in interactive network viz (default: 400)")
    ap.add_argument("--skip_viz", action="store_true",
                    help="Skip visualization generation")
    args = ap.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.abspath(args.output_dir)

    data_out = os.path.join(output_dir, "data")
    network_out = os.path.join(output_dir, "network")
    stats_out = os.path.join(output_dir, "stats")
    viz_out = os.path.join(output_dir, "viz")

    # Create output dirs
    for d in [data_out, network_out, stats_out, viz_out]:
        os.makedirs(d, exist_ok=True)

    python = sys.executable

    # Step 1: Load & Identify
    cmd = [
        python, os.path.join(base_dir, "data", "load_and_identify.py"),
        "--input_dir", os.path.abspath(args.part1_dir),
        "--output_dir", data_out,
    ]
    if args.include_wos:
        cmd.append("--include_wos")
    run_step("Step 1: Load records & identify COI/industry ties", cmd)

    # Step 2: Build network
    run_step("Step 2: Build co-authorship network", [
        python, os.path.join(base_dir, "analysis", "network.py"),
        "--input_dir", data_out,
        "--output_dir", network_out,
        "--min_papers", str(args.min_papers),
    ])

    # Step 3: Statistical analysis
    run_step("Step 3: Statistical analysis", [
        python, os.path.join(base_dir, "analysis", "statistics.py"),
        "--input_dir", data_out,
        "--network_dir", network_out,
        "--output_dir", stats_out,
    ])

    if not args.skip_viz:
        # Step 4: Static & interactive plots
        run_step("Step 4: Generate visualizations", [
            python, os.path.join(base_dir, "viz", "plots.py"),
            "--data_dir", data_out,
            "--network_dir", network_out,
            "--stats_dir", stats_out,
            "--output_dir", viz_out,
        ])

        # Step 5: Interactive network HTML
        run_step("Step 5: Generate interactive network", [
            python, os.path.join(base_dir, "viz", "network_viz.py"),
            "--network_dir", network_out,
            "--output_dir", viz_out,
            "--max_nodes", str(args.max_viz_nodes),
        ])

    print(f"\n{'='*60}")
    print(f"  Pipeline complete!")
    print(f"{'='*60}")
    print(f"\nOutputs:")
    print(f"  Data:      {data_out}")
    print(f"  Network:   {network_out}")
    print(f"  Stats:     {stats_out}")
    print(f"  Viz:       {viz_out}")
    print(f"\nTo launch the interactive dashboard:")
    print(f"  cd {base_dir}")
    print(f"  streamlit run viz/app.py")
    print()


if __name__ == "__main__":
    main()
