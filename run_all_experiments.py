#!/usr/bin/env python3
"""
Run all experiments for all group members
"""
import os
import json
import subprocess
import time
from pathlib import Path

def run_experiment(member_name, exp_name, config_path):
    """Run a single experiment"""
    print(f"\n{'='*60}")
    print(f"Running: {member_name} - {exp_name}")
    print(f"{'='*60}")

    # Create log directory
    log_dir = f"experiments/{member_name}/logs"
    os.makedirs(log_dir, exist_ok=True)

    log_file = f"{log_dir}/{exp_name}.log"

    # Run training
    cmd = f"cd nanoGPT && python train.py config/train_shakespeare_char.py ../{config_path}"

    start_time = time.time()

    try:
        # Run command and capture output
        with open(log_file, 'w') as f:
            f.write(f"Experiment: {exp_name}\n")
            f.write(f"Config: {config_path}\n")
            f.write(f"Started: {time.ctime(start_time)}\n")
            f.write("="*60 + "\n\n")

            result = subprocess.run(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            f.write(result.stdout)

            end_time = time.time()
            duration = end_time - start_time

            f.write(f"\n{'='*60}\n")
            f.write(f"Completed: {time.ctime(end_time)}\n")
            f.write(f"Duration: {duration:.2f} seconds\n")

        print(f"✓ Completed in {duration:.2f} seconds")
        print(f"  Log saved to: {log_file}")

        return {
            "status": "success",
            "duration": duration,
            "log_file": log_file
        }

    except Exception as e:
        print(f"✗ Failed: {str(e)}")
        return {
            "status": "failed",
            "error": str(e),
            "log_file": log_file
        }

def main():
    """Run all experiments"""

    # Load experiment summary
    with open("experiments/experiment_summary.json", 'r') as f:
        summary = json.load(f)

    experiments = summary["experiments"]
    total = len(experiments)

    print(f"\n{'#'*60}")
    print(f"# nanoGPT Experiments - Group Assignment")
    print(f"# Total experiments: {total}")
    print(f"# Experiments per member: {total // 4}")
    print(f"{'#'*60}\n")

    results = []
    start_time = time.time()

    for idx, exp in enumerate(experiments, 1):
        member = exp["member"]
        exp_name = exp["exp_name"]
        config_path = exp["config_path"]

        print(f"\nProgress: {idx}/{total}")

        result = run_experiment(member, exp_name, config_path)
        result.update({
            "member": member,
            "exp_name": exp_name,
            "config_path": config_path
        })
        results.append(result)

        # Save intermediate results
        with open("experiments/experiment_results.json", 'w') as f:
            json.dump(results, f, indent=2)

    end_time = time.time()
    total_duration = end_time - start_time

    # Save final results
    final_summary = {
        "total_experiments": total,
        "start_time": time.ctime(start_time),
        "end_time": time.ctime(end_time),
        "total_duration_seconds": total_duration,
        "total_duration_hours": total_duration / 3600,
        "results": results
    }

    with open("experiments/final_results.json", 'w') as f:
        json.dump(final_summary, f, indent=2)

    # Print summary
    print(f"\n{'#'*60}")
    print(f"# ALL EXPERIMENTS COMPLETED")
    print(f"# Total time: {total_duration/60:.2f} minutes")
    print(f"# Results saved to: experiments/final_results.json")
    print(f"{'#'*60}\n")

    # Count successes and failures
    successes = sum(1 for r in results if r["status"] == "success")
    failures = total - successes

    print(f"Success: {successes}/{total}")
    print(f"Failed: {failures}/{total}")

if __name__ == "__main__":
    main()
