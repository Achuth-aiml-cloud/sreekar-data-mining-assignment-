import os
import sys
import json
import subprocess
import time

def run_experiment(member_name, exp_name, config_path):
    """Run a single experiment"""
    print(f"\n{'='*60}")
    print(f"Running: {exp_name}")
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
    """Run experiments for specified member"""

    if len(sys.argv) != 2:
        print("Usage: python run_member_experiments.py <member_number>")
        print("Example: python run_member_experiments.py 1")
        sys.exit(1)

    member_num = sys.argv[1]
    member_name = f"member{member_num}"

    # Check if member exists
    if not os.path.exists(f"experiments/{member_name}"):
        print(f"Error: {member_name} not found!")
        sys.exit(1)

    # Load experiment summary
    with open("experiments/experiment_summary.json", 'r') as f:
        summary = json.load(f)

    # Filter experiments for this member
    member_experiments = [e for e in summary["experiments"] if e["member"] == member_name]

    if not member_experiments:
        print(f"No experiments found for {member_name}")
        sys.exit(1)

    total = len(member_experiments)

    print(f"\n{'#'*60}")
    print(f"# nanoGPT Experiments - {member_name.upper()}")
    print(f"# Total experiments: {total}")
    print(f"{'#'*60}\n")

    results = []
    start_time = time.time()

    for idx, exp in enumerate(member_experiments, 1):
        exp_name = exp["exp_name"]
        config_path = exp["config_path"]

        print(f"\nProgress: {idx}/{total}")

        result = run_experiment(member_name, exp_name, config_path)
        result.update({
            "member": member_name,
            "exp_name": exp_name,
            "config_path": config_path
        })
        results.append(result)

        # Save intermediate results
        with open(f"experiments/{member_name}/results_summary.json", 'w') as f:
            json.dump(results, f, indent=2)

    end_time = time.time()
    total_duration = end_time - start_time

    # Save final results
    final_summary = {
        "member": member_name,
        "total_experiments": total,
        "start_time": time.ctime(start_time),
        "end_time": time.ctime(end_time),
        "total_duration_seconds": total_duration,
        "total_duration_minutes": total_duration / 60,
        "results": results
    }

    with open(f"experiments/{member_name}/final_results.json", 'w') as f:
        json.dump(final_summary, f, indent=2)

    # Print summary
    print(f"\n{'#'*60}")
    print(f"# {member_name.upper()} EXPERIMENTS COMPLETED")
    print(f"# Total time: {total_duration/60:.2f} minutes")
    print(f"# Results saved to: experiments/{member_name}/final_results.json")
    print(f"{'#'*60}\n")

    # Count successes and failures
    successes = sum(1 for r in results if r["status"] == "success")
    failures = total - successes

    print(f"Success: {successes}/{total}")
    print(f"Failed: {failures}/{total}")

if __name__ == "__main__":
    main()
