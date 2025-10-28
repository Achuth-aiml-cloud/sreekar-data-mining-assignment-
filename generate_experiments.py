#!/usr/bin/env python3
"""
Generate experiment configurations for nanoGPT assignment
4 members, 32 experiments each = 128 total experiments
"""
import itertools
import json
import os

# Define the hyperparameter space
# Member 1: block_size=64, n_layer=4
# Member 2: block_size=64, n_layer=6
# Member 3: block_size=128, n_layer=4
# Member 4: block_size=128, n_layer=6

# Common parameters that vary for all members
n_heads = [4, 8]
n_embds = [128, 256]
batch_sizes = [8, 16]
max_iters_options = [25, 50]
dropouts = [0.1, 0.2]

# Base configuration
base_config = {
    "dataset": "shakespeare_char",
    "gradient_accumulation_steps": 1,
    "learning_rate": 1e-3,
    "decay_lr": True,
    "warmup_iters": 10,
    "min_lr": 1e-4,
    "eval_interval": 10,
    "eval_iters": 20,
    "log_interval": 1,
    "always_save_checkpoint": False,
    "device": "cpu",
    "compile": False,
}

# Member configurations
members = [
    {"name": "member1", "block_size": 64, "n_layer": 4},
    {"name": "member2", "block_size": 64, "n_layer": 6},
    {"name": "member3", "block_size": 128, "n_layer": 4},
    {"name": "member4", "block_size": 128, "n_layer": 6},
]

def generate_experiments():
    """Generate all experiment configurations"""

    all_experiments = []

    for member in members:
        member_name = member["name"]
        block_size = member["block_size"]
        n_layer = member["n_layer"]

        print(f"\nGenerating experiments for {member_name}...")
        print(f"  Fixed: block_size={block_size}, n_layer={n_layer}")

        # Generate all combinations
        combinations = list(itertools.product(
            n_heads, n_embds, batch_sizes, max_iters_options, dropouts
        ))

        print(f"  Total experiments: {len(combinations)}")

        for idx, (n_head, n_embd, batch_size, max_iters, dropout) in enumerate(combinations, 1):
            # Ensure n_embd is divisible by n_head
            if n_embd % n_head != 0:
                print(f"  Skipping incompatible: n_embd={n_embd}, n_head={n_head}")
                continue

            # Create experiment config
            exp_config = base_config.copy()
            exp_config.update({
                "block_size": block_size,
                "n_layer": n_layer,
                "n_head": n_head,
                "n_embd": n_embd,
                "batch_size": batch_size,
                "max_iters": max_iters,
                "dropout": dropout,
                "lr_decay_iters": max_iters,  # Match max_iters
            })

            # Create experiment name
            exp_name = f"exp_{idx:03d}_bs{block_size}_nl{n_layer}_nh{n_head}_ne{n_embd}_bsz{batch_size}_mi{max_iters}_dr{dropout}"

            # Set output directory
            exp_config["out_dir"] = f"experiments/{member_name}/results/{exp_name}"

            # Save config file
            config_dir = f"experiments/{member_name}/configs"
            os.makedirs(config_dir, exist_ok=True)

            config_path = f"{config_dir}/{exp_name}.py"
            with open(config_path, 'w') as f:
                f.write("# Experiment configuration\n")
                for key, value in exp_config.items():
                    if isinstance(value, str):
                        f.write(f'{key} = "{value}"\n')
                    else:
                        f.write(f'{key} = {value}\n')

            all_experiments.append({
                "member": member_name,
                "exp_name": exp_name,
                "config_path": config_path,
                "config": exp_config
            })

    return all_experiments

if __name__ == "__main__":
    experiments = generate_experiments()

    # Save experiment summary
    summary = {
        "total_experiments": len(experiments),
        "experiments_per_member": len(experiments) // 4,
        "experiments": experiments
    }

    with open("experiments/experiment_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Total experiments generated: {len(experiments)}")
    print(f"Experiments per member: {len(experiments) // 4}")
    print(f"Configuration files saved in experiments/memberX/configs/")
    print(f"Summary saved in experiments/experiment_summary.json")
    print(f"{'='*60}\n")
