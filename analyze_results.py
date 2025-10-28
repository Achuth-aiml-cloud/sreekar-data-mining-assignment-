#!/usr/bin/env python3
"""
Analyze and visualize experiment results
"""
import json
import os
import re
from pathlib import Path
import pandas as pd

def parse_log_file(log_file):
    """Extract metrics from a log file"""
    with open(log_file, 'r') as f:
        content = f.read()

    metrics = {
        'train_losses': [],
        'val_losses': [],
        'iter_times': [],
        'final_train_loss': None,
        'final_val_loss': None,
        'duration': None,
        'num_parameters': None
    }

    # Extract number of parameters
    param_match = re.search(r'number of parameters: ([\d.]+)M', content)
    if param_match:
        metrics['num_parameters'] = float(param_match.group(1))

    # Extract step losses (validation)
    step_matches = re.findall(r'step \d+: train loss ([\d.]+), val loss ([\d.]+)', content)
    for train_loss, val_loss in step_matches:
        metrics['train_losses'].append(float(train_loss))
        metrics['val_losses'].append(float(val_loss))

    # Get final losses
    if metrics['train_losses']:
        metrics['final_train_loss'] = metrics['train_losses'][-1]
        metrics['final_val_loss'] = metrics['val_losses'][-1]

    # Extract duration
    duration_match = re.search(r'Duration: ([\d.]+) seconds', content)
    if duration_match:
        metrics['duration'] = float(duration_match.group(1))

    return metrics

def parse_config_name(exp_name):
    """Parse experiment name to extract hyperparameters"""
    # exp_001_bs64_nl4_nh4_ne128_bsz8_mi25_dr0.1
    pattern = r'exp_(\d+)_bs(\d+)_nl(\d+)_nh(\d+)_ne(\d+)_bsz(\d+)_mi(\d+)_dr([\d.]+)'
    match = re.match(pattern, exp_name)

    if match:
        return {
            'exp_num': int(match.group(1)),
            'block_size': int(match.group(2)),
            'n_layer': int(match.group(3)),
            'n_head': int(match.group(4)),
            'n_embd': int(match.group(5)),
            'batch_size': int(match.group(6)),
            'max_iters': int(match.group(7)),
            'dropout': float(match.group(8))
        }
    return {}

def analyze_experiments():
    """Analyze all experiment results"""

    results = []

    for member in ['member1', 'member2', 'member3', 'member4']:
        log_dir = f'experiments/{member}/logs'

        if not os.path.exists(log_dir):
            continue

        for log_file in sorted(Path(log_dir).glob('*.log')):
            exp_name = log_file.stem

            # Parse config from name
            config = parse_config_name(exp_name)

            if not config:
                continue

            # Parse log file
            metrics = parse_log_file(log_file)

            # Combine
            result = {
                'member': member,
                'exp_name': exp_name,
                **config,
                **metrics
            }

            results.append(result)

    return results

def create_summary_table(results):
    """Create a summary table of all experiments"""

    df = pd.DataFrame(results)

    if df.empty:
        print("No results found yet!")
        return df

    # Select key columns
    columns = [
        'member', 'exp_name', 'exp_num', 'block_size', 'n_layer', 'n_head', 'n_embd',
        'batch_size', 'max_iters', 'dropout', 'num_parameters',
        'final_train_loss', 'final_val_loss', 'duration'
    ]

    summary_df = df[columns].copy()

    # Calculate overfitting gap
    summary_df['val_train_gap'] = summary_df['final_val_loss'] - summary_df['final_train_loss']

    return summary_df

def print_statistics(df):
    """Print summary statistics"""

    if df.empty:
        return

    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY STATISTICS")
    print("="*80)

    print(f"\nTotal experiments completed: {len(df)}")
    print(f"Total experiments expected: 128")
    print(f"Progress: {len(df)/128*100:.1f}%")

    print("\n" + "-"*80)
    print("BY MEMBER:")
    print("-"*80)
    for member in df['member'].unique():
        member_df = df[df['member'] == member]
        print(f"{member}: {len(member_df)}/32 experiments")

    print("\n" + "-"*80)
    print("LOSS STATISTICS:")
    print("-"*80)
    print(f"Best validation loss: {df['final_val_loss'].min():.4f}")
    best_idx = df['final_val_loss'].idxmin()
    best_exp = df.loc[best_idx]
    print(f"  Experiment: {best_exp['exp_name']}")
    print(f"  Config: bs={best_exp['block_size']}, nl={best_exp['n_layer']}, "
          f"nh={best_exp['n_head']}, ne={best_exp['n_embd']}, "
          f"bsz={best_exp['batch_size']}, mi={best_exp['max_iters']}, dr={best_exp['dropout']}")

    print("\n" + "-"*80)
    print("DURATION STATISTICS:")
    print("-"*80)
    print(f"Average duration: {df['duration'].mean():.2f} seconds")
    print(f"Total time so far: {df['duration'].sum():.2f} seconds ({df['duration'].sum()/60:.2f} minutes)")

    print("\n" + "-"*80)
    print("PARAMETER STATISTICS:")
    print("-"*80)
    print(f"Parameter range: {df['num_parameters'].min():.2f}M - {df['num_parameters'].max():.2f}M")

    print("\n" + "="*80)

def save_results(df):
    """Save results to CSV"""

    if df.empty:
        return

    output_file = 'experiments/analysis_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Save best configurations
    best_configs = df.nsmallest(10, 'final_val_loss')[
        ['exp_name', 'member', 'block_size', 'n_layer', 'n_head', 'n_embd',
         'batch_size', 'max_iters', 'dropout', 'final_val_loss', 'val_train_gap']
    ]

    best_file = 'experiments/best_configurations.csv'
    best_configs.to_csv(best_file, index=False)
    print(f"Best configurations saved to: {best_file}")

def main():
    """Main analysis function"""

    print("\n" + "#"*80)
    print("# nanoGPT EXPERIMENT RESULTS ANALYSIS")
    print("#"*80)

    # Analyze experiments
    results = analyze_experiments()

    if not results:
        print("\nNo experiments completed yet. Please wait for experiments to finish.")
        return

    # Create summary table
    df = create_summary_table(results)

    # Print statistics
    print_statistics(df)

    # Save results
    save_results(df)

    # Print sample of results
    print("\n" + "-"*80)
    print("SAMPLE RESULTS (Top 10 by validation loss):")
    print("-"*80)
    print(df.nsmallest(10, 'final_val_loss')[
        ['member', 'exp_name', 'final_val_loss', 'val_train_gap', 'duration']
    ].to_string(index=False))

    print("\n" + "#"*80)
    print("# ANALYSIS COMPLETE")
    print("#"*80 + "\n")

if __name__ == "__main__":
    main()
