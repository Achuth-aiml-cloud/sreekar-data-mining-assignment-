#!/usr/bin/env python3
"""
Visualize experiment results with plots
"""
import json
import os
import re
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

def parse_log_file_detailed(log_file):
    """Extract all iteration data from a log file"""
    with open(log_file, 'r') as f:
        content = f.read()

    data = {
        'iterations': [],
        'train_losses': [],
        'val_losses': [],
    }

    # Extract iter losses
    iter_matches = re.findall(r'iter (\d+): loss ([\d.]+)', content)
    for iter_num, loss in iter_matches:
        data['iterations'].append(int(iter_num))
        data['train_losses'].append(float(loss))

    # Extract step losses (validation)
    step_matches = re.findall(r'step (\d+): train loss ([\d.]+), val loss ([\d.]+)', content)
    val_data = {}
    for step, train_loss, val_loss in step_matches:
        val_data[int(step)] = {
            'train': float(train_loss),
            'val': float(val_loss)
        }

    data['val_data'] = val_data

    return data

def plot_loss_curves(member='member1', n_plots=6):
    """Plot loss curves for best experiments"""

    log_dir = f'experiments/{member}/logs'
    if not os.path.exists(log_dir):
        print(f"No logs found for {member}")
        return

    # Get all log files
    log_files = sorted(Path(log_dir).glob('*.log'))[:n_plots]

    if not log_files:
        print(f"No experiments found for {member}")
        return

    # Create plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Training Loss Curves - {member.upper()}', fontsize=16)

    for idx, log_file in enumerate(log_files):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        exp_name = log_file.stem

        # Parse log file
        data = parse_log_file_detailed(log_file)

        if not data['iterations']:
            continue

        # Plot training loss
        ax.plot(data['iterations'], data['train_losses'],
                label='Train Loss', alpha=0.7, linewidth=1)

        # Plot validation loss
        if data['val_data']:
            val_iters = sorted(data['val_data'].keys())
            val_losses = [data['val_data'][i]['val'] for i in val_iters]
            ax.scatter(val_iters, val_losses, color='red',
                      label='Val Loss', s=50, zorder=5)

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title(exp_name.replace(f'{member}_', ''), fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = f'experiments/{member}/loss_curves.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Loss curves saved to: {output_file}")
    plt.close()

def plot_hyperparameter_comparison():
    """Plot comparison of hyperparameters"""

    # Load results
    csv_file = 'experiments/analysis_results.csv'
    if not os.path.exists(csv_file):
        print("No analysis results found. Run analyze_results.py first.")
        return

    df = pd.read_csv(csv_file)

    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Hyperparameter Impact on Validation Loss', fontsize=16)

    # 1. Block size
    ax = axes[0, 0]
    sns.boxplot(data=df, x='block_size', y='final_val_loss', ax=ax)
    ax.set_title('Block Size vs Val Loss')
    ax.set_xlabel('Block Size')
    ax.set_ylabel('Validation Loss')

    # 2. Number of layers
    ax = axes[0, 1]
    sns.boxplot(data=df, x='n_layer', y='final_val_loss', ax=ax)
    ax.set_title('Number of Layers vs Val Loss')
    ax.set_xlabel('Number of Layers')
    ax.set_ylabel('Validation Loss')

    # 3. Number of heads
    ax = axes[0, 2]
    sns.boxplot(data=df, x='n_head', y='final_val_loss', ax=ax)
    ax.set_title('Number of Heads vs Val Loss')
    ax.set_xlabel('Number of Heads')
    ax.set_ylabel('Validation Loss')

    # 4. Embedding dimension
    ax = axes[1, 0]
    sns.boxplot(data=df, x='n_embd', y='final_val_loss', ax=ax)
    ax.set_title('Embedding Dimension vs Val Loss')
    ax.set_xlabel('Embedding Dimension')
    ax.set_ylabel('Validation Loss')

    # 5. Batch size
    ax = axes[1, 1]
    sns.boxplot(data=df, x='batch_size', y='final_val_loss', ax=ax)
    ax.set_title('Batch Size vs Val Loss')
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Validation Loss')

    # 6. Dropout
    ax = axes[1, 2]
    sns.boxplot(data=df, x='dropout', y='final_val_loss', ax=ax)
    ax.set_title('Dropout vs Val Loss')
    ax.set_xlabel('Dropout Rate')
    ax.set_ylabel('Validation Loss')

    plt.tight_layout()

    output_file = 'experiments/hyperparameter_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Hyperparameter comparison saved to: {output_file}")
    plt.close()

def plot_overfitting_analysis():
    """Plot overfitting analysis"""

    # Load results
    csv_file = 'experiments/analysis_results.csv'
    if not os.path.exists(csv_file):
        print("No analysis results found. Run analyze_results.py first.")
        return

    df = pd.read_csv(csv_file)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Overfitting Analysis', fontsize=16)

    # 1. Train vs Val loss scatter
    ax = axes[0]
    scatter = ax.scatter(df['final_train_loss'], df['final_val_loss'],
                        c=df['val_train_gap'], cmap='coolwarm',
                        s=100, alpha=0.6)
    ax.plot([df['final_train_loss'].min(), df['final_train_loss'].max()],
            [df['final_train_loss'].min(), df['final_train_loss'].max()],
            'k--', alpha=0.3, label='Perfect fit')
    ax.set_xlabel('Final Training Loss')
    ax.set_ylabel('Final Validation Loss')
    ax.set_title('Training vs Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Val-Train Gap')

    # 2. Overfitting gap by dropout
    ax = axes[1]
    sns.boxplot(data=df, x='dropout', y='val_train_gap', ax=ax)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_title('Overfitting Gap by Dropout Rate')
    ax.set_xlabel('Dropout Rate')
    ax.set_ylabel('Validation - Training Loss')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = 'experiments/overfitting_analysis.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Overfitting analysis saved to: {output_file}")
    plt.close()

def plot_member_comparison():
    """Compare performance across members"""

    # Load results
    csv_file = 'experiments/analysis_results.csv'
    if not os.path.exists(csv_file):
        print("No analysis results found. Run analyze_results.py first.")
        return

    df = pd.read_csv(csv_file)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Performance Comparison Across Group Members', fontsize=16)

    # 1. Val loss by member
    ax = axes[0]
    sns.boxplot(data=df, x='member', y='final_val_loss', ax=ax)
    ax.set_title('Validation Loss Distribution by Member')
    ax.set_xlabel('Group Member')
    ax.set_ylabel('Final Validation Loss')
    ax.grid(True, alpha=0.3)

    # 2. Duration by member
    ax = axes[1]
    sns.boxplot(data=df, x='member', y='duration', ax=ax)
    ax.set_title('Training Duration by Member')
    ax.set_xlabel('Group Member')
    ax.set_ylabel('Duration (seconds)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = 'experiments/member_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Member comparison saved to: {output_file}")
    plt.close()

def main():
    """Main visualization function"""

    print("\n" + "#"*80)
    print("# nanoGPT EXPERIMENT VISUALIZATION")
    print("#"*80 + "\n")

    # Plot loss curves for each member
    for member in ['member1', 'member2', 'member3', 'member4']:
        print(f"Generating loss curves for {member}...")
        plot_loss_curves(member, n_plots=6)

    # Plot comparisons (if enough data)
    csv_file = 'experiments/analysis_results.csv'
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)

        if len(df) >= 10:
            print("\nGenerating hyperparameter comparison plots...")
            plot_hyperparameter_comparison()

            print("Generating overfitting analysis...")
            plot_overfitting_analysis()

            if len(df['member'].unique()) > 1:
                print("Generating member comparison...")
                plot_member_comparison()
        else:
            print(f"\nNeed at least 10 experiments for comparison plots (found {len(df)})")

    print("\n" + "#"*80)
    print("# VISUALIZATION COMPLETE")
    print("#"*80 + "\n")

if __name__ == "__main__":
    main()
