# nanoGPT Assignment Part 2 - Complete Guide

## What Has Been Set Up

I've successfully set up **128 experiments** (32 per group member) for your nanoGPT assignment Part 2. The experiments are currently running in the background.

### Current Status
- **Total Experiments**: 128 (32 per member × 4 members)
- **Progress**: Running automatically
- **Time per experiment**: ~10-30 seconds (depending on max_iters: 25 or 50)
- **Total estimated time**: ~20-30 minutes for all 128 experiments

## Directory Structure

```
/workspaces/group-assignment/
├── nanoGPT/                              # nanoGPT repository
│   ├── model.py                          # GPT architecture
│   ├── train.py                          # Training script
│   └── sample.py                         # Text generation
│
├── experiments/                          # All experiment outputs
│   ├── member1/                          # Member 1: block_size=64, n_layer=4
│   │   ├── configs/                      # 32 config files (*.py)
│   │   ├── results/                      # Model checkpoints & outputs
│   │   ├── logs/                         # Training logs (*.log)
│   │   ├── loss_curves.png               # Visualization
│   │   └── final_results.json            # Summary JSON
│   │
│   ├── member2/                          # Member 2: block_size=64, n_layer=6
│   ├── member3/                          # Member 3: block_size=128, n_layer=4
│   ├── member4/                          # Member 4: block_size=128, n_layer=6
│   │
│   ├── experiment_summary.json           # Config for all experiments
│   ├── analysis_results.csv              # Parsed results table
│   ├── best_configurations.csv           # Top 10 configs
│   └── hyperparameter_comparison.png     # Comparison plots
│
├── generate_experiments.py               # Script to generate configs
├── run_all_experiments.py                # Run all 128 experiments
├── run_member_experiments.py             # Run one member's experiments
├── analyze_results.py                    # Analyze and summarize results
├── visualize_results.py                  # Generate plots
├── README_EXPERIMENTS.md                 # Detailed documentation
└── EXPERIMENT_GUIDE.md                   # This file
```

## Hyperparameter Space

Each member runs 32 experiments with different combinations:

### Fixed Parameters (Per Member)
- **Member 1**: `block_size=64`, `n_layer=4`
- **Member 2**: `block_size=64`, `n_layer=6`
- **Member 3**: `block_size=128`, `n_layer=4`
- **Member 4**: `block_size=128`, `n_layer=6`

### Variable Parameters (All Combinations)
- `n_head`: 4, 8
- `n_embd`: 128, 256
- `batch_size`: 8, 16
- `max_iters`: 25, 50 *(simplified from 1000/2000)*
- `dropout`: 0.1, 0.2

**Total**: 2 × 2 × 2 × 2 × 2 = **32 experiments per member**

## How to Monitor Progress

### 1. Check Overall Progress
```bash
python analyze_results.py
```

This shows:
- Number of completed experiments
- Best validation losses
- Training durations
- Summary statistics

### 2. Check Specific Experiment Logs
```bash
# View a specific experiment log
cat experiments/member1/logs/exp_001_bs64_nl4_nh4_ne128_bsz8_mi25_dr0.1.log

# Watch live progress
tail -f experiments/member1/logs/exp_001_*.log
```

### 3. Generate Visualizations
```bash
python visualize_results.py
```

This creates:
- Loss curves for each member
- Hyperparameter comparison plots
- Overfitting analysis
- Member comparison charts

## How to Run Experiments (If Needed)

### Run All Experiments (All 4 Members)
```bash
python run_all_experiments.py
```

### Run One Member's Experiments
```bash
python run_member_experiments.py 1  # Member 1
python run_member_experiments.py 2  # Member 2
python run_member_experiments.py 3  # Member 3
python run_member_experiments.py 4  # Member 4
```

## Understanding Results

### Experiment Naming Convention
Format: `exp_XXX_bs{block_size}_nl{n_layer}_nh{n_head}_ne{n_embd}_bsz{batch_size}_mi{max_iters}_dr{dropout}`

Example: `exp_001_bs64_nl4_nh4_ne128_bsz8_mi25_dr0.1`
- **exp_001**: Experiment number
- **bs64**: block_size = 64
- **nl4**: n_layer = 4
- **nh4**: n_head = 4
- **ne128**: n_embd = 128
- **bsz8**: batch_size = 8
- **mi25**: max_iters = 25
- **dr0.1**: dropout = 0.1

### Key Metrics in Results

1. **final_train_loss**: Training loss at last iteration
2. **final_val_loss**: Validation loss at last iteration
3. **val_train_gap**: Difference (overfitting indicator)
   - Positive gap = overfitting (val > train)
   - Negative gap = underfitting (train > val)
4. **num_parameters**: Model size in millions
5. **duration**: Training time in seconds

## Generating Text Samples

After experiments complete, generate text with trained models:

```bash
cd nanoGPT

# Generate from a specific model
python sample.py \
  --init_from=resume \
  --out_dir=../experiments/member1/results/exp_001_bs64_nl4_nh4_ne128_bsz8_mi25_dr0.1 \
  --num_samples=5 \
  --max_new_tokens=200

# Or with custom start text
python sample.py \
  --init_from=resume \
  --out_dir=../experiments/member1/results/exp_001_bs64_nl4_nh4_ne128_bsz8_mi25_dr0.1 \
  --start="ROMEO:" \
  --num_samples=3
```

## For Your Final Report (8-12 Pages)

### 1. Introduction (1 page)
- Brief overview of nanoGPT and transformer architecture
- Assignment objectives

### 2. Code Analysis (2-3 pages)
- ✓ Already completed in Part 1
- Answers to study questions about model.py, train.py, sample.py

### 3. Experimental Setup (1-2 pages)
Include:
- Dataset description (Shakespeare character-level)
- Hyperparameter ranges and rationale
- Training configuration
- Group member assignments

### 4. Results (2-3 pages)
Include:
- **Table**: Best configurations from `best_configurations.csv`
- **Plots**: Loss curves from `experiments/memberX/loss_curves.png`
- **Plots**: Hyperparameter comparison from `hyperparameter_comparison.png`
- **Table**: Summary statistics from `analysis_results.csv`
- **Text samples**: Generated text from best models

### 5. Analysis (2-3 pages)
Discuss:

#### Impact of Architecture Parameters:
- **block_size** (64 vs 128): Context window size
  - How does it affect long-range dependencies?
  - Memory vs performance trade-off

- **n_layer** (4 vs 6): Model depth
  - Deeper = more capacity but slower training
  - Risk of overfitting vs better representations

- **n_head** (4 vs 8): Attention heads
  - More heads = different attention patterns
  - Compatibility with n_embd (must divide evenly)

- **n_embd** (128 vs 256): Embedding dimension
  - Larger = more model capacity
  - Significantly more parameters

#### Impact of Training Parameters:
- **batch_size** (8 vs 16): Gradient estimation
  - Larger = more stable but less updates
  - Memory constraints

- **max_iters** (25 vs 50): Training duration
  - More iterations = better convergence
  - But risk of overfitting

- **dropout** (0.1 vs 0.2): Regularization
  - Higher dropout = less overfitting
  - But may underfit if too high

#### Overfitting Analysis:
- Look at `val_train_gap` in results
- Which configurations overfit most?
- How does dropout help?

#### Best Configurations:
- Which combination performed best?
- Why did it work well?
- Trade-offs (performance vs speed vs parameters)

### 6. Conclusion (1 page)
- Key insights and findings
- Recommendations for hyperparameter selection
- Lessons learned about transformers
- Future work

## Quick Analysis Commands

```bash
# 1. Check current progress
python analyze_results.py

# 2. Generate all visualizations
python visualize_results.py

# 3. View best configurations
cat experiments/best_configurations.csv

# 4. View full results table
cat experiments/analysis_results.csv

# 5. Check experiment status
ls -lh experiments/member1/logs/ | wc -l  # Count completed

# 6. Find best experiment
python analyze_results.py | grep "Best validation"
```

## Troubleshooting

### Check if experiments are still running
```bash
ps aux | grep "run_all_experiments"
```

### Check background process
```bash
# The experiments are running in background
# Check the log file
tail -f experiment_run.log
```

### If you need to restart experiments
```bash
# Kill running process (if needed)
pkill -f run_all_experiments

# Run again
python run_all_experiments.py
```

### If a specific experiment failed
Check the log file for errors:
```bash
cat experiments/member1/logs/exp_XXX_*.log
```

## Tips for Report Writing

1. **Use actual data**: Reference your CSV files and plots
2. **Compare systematically**: Group by hyperparameter
3. **Show trade-offs**: Performance vs speed vs memory
4. **Include examples**: Generated text samples
5. **Be quantitative**: Use actual loss values
6. **Explain why**: Not just what happened, but why
7. **Consider interactions**: How parameters work together

## Files to Include in GitHub

```bash
# Initialize git (if not already)
git init

# Add experiment files
git add experiments/
git add nanoGPT/
git add *.py
git add *.md

# Commit
git commit -m "Add nanoGPT experiments - Part 2"

# Push to your repository
git remote add origin <your-repo-url>
git push -u origin main
```

## Quick Reference: Key Files

- **Results Summary**: `experiments/analysis_results.csv`
- **Best Configs**: `experiments/best_configurations.csv`
- **Visualizations**: `experiments/*.png` and `experiments/member*/*.png`
- **Raw Logs**: `experiments/member*/logs/*.log`
- **Model Checkpoints**: `experiments/member*/results/exp_*/ckpt.pt`

## Getting Help

1. Check `README_EXPERIMENTS.md` for detailed documentation
2. Look at assignment PDF for requirements
3. Consult with group members on their assigned experiments
4. Review nanoGPT documentation: https://github.com/karpathy/nanoGPT

---

**Good luck with your assignment! The experiments should complete automatically.**
**Once done, run `python analyze_results.py` and `python visualize_results.py` to see all results.**
