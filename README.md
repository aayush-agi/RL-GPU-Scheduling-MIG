# RL-Based GPU Scheduling with MIG Partitioning

Improved implementation of reinforcement learning for GPU scheduling with NVIDIA Multi-Instance GPU (MIG) partitioning.

## 🎯 Key Results

| Metric | Original | Our Approach | Improvement |
|--------|----------|--------------|-------------|
| **Late Jobs** | ~87% | **37.7%** | ↓ 57% |
| **vs Best Heuristic** | Loses by 5% | **Wins by 11.7%** | ✅ |
| **Training Speed** | 1× | **10-50×** | NumPy optimization |

## 📊 Performance Comparison

| Method | Late Jobs (%) | Avg. Tardiness | Energy (MJ) |
|--------|---------------|----------------|-------------|
| **RL-PPO (Enhanced)** | **37.7 ± 5.7** | **0.91 ± 0.48** | 2.50 ± 0.04 |
| EFT | 43.1 ± 5.6 | 1.04 ± 0.63 | 2.48 ± 0.05 |
| Largest-First | 42.7 ± 5.7 | 1.02 ± 0.73 | 2.51 ± 0.05 |
| Smallest-First | 54.3 ± 4.7 | 1.21 ± 0.58 | 2.42 ± 0.05 |
| Random | 50.1 ± 4.7 | 1.13 ± 0.57 | 2.49 ± 0.04 |

## 🔑 Key Improvements Over Original

| Enhancement | Original | Improved | Impact |
|-------------|----------|----------|--------|
| **Environment** | Pandas | NumPy | 10-50× faster |
| **Deadlines** | 1.0-1.5× | 2.0-4.0× | Learnable problem |
| **Observation** | Basic | +slice sizes, +urgency | Better learning |
| **Rewards** | End-only | Immediate | Better credit assignment |
| **Network** | [256,256] | [256,256,128] | More capacity |
| **Training** | 200k steps | 500k steps | More learning |
| **LR** | Fixed | Annealing | Stability |

## 📁 Repository Structure

```
RL-GPU-scheduling/
├── README.md                           # This file
├── RL_GPU_Scheduling_Summary.pdf       # Technical report
├── RL_GPU_Scheduling_Summary.tex       # LaTeX source
├── RL_project_scheduling.ipynb         # Original implementation
├── colab/                              # Improved notebooks
│   ├── RL_GPU_Scheduling_PUBLICATION.ipynb   # ⭐ Best - Publication ready
│   ├── RL_GPU_Scheduling_ENHANCED.ipynb      # Enhanced version (37.7% late)
│   ├── RL_GPU_Scheduling_FINAL_COMPARISON.ipynb  # Full comparison
│   ├── RL_GPU_Scheduling_Fast.ipynb          # Speed-optimized
│   └── ...
└── docs/
    └── IPDPS_2026_paper.pdf            # Reference paper
```

## 🚀 Quick Start

### Recommended: Use Publication Notebook

1. Open `colab/RL_GPU_Scheduling_PUBLICATION.ipynb` in Google Colab
2. Set runtime to **GPU (A100 recommended)**
3. Run all cells (~60-90 min training time)
4. Results include graphs and LaTeX tables

### Alternative: Enhanced Notebook

For fastest results with all improvements:
- Use `colab/RL_GPU_Scheduling_ENHANCED.ipynb`

## 💡 Key Insights

### Why Original RL Failed to Beat Heuristics

The original implementation had **tight deadlines (1.0-1.5×)** which made the problem **greedy-optimal**:
- "Always pick largest slice" was the optimal strategy
- Simple heuristics naturally do this
- RL couldn't learn anything better

### What Made RL Win

1. **Relaxed deadlines (2-4×)** created room for optimization
2. **Slice sizes in observation** let RL learn which slices are faster
3. **Immediate rewards** provided better credit assignment
4. **More training (500k steps)** allowed policy convergence

## 📈 Approaches Tried

| Version | Changes | Late % | Status |
|---------|---------|--------|--------|
| Original | Pandas, tight deadlines | ~87% | Baseline |
| Fast | NumPy (10-50× faster) | ~87% | Speed only |
| Improved | +LR annealing, +deeper net | ~48% | Better |
| Relaxed | Deadlines 2-4× | ~48% | Same |
| **Enhanced** | **+Slice sizes, +immediate rewards** | **37.7%** | ✅ **Best** |

## 📄 Citation

If you use this code, please cite:

```bibtex
@techreport{rl_gpu_scheduling_improved,
  title={Reinforcement Learning for GPU Scheduling with MIG Partitioning: Improved Implementation},
  year={2024},
  note={Technical Report}
}
```

## 📋 Requirements

- Python 3.8+
- PyTorch
- stable-baselines3
- sb3-contrib
- gymnasium
- numpy
- matplotlib

Install via:
```bash
pip install stable-baselines3 sb3-contrib gymnasium matplotlib
```

## 🔬 Technical Details

### Environment
- **State space**: Job features + queue statistics + slice status + extras
- **Action space**: Discrete (select GPU slice)
- **Reward**: Weighted combination of tardiness and energy

### Training Configuration
- **Algorithm**: MaskablePPO (PPO with action masking)
- **Network**: [256, 256, 128] MLP
- **Timesteps**: 500,000
- **LR**: 3e-4 → 1e-5 (annealing)
- **Entropy**: 0.02 → 0.001 (decaying)

## 📝 License

MIT License

