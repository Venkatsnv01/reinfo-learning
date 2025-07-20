## Experiments

### Experiment 1: High Step Limit
- **max_steps**: 10,000
- **Result**: Training curve saved as `1_training_curve.png`
- **Issue**: Agent had too much time to explore inefficiently, and model was not saved

### Experiment 2: Reduced Step Limit + Epsilon Decay
- **max_steps**: 200
- **epsilon**: `max(0.05, 0.3 - 0.01 * (n_episode / 1000))`
- **Result**: Training curve saved as `2_training_curve.png`
- **Improvement**: Better exploration-exploitation balance

### Experiment 3: Optimal Configuration
- **Same parameters as Experiment 2**
- **Result**: Training curve saved as `3_training_curve.png`
- **Outcome**: Agent found optimal path (longer route avoiding obstacles)

### Experiment 4: Final in readme.md
