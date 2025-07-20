# reinfo-learning

# Deep Q-Network (DQN) for Continuous Maze Navigation

This project implements a Deep Q-Network (DQN) to solve a continuous 2D maze navigation problem with complex reward structures and obstacles. Through a series of experiments—ranging from step-limit tuning to epsilon decay strategies and architectural improvements—the agent successfully learns both single and multi-path solutions to reach the goal, avoiding danger zones and optimizing rewards.


## Environment
- **State Space**: 2D continuous position (x, y) 
- **Action Space**: 4 discrete actions (up, down, left, right)
- **Goal**: Navigate from start to goal 
- **Obstacles**: Walls and danger zones (red rectangles)
- **Rewards**: 
  - Goal reached: +100
  - Danger zone: -10 (episode ends)
  - Wall collision: -1
  - Distance-based shaping: reward for moving closer to goal
  - Step penalty: -0.01 per step


## Multi-Path Learning with Improved Architecture
- **max_steps**: 200
- **epsilon**: Exponential decay from 1.0 to 0.1
- **Network**: 4-layer architecture (256→256→128→4) with dropout
- **Learning rate**: 0.001 with gradient clipping
- **Key Achievement**: Agent learned multiple optimal paths to the goal
- **Improvement**: Enhanced exploration-exploitation balance with proper consecutive success tracking
