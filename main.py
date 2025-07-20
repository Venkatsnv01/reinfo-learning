import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from env import ContinuousMazeEnv
from dqn_model import Qneuralnet
from utils import ReplayBuffer, train

# saving model folder
os.makedirs("models", exist_ok=True)
os.makedirs("experiments_graphs", exist_ok=True)

# toggle render, train and test (submitting with test and render on)
train_dqn = False
render = True
test_dqn = True

# Device config
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    # Check available memory
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    device = torch.device("cpu")
    print("CUDA not available, using CPU")

# Hyperparameters
dim_states = 2
dim_actions = 4
learning_rate = 0.001
gamma = 0.99  
buffer_limit = 50_000
batch_size = 64  
num_episodes = 10_000
max_steps = 200

# Epsilon decay parameters
epsilon_start = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995

# Create models 
q_net = Qneuralnet(dimension_action=dim_actions, dimension_states=dim_states).to(device)
q_target = Qneuralnet(dimension_action=dim_actions, dimension_states=dim_states).to(device)
q_target.load_state_dict(q_net.state_dict())
print("Models created successfully")

if train_dqn:
    env = ContinuousMazeEnv(render_mode="human" if render else None)

    memory = ReplayBuffer(buffer_limit=buffer_limit)
    optimizer = optim.Adam(q_net.parameters(), lr=learning_rate)
    print("Memory buffer and optimizer created successfully")

    print_interval = 20
    episode_reward = 0.0
    rewards = []
    consecutive_successes = 0
    epsilon = epsilon_start

    for n_episode in range(num_episodes):
        s, _ = env.reset()
        state = s
        done = False
        episode_reward = 0.0
        reached_goal = False

        for _ in range(max_steps):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            action = q_net.sample_action(state_tensor, epsilon)
            s_prime, reward, done, _, _ = env.step(action)

            # Check if goal was reached
            if reward == 100.0:
                reached_goal = True

            done_mask = 0.0 if done else 1.0
            memory.put((state, action, reward, s_prime, done_mask))
            state = s_prime
            episode_reward += reward

            if done:
                break

        if memory.size() > 2000:
            train(q_net, q_target, memory, optimizer, batch_size, gamma, device)  

        # Update consecutive successes
        if reached_goal:
            consecutive_successes += 1
        else:
            consecutive_successes = 0

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if n_episode % print_interval == 0 and n_episode != 0:
            q_target.load_state_dict(q_net.state_dict())
            print(f"Episode {n_episode}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.3f}, Buffer size: {memory.size()}, Consecutive successes: {consecutive_successes}")

        rewards.append(episode_reward)

        # Check for 100 consecutive successful episodes (reaching goal)
        if consecutive_successes >= 100 and epsilon <= epsilon_min:
            print(f"Training completed! Agent reached goal for 100 consecutive episodes with epsilon = {epsilon:.3f}")
            break

    env.close()
    model_path = os.path.join(os.path.dirname(__file__), "models", "maze_dqn.pth")
    torch.save(q_net.state_dict(), model_path)
    print("Model saved successfully")

    plt.figure(figsize=(12, 8))
    
    # Plot rewards
    plt.subplot(2, 1, 1)
    plt.plot(rewards, label="Reward per Episode", alpha=0.6)
    plt.xlabel("Episode")
    plt.ylabel("Rewards")
    plt.grid(True)
    plt.legend()
    plt.title("Training Progress")
    
    # Plot moving average
    window_size = 100
    if len(rewards) >= window_size:
        moving_avg = []
        for i in range(window_size, len(rewards)):
            moving_avg.append(sum(rewards[i-window_size:i]) / window_size)
        plt.plot(range(window_size, len(rewards)), moving_avg, label=f"Moving Average ({window_size} episodes)", linewidth=2)
        plt.legend()
    
    training_curve_path = os.path.join(os.path.dirname(__file__), "experiments_graphs", "training_curve.png")
    plt.savefig(training_curve_path, dpi=300, bbox_inches='tight')
    print("Training curve saved successfully")
    plt.show()

# Test phase
if test_dqn:
    print("Testing trained model:")
    env = ContinuousMazeEnv(render_mode="human" if render else None)
    

    model_path = os.path.join(os.path.dirname(__file__), "models", "maze_dqn.pth")
    q_net.load_state_dict(torch.load(model_path, map_location=device))
    q_net = q_net.to(device)
    print("Test model loaded successfully")

    for test_episode in range(10):
        state, _ = env.reset()
        episode_reward = 0
        reached_goal = False

        for step in range(max_steps):
            env.render()
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            action = q_net.sample_action(state_tensor, epsilon=0.0)
            s_prime, reward, done, _, _ = env.step(action)
            state = s_prime
            episode_reward += reward
            
            if reward == 100.0:
                reached_goal = True

            if done:
                break

        print(f"Test Episode {test_episode + 1}: Reward = {episode_reward:.2f}, Reached Goal = {reached_goal}")
    env.close()
