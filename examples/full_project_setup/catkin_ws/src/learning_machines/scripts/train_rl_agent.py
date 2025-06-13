import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque

from robobo_interface import SimulationRobobo
from rl_environment import RoboboRLEnvironment


def setup_results_directory():
    """Setup results directory with multiple fallback options"""

    # try the Docker-mounted results directory (most reliable)
    docker_results = "/root/results"
    if os.path.exists("/root") and os.access("/root", os.W_OK):
        os.makedirs(docker_results, exist_ok=True)
        print(f"Using Docker results directory: {docker_results}")
        return docker_results

    #try relative to current working directory
    cwd_results = os.path.join(os.getcwd(), "results")
    try:
        os.makedirs(cwd_results, exist_ok=True)
        test_file = os.path.join(cwd_results, "test_write.txt")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        print(f"Using CWD results directory: {cwd_results}")
        return cwd_results
    except (OSError, PermissionError) as e:
        print(f"Cannot write to CWD results: {e}")

    #use current working directory as fallback
    fallback_dir = os.getcwd()
    print(f"Using fallback directory: {fallback_dir}")
    return fallback_dir

RESULTS_DIR = setup_results_directory()


def save_file_safely(filepath, save_function):
    """Safely save a file with error handling"""
    try:
        save_function()
        print(f"Saved: {filepath}")
        return True
    except Exception as e:
        print(f"Failed to save {filepath}: {e}")

        alt_path = os.path.join(os.getcwd(), os.path.basename(filepath))
        try:
            if filepath.endswith('.png'):
                plt.savefig(alt_path)
            elif filepath.endswith('.csv'):
                pass
            print(f"Saved to alternative location: {alt_path}")
            return True
        except Exception as e2:
            print(f"Alternative save also failed: {e2}")
            return False


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.fc(x)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

        self.memory = deque(maxlen=5000)
        self.gamma = 0.98984763
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9675
        self.batch_size = 64

    def act(self, state):
        if random.random() < self.epsilon:
            return random.choice([0, 1, 2])
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(self.model(state)).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target = rewards + self.gamma * next_q * (1 - dones)

        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())


def train(episodes=100, steps_per_episode=40):
    print(f"Starting training with results directory: {RESULTS_DIR}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Write access test: {os.access(RESULTS_DIR, os.W_OK)}")

    rob = SimulationRobobo()
    env = RoboboRLEnvironment(rob)
    state_size = 8
    action_size = 3

    agent = DQNAgent(state_size, action_size)

    reward_history = []
    epsilon_history = []

    patience = 20
    best_avg_reward = -float('inf')
    early_stop_counter = 0

    for e in range(episodes):
        rob.play_simulation()
        state = env.reset()
        total_reward = 0

        for step in range(steps_per_episode):
            action = agent.act(state)
            next_state, base_reward, done, _ = env.step(action)

            # shaped reward
            distance_reward = env.get_distance_delta() * 10
            obstacle_penalty = -10 if env.too_close_to_obstacle() else 0

            ir_front = max(state[:5])
            hit_obstacle = ir_front > 100
            collision_imminent = 80 < ir_front <= 100
            very_close = 70 < ir_front <= 80
            close = 40 < ir_front <= 70

            if hit_obstacle:
                rule_based_reward = -25
            elif collision_imminent:
                rule_based_reward = 5 if action in [1, 2] else -15
            elif very_close:
                rule_based_reward = 3 if action in [1, 2] else -8
            elif close:
                rule_based_reward = 1 if action in [1, 2] else -1
            else:
                rule_based_reward = 6 if action == 0 else (-3 if action in [1, 2] else -5)

            shaped_reward = base_reward + distance_reward + obstacle_penalty + rule_based_reward

            agent.remember(state, action, shaped_reward, next_state, done)
            agent.replay()
            state = next_state
            total_reward += shaped_reward

            if done:
                break

        agent.update_target_model()
        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)

        reward_history.append(total_reward)
        epsilon_history.append(agent.epsilon)
        print(f"Episode {e + 1}, reward: {total_reward:.2f}, epsilon: {agent.epsilon:.3f}")
        rob.stop_simulation()


        if e >= 10:
            avg_reward = np.mean(reward_history[-10:])
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                early_stop_counter = 0

                best_model_path = os.path.join(RESULTS_DIR, "best_model.pth")
                try:
                    torch.save(agent.model.state_dict(), best_model_path)
                    print(f"Saved best model to: {best_model_path}")
                except Exception as e:
                    print(f"Failed to save best model: {e}")
                    alt_path = os.path.join(os.getcwd(), "best_model.pth")
                    try:
                        torch.save(agent.model.state_dict(), alt_path)
                        print(f"Saved best model to alternative location: {alt_path}")
                    except Exception as e2:
                        print(f"Alternative model save failed: {e2}")
            else:
                early_stop_counter += 1

            if early_stop_counter >= patience:
                print("Early stopping triggered.")
                break

    final_model_path = os.path.join(RESULTS_DIR, "final_model.pth")
    try:
        torch.save(agent.model.state_dict(), final_model_path)
        print(f"Saved final model: {final_model_path}")
    except Exception as e:
        print(f"Failed to save final model: {e}")

    reward_plot_path = os.path.join(RESULTS_DIR, "reward_curve.png")
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(reward_history, label='Reward', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Training Progress: Reward over Episodes')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(reward_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved reward plot: {reward_plot_path}")
    except Exception as e:
        print(f"Failed to save reward plot: {e}")

    epsilon_plot_path = os.path.join(RESULTS_DIR, "epsilon_curve.png")
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(epsilon_history, label='Epsilon', color='orange', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Epsilon (Exploration Rate)')
        plt.title('Training Progress: Epsilon Decay')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(epsilon_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved epsilon plot: {epsilon_plot_path}")
    except Exception as e:
        print(f"Failed to save epsilon plot: {e}")

    csv_path = os.path.join(RESULTS_DIR, "training_log.csv")
    try:
        df = pd.DataFrame({
            "episode": list(range(1, len(reward_history) + 1)),
            "reward": reward_history,
            "epsilon": epsilon_history
        })
        df.to_csv(csv_path, index=False)
        print(f"Saved training log: {csv_path}")
    except Exception as e:
        print(f"Failed to save training log: {e}")
        try:
            alt_csv_path = "training_log.csv"
            df.to_csv(alt_csv_path, index=False)
            print(f"Saved training log to: {alt_csv_path}")
        except Exception as e2:
            print(f"Alternative CSV save failed: {e2}")

    print(f"\nTraining complete!")
    print(f"Total episodes: {len(reward_history)}")
    print(f"Best average reward: {best_avg_reward:.2f}")
    print(f"Results directory: {RESULTS_DIR}")

    try:
        files = os.listdir(RESULTS_DIR)
        print(f"Files saved: {files}")
    except Exception as e:
        print(f"Could not list results directory: {e}")


if __name__ == "__main__":
    train()
