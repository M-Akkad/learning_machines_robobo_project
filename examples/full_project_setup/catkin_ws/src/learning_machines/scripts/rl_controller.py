# import numpy as np
# import random

# class RLController:
#     def __init__(self, num_states=3 ** 3, num_actions=4, alpha=0.1, gamma=0.9, epsilon=0.1):
#         self.num_states = num_states
#         self.num_actions = num_actions
#         self.alpha = alpha  # learning rate
#         self.gamma = gamma  # discount factor
#         self.epsilon = epsilon  # exploration rate

#         # Initialize Q-table
#         self.q_table = np.zeros((num_states, num_actions))

#     def discretize_state(self, ir_values):
#         # Simple discretization: for 3 front sensors (FL, C, FR)
#         # 0 = far, 1 = medium, 2 = near
#         def bin_ir(v):
#             if v < 15:
#                 return 0
#             elif v < 50:
#                 return 1
#             else:
#                 return 2

#         bins = [bin_ir(ir_values[0]), bin_ir(ir_values[2]), bin_ir(ir_values[4])]
#         # Map to unique state index
#         state_idx = bins[0] * 9 + bins[1] * 3 + bins[2]
#         return state_idx

#     def select_action(self, state_idx):
#         # Epsilon-greedy
#         if random.uniform(0, 1) < self.epsilon:
#             return random.randint(0, self.num_actions - 1)
#         else:
#             return np.argmax(self.q_table[state_idx])

#     def update(self, state_idx, action, reward, next_state_idx, done):
#         max_q_next = np.max(self.q_table[next_state_idx])
#         td_target = reward + self.gamma * max_q_next * (not done)
#         td_error = td_target - self.q_table[state_idx, action]

#         self.q_table[state_idx, action] += self.alpha * td_error


import random
import numpy as np

class RLController:
    def __init__(self, num_states=16, num_actions=4, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.q_table = np.zeros((num_states, num_actions))

    def discretize_state(self, ir_values):
        """
        Better discretization:
        Use simple binary encoding:
        - front blocked? (C sensor)
        - left blocked? (FL, L sensors)
        - right blocked? (FR, R sensors)
        → 3 bits → 8 possible states (but we leave num_states customizable)

        We use threshold 50 as example → you can tune it.
        """
        threshold = 50

        front = 1 if ir_values[2] is not None and ir_values[2] > threshold else 0
        left = 1 if (ir_values[0] is not None and ir_values[0] > threshold) or (ir_values[1] is not None and ir_values[1] > threshold) else 0
        right = 1 if (ir_values[3] is not None and ir_values[3] > threshold) or (ir_values[4] is not None and ir_values[4] > threshold) else 0

        # Encode to state index
        state_idx = front * 4 + left * 2 + right * 1
        return state_idx

    def select_action(self, state_idx):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.num_actions - 1)  # Explore
        else:
            return np.argmax(self.q_table[state_idx])  # Exploit

    def update(self, state_idx, action, reward, next_state_idx, done):
        best_next_action = np.argmax(self.q_table[next_state_idx])
        target = reward + (0 if done else self.gamma * self.q_table[next_state_idx, best_next_action])
        self.q_table[state_idx, action] += self.alpha * (target - self.q_table[state_idx, action])

    def save_q_table(self, path):
        np.save(path, self.q_table)

    def load_q_table(self, path):
        self.q_table = np.load(path)

