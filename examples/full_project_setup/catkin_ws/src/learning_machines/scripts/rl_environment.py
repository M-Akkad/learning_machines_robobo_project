import math
from collections import deque
from robobo_interface.datatypes import Position

class RoboboRLEnvironment:
    def __init__(self, robobo):
        self.rob = robobo
        self.last_position = None
        self.recent_positions = deque(maxlen=10)
        self.recent_actions = deque(maxlen=5)
        self.ir_threshold = 100.0

    def reset(self):
        self.rob.play_simulation()
        self.rob.reset_wheels()
        self.last_position = self.rob.get_position()
        self.recent_positions.clear()
        self.recent_actions.clear()
        self.recent_positions.append(self.last_position)
        return self.get_state()

    def get_state(self):
        ir_values = self.rob.read_irs()
        return [val if val is not None else 0.0 for val in ir_values]

    def step(self, action):
        if action == 0:
            self.rob.move_blocking(50, 50, 500)  # forward
        elif action == 1:
            self.rob.move_blocking(50, -50, 300)  # turn right
        elif action == 2:
            self.rob.move_blocking(-50, 50, 300)  # turn left

        ir_values = self.get_state()
        current_position = self.rob.get_position()

        self.recent_actions.append(action)

        base_reward = self.compute_base_reward(current_position, ir_values, action)
        done = any(ir > 120 for ir in ir_values[:5])

        self.recent_positions.append(current_position)
        self.last_position = current_position

        return ir_values, base_reward, done, {}

    def compute_base_reward(self, position, ir_values, action):
        distance = self.euclidean(self.last_position, position)
        reward = 0.5 * distance

        if self.too_close_to_obstacle(ir_values):
            min_ir = min(ir_values[:5])
            reward -= (min_ir - self.ir_threshold) * 0.1

        if distance < 0.01:
            reward -= 2

        if len(self.recent_positions) == self.recent_positions.maxlen:
            avg_x = sum(p.x for p in self.recent_positions) / len(self.recent_positions)
            avg_y = sum(p.y for p in self.recent_positions) / len(self.recent_positions)
            if self.euclidean(position, Position(avg_x, avg_y, 0)) < 0.02:
                reward -= 3

        if self.is_oscillating():
            reward -= 10

        if distance > 0.01 and action == 0 and not self.too_close_to_obstacle(ir_values):
            reward += 3

        return reward

    def is_oscillating(self):
        if len(self.recent_actions) < 4:
            return False
        actions = list(self.recent_actions)
        pattern1 = [1, 2, 1, 2]
        pattern2 = [2, 1, 2, 1]
        reverse_switching = all(actions[i] != actions[i+1] for i in range(len(actions)-1))
        return actions == pattern1 or actions == pattern2 or reverse_switching

    def get_distance_delta(self):
        current_position = self.rob.get_position()
        return self.euclidean(self.last_position, current_position)

    def too_close_to_obstacle(self, ir_values=None):
        if ir_values is None:
            ir_values = self.get_state()
        return any(ir > self.ir_threshold for ir in ir_values[:5])

    def euclidean(self, p1: Position, p2: Position):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)