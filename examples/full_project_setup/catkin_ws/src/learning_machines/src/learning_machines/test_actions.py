import cv2

from data_files import FIGURES_DIR
from robobo_interface import (
    IRobobo,
    Emotion,
    LedId,
    LedColor,
    SoundEmotion,
    SimulationRobobo,
    HardwareRobobo,
)
from rl_controller import RLController
import numpy as np
import os
from data_files import RESULTS_DIR
from pathlib import Path
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"




def test_emotions(rob: IRobobo):
    rob.set_emotion(Emotion.HAPPY)
    rob.talk("Hello")
    rob.play_emotion_sound(SoundEmotion.PURR)
    rob.set_led(LedId.FRONTCENTER, LedColor.GREEN)


def test_move_and_wheel_reset(rob: IRobobo):
    rob.move_blocking(100, 100, 1000)
    print("before reset: ", rob.read_wheels())
    rob.reset_wheels()
    rob.sleep(1)
    print("after reset: ", rob.read_wheels())


def test_sensors(rob: IRobobo):
    print("IRS data: ", rob.read_irs())
    image = rob.read_image_front()
    cv2.imwrite(str(FIGURES_DIR / "photo.png"), image)
    print("Phone pan: ", rob.read_phone_pan())
    print("Phone tilt: ", rob.read_phone_tilt())
    print("Current acceleration: ", rob.read_accel())
    print("Current orientation: ", rob.read_orientation())


def test_phone_movement(rob: IRobobo):
    rob.set_phone_pan_blocking(20, 100)
    print("Phone pan after move to 20: ", rob.read_phone_pan())
    rob.set_phone_tilt_blocking(50, 100)
    print("Phone tilt after move to 50: ", rob.read_phone_tilt())


def test_sim(rob: SimulationRobobo):
    print("Current simulation time:", rob.get_sim_time())
    print("Is the simulation currently running? ", rob.is_running())
    rob.stop_simulation()
    print("Simulation time after stopping:", rob.get_sim_time())
    print("Is the simulation running after shutting down? ", rob.is_running())
    rob.play_simulation()
    print("Simulation time after starting again: ", rob.get_sim_time())
    print("Current robot position: ", rob.get_position())
    print("Current robot orientation: ", rob.get_orientation())

    pos = rob.get_position()
    orient = rob.get_orientation()
    rob.set_position(pos, orient)
    print("Position the same after setting to itself: ", pos == rob.get_position())
    print("Orient the same after setting to itself: ", orient == rob.get_orientation())


def test_hardware(rob: HardwareRobobo):
    print("Phone battery level: ", rob.phone_battery())
    print("Robot battery level: ", rob.robot_battery())


def run_all_actions(rob: IRobobo):
    test_avoid_obstacles(rob)



# def test_avoid_obstacles(rob: IRobobo):
#     import rospy
#     import time
#     import json

#     rospy.loginfo("Starting obstacle avoidance test")

#     if isinstance(rob, SimulationRobobo):
#         rob.play_simulation()

#     start_time = time.time()
#     ir_log = []
#     time_log = []

#     try:
#         for _ in range(50):  # Run for 50 iterations
#             now = time.time() - start_time
#             ir_values = rob.read_irs()
#             rospy.loginfo(f"[DEBUG] IR values: {ir_values}")

#             ir_log.append(ir_values)
#             time_log.append(now)

#             front_ir_values = ir_values[0:5]
#             obstacle_detected = any(val is not None and val > 25 for val in front_ir_values)

#             if obstacle_detected:
#                 rob.set_emotion(Emotion.SURPRISED)
#                 rob.talk("Obstacle ahead!")
#                 rob.move_blocking(-50, 50, 300)  # Turn in place
#             else:
#                 rob.set_emotion(Emotion.HAPPY)
#                 rob.move_blocking(50, 50, 500)  # Move forward

#     except KeyboardInterrupt:
#         print("Stopped obstacle avoidance loop")

#     if isinstance(rob, SimulationRobobo):
#         rob.stop_simulation()

#     # Save logs
#     import os
#     print("Saving data to:", os.path.abspath("trial_1_data.json"))

#     with open("trial_1_data.json", "w") as f:
#         json.dump({"time": time_log, "ir": ir_log}, f)


def test_avoid_obstacles(rob: IRobobo):
    import rospy
    import time
    import json
    from robobo_interface import Emotion

    rospy.loginfo("Starting obstacle avoidance test with RL controller")

    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    controller = RLController()

    start_time = time.time()
    ir_log = []
    time_log = []
    reward_log = []
    action_log = []

    q_table_path = RESULTS_DIR / "q_table.npy"
    if q_table_path.exists():
        controller.load_q_table(q_table_path)
        print("Loaded existing Q-table.")


    try:
        for step in range(1000):
            now = time.time() - start_time
            ir_values = rob.read_irs()
            rospy.loginfo(f"[DEBUG] IR values: {ir_values}")

            ir_log.append(ir_values)
            time_log.append(now)

            front_ir_values = ir_values[0:5]
            state_idx = controller.discretize_state(ir_values)
            action = controller.select_action(state_idx)

            # Execute action
            if action == 0:
                rob.set_emotion(Emotion.HAPPY)
                rob.move_blocking(50, 50, 500)  # Forward
            elif action == 1:
                rob.set_emotion(Emotion.SURPRISED)
                rob.move_blocking(-50, 50, 300)  # Turn left
            elif action == 2:
                rob.set_emotion(Emotion.SURPRISED)
                rob.move_blocking(50, -50, 300)  # Turn right
            elif action == 3:
                rob.set_emotion(Emotion.SAD)
                rob.move_blocking(0, 0, 500)  # Stop

            # Compute reward
            if any(val is not None and val > 50 for val in front_ir_values):
                reward = -5  # very close to obstacle → BAD
                if action == 0:  # tried to move forward into obstacle → extra penalty
                    reward -= 2
            elif any(val is not None and val > 30 for val in front_ir_values):
                reward = -2  # somewhat close → cautious
                if action == 0:
                    reward -= 1
            else:
                if action == 0:
                    reward = +3  # moving forward in free space → BEST!
                elif action in [1, 2]:  # turning unnecessarily
                    reward = -1
                elif action == 3:  # stopping unnecessarily
                    reward = -3


            reward_log.append(reward)
            action_log.append(action)

            # Read new state and update controller
            next_ir_values = rob.read_irs()
            next_state_idx = controller.discretize_state(next_ir_values)

            controller.update(state_idx, action, reward, next_state_idx, done=False)

    except KeyboardInterrupt:
        print("Stopped obstacle avoidance loop")

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()
 
    controller.save_q_table(q_table_path)
    print("Saved Q-table.")


    # Save logs
    # Get directory of this file
    output_path = Path("/root/results") / "trial_1_data.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump({
            "time": [float(t) for t in time_log],
            "ir": [[float(v) if v is not None else None for v in ir_row] for ir_row in ir_log],
            "reward": [float(r) for r in reward_log],
            "action": [int(a) for a in action_log]
        }, f)

    print("Saving data to:", output_path.resolve())



