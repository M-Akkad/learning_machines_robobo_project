import os
import torch
from robobo_interface import SimulationRobobo, HardwareRobobo, IRobobo
from rl_environment import RoboboRLEnvironment
from train_rl_agent import DQN

def setup_results_directory():
    docker_results = "/root/results"
    if os.path.exists("/root") and os.access("/root", os.W_OK):
        os.makedirs(docker_results, exist_ok=True)
        print(f"Using Docker results directory: {docker_results}")
        return docker_results
    fallback = os.getcwd()
    print(f"Fallback directory used: {fallback}")
    return fallback

RESULTS_DIR = setup_results_directory()
MODEL_PATH = os.path.join(RESULTS_DIR, "best_model.pth")

DEFAULT_IR_THRESHOLD = 100.0
DEFAULT_STEPS = 40
DEFAULT_MODE = "simulation"

def load_model():
    model = DQN(input_size=8, output_size=3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model

def print_detailed_sensor_info(rob: IRobobo):
    ir = rob.read_irs()
    pos = rob.get_position()
    ori = rob.get_orientation()
    acc = rob.read_accel()

    print("  IR Sensor values:", ir)
    print(f"  Position: x={pos.x:.3f}, y={pos.y:.3f}, z={pos.z:.3f}")

    try:
        print(f"  Orientation: alpha={ori.alpha:.1f}, beta={ori.beta:.1f}, gamma={ori.gamma:.1f}")
    except AttributeError:
        print(f"  Orientation: {ori}")

    print(f"  Accel: x={acc.x:.2f}, y={acc.y:.2f}, z={acc.z:.2f}")

def test_model(rob: IRobobo, ir_threshold=DEFAULT_IR_THRESHOLD, steps=DEFAULT_STEPS, mode="simulation"):
    print(f"Running {mode} test with IR threshold: {ir_threshold}, Steps: {steps}")
    env = RoboboRLEnvironment(rob)
    env.ir_threshold = ir_threshold

    model = load_model()
    rob.play_simulation()
    state = env.reset()
    total_reward = 0

    for step in range(steps):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = torch.argmax(model(state_tensor)).item()

        print(f"\nStep {step + 1}/{steps} | Action: {action}")
        print_detailed_sensor_info(rob)

        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state

        if done:
            print("Stopped early due to termination condition.")
            break

    rob.stop_simulation()
    print(f"Test complete. Total reward: {total_reward:.2f}")

def run_trained_model_test(rob, ir_threshold=DEFAULT_IR_THRESHOLD, steps=DEFAULT_STEPS, mode=DEFAULT_MODE):
    test_model(rob, ir_threshold=ir_threshold, steps=steps, mode=mode)

if __name__ == "__main__":
    print("Test Trained Model")
    print(f"Model path: {MODEL_PATH}")

    mode = input("Select mode [simulation/hardware] (default: simulation): ").strip().lower() or DEFAULT_MODE
    ir_thresh = input(f"Set IR threshold (default: {DEFAULT_IR_THRESHOLD}): ").strip()
    steps = input(f"Set number of steps (default: {DEFAULT_STEPS}): ").strip()

    ir_thresh = float(ir_thresh) if ir_thresh else DEFAULT_IR_THRESHOLD
    steps = int(steps) if steps else DEFAULT_STEPS

    if mode == "hardware":
        robobo = HardwareRobobo()
    else:
        robobo = SimulationRobobo()

    test_model(robobo, ir_threshold=ir_thresh, steps=steps, mode=mode)