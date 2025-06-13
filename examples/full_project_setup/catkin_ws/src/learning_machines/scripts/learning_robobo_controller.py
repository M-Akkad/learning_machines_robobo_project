#!/usr/bin/env python3
import sys

from robobo_interface import SimulationRobobo, HardwareRobobo
from test_actions import run_trained_model_test

DEFAULT_IR_THRESHOLD = 100.0
DEFAULT_STEPS = 40.0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError(
            "To run, pass `--hardware` or `--simulation` to specify the mode."
        )

    ir_threshold = DEFAULT_IR_THRESHOLD
    steps = DEFAULT_STEPS

    if sys.argv[1] == "--hardware":
        rob = HardwareRobobo()
        mode = "hardware"
    elif sys.argv[1] == "--simulation":
        rob = SimulationRobobo()
        mode = "simulation"
    else:
        raise ValueError(f"{sys.argv[1]} is not a valid mode.")

    if "--ir" in sys.argv:
        idx = sys.argv.index("--ir")
        ir_threshold = float(sys.argv[idx + 1])
    if "--steps" in sys.argv:
        idx = sys.argv.index("--steps")
        steps = int(sys.argv[idx + 1])

    # run_trained_model_test(rob, ir_threshold=25.0, steps=40, mode="hardware")
    run_trained_model_test(rob, ir_threshold=100.0, steps=60, mode="simulation")
    # train()
