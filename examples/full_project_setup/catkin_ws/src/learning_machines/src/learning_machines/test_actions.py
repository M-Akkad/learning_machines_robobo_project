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



def test_avoid_obstacles(rob: IRobobo):
    import rospy
    rospy.loginfo("Starting obstacle avoidance test")

    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()  # Start CoppeliaSim simulation

    try:
        while not rospy.is_shutdown():
            ir_values = rob.read_irs()
            front_ir_values = ir_values[2:6]  # FrontL, FrontR, FrontC, FrontRR

            print("Full IR values:", ir_values)
            print("Front IR values:", front_ir_values)

            obstacle_detected = any(
                val is not None and val < 0.2 for val in front_ir_values
            )
            print("Obstacle detected?", obstacle_detected)

            if obstacle_detected:
                rob.set_emotion(Emotion.SURPRISED)
                rob.talk("Obstacle ahead!")
                rob.move_blocking(-30, 30, 300)  # Turn in place
            else:
                rob.set_emotion(Emotion.HAPPY)
                rob.move_blocking(50, 50, 500)   # Move forward

    except KeyboardInterrupt:
        print("Stopped obstacle avoidance loop")

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()


