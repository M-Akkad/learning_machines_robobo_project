import rospy
from robobo_control.robot_hardware import HardwareRobobo  # adjust path if needed

def avoid_obstacles():
    rob = HardwareRobobo(camera=False)

    try:
        while not rospy.is_shutdown():
            ir_values = rob.read_irs()

            # Check front sensors: [FrontL, FrontR, FrontC, FrontRR]
            front_ir_values = ir_values[2:6]
            obstacle_detected = any(val is not None and val < 0.2 for val in front_ir_values)

            if obstacle_detected:
                rob.set_emotion("surprise")
                rob.talk("Obstacle ahead!")
                rob.move_blocking(-30, 30, 300)  # Turn right
            else:
                rob.set_emotion("happy")
                rob.move_blocking(50, 50, 500)   # Go forward

    except rospy.ROSInterruptException:
        print("ROS node interrupted")
    except KeyboardInterrupt:
        print("Manual stop")

if __name__ == "__main__":
    rospy.init_node("robobo_obstacle_avoider")
    avoid_obstacles()
