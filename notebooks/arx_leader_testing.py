from lerobot.common.robot_devices.robots.arx import ARXRobot
from lerobot.common.robot_devices.robots.configs import ARXRobotConfig
from lerobot.common.robot_devices.robots.dynamixel_calibration import run_arm_calibration

import time
import torch
import numpy as np
config = ARXRobotConfig()
robot = ARXRobot(config)

print(robot.leader_arms)
robot.connect()

while True:
    pos = robot.leader_arms["main"].read("Present_Position")
    print(np.round(pos, 2))
    time.sleep(0.01)

robot.disconnect()
