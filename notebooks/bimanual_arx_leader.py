from lerobot.common.robot_devices.robots.arx import ARXRobot
from lerobot.common.robot_devices.robots.configs import ARXRobotConfig
from lerobot.common.robot_devices.robots.dynamixel_calibration import run_arm_calibration

from lerobot.common.robot_devices.robots.configs import ARXBimanualRobotConfig

import time
import torch
import numpy as np
config = ARXBimanualRobotConfig()

robot = ARXRobot(config)

print(robot.leader_arms)
print("connect to robot")
robot.connect()

while True:
    pos_right = robot.leader_arms["right"].read("Present_Position")
    pos_left = robot.leader_arms["left"].read("Present_Position")
    print("Right leader", np.round(pos_right, 2))
    print("Left leader", np.round(pos_left, 2))
    time.sleep(0.1)

robot.disconnect()
