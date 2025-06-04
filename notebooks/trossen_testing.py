from lerobot.common.robot_devices.robots.trossen import TrossenRobot
from lerobot.common.robot_devices.robots.configs import TrossenBimanualRobotConfig
import time
import numpy as np

config = TrossenBimanualRobotConfig()
robot = TrossenRobot(config)

robot.connect()
# robot.follower_arms["left"].write("Goal_Position", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.088])
# time.sleep(5)


while True:

    robot.teleop_step()
    time.sleep(0.03)

robot.disconnect()
time.sleep(2)
    