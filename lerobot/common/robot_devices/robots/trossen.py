# Copyright 2024 The HuggingFace Inc. team. All rights reserved.

import json
import logging
import time
import warnings
from pathlib import Path
import trossen_arm as trossen

import numpy as np
import torch

from lerobot.common.robot_devices.cameras.utils import make_cameras_from_configs
from lerobot.common.robot_devices.motors.utils import MotorsBus, make_motors_buses_from_configs
from lerobot.common.robot_devices.robots.configs import ManipulatorRobotConfig
from lerobot.common.robot_devices.robots.utils import get_arm_id
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError


def ensure_safe_goal_position(
    goal_pos: torch.Tensor, present_pos: torch.Tensor, max_relative_target: float | list[float]
):
    # Cap relative action target magnitude for safety.
    diff = goal_pos - present_pos
    max_relative_target = torch.tensor(max_relative_target)
    safe_diff = torch.minimum(diff, max_relative_target)
    safe_diff = torch.maximum(safe_diff, -max_relative_target)
    safe_goal_pos = present_pos + safe_diff

    # if not torch.allclose(goal_pos, safe_goal_pos):
    #     logging.warning(
    #         "Relative goal position magnitude had to be clamped to be safe.\n"
    #         f"  requested relative goal position target: {diff}\n"
    #         f"    clamped relative goal position target: {safe_diff}"
    #     )

    return safe_goal_pos

def map_range(x, src_min, src_max, dst_min, dst_max):
    # position of x inside the source range, as a fraction 0–1
    t = (x - src_min) / (src_max - src_min)
    # stretch and shift that fraction into the target range
    return t * (dst_max - dst_min) + dst_min

class TrossenRobot:
    def __init__(
        self,
        config: ManipulatorRobotConfig,
    ):
        self.config = config
        self.robot_type = self.config.type
        self.calibration_dir = Path(self.config.calibration_dir)
        self.leader_arms = make_motors_buses_from_configs(self.config.leader_arms)
        self.follower_arms = make_motors_buses_from_configs(self.config.follower_arms)
        self.cameras = make_cameras_from_configs(self.config.cameras)
        self.is_connected = False
        self.logs = {}

    @property
    def camera_features(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            key = f"observation.images.{cam_key}"
            cam_ft[key] = {
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return cam_ft

    @property
    def motor_features(self) -> dict:
        return {
            "action": {
                "dtype": "float32",
                "shape": (14,),
                "names": ["left_joint1", 
                          "left_joint2", 
                          "left_joint3", 
                          "left_joint4", 
                          "left_joint5", 
                          "left_joint6", 
                          "left_gripper"
                          "right_joint1", 
                          "right_joint2", 
                          "right_joint3", 
                          "right_joint4", 
                          "right_joint5", 
                          "right_joint6", 
                          "right_gripper"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (14,),
                "names": ["left_joint1", 
                          "left_joint2", 
                          "left_joint3", 
                          "left_joint4", 
                          "left_joint5", 
                          "left_joint6", 
                          "left_gripper"
                          "right_joint1", 
                          "right_joint2", 
                          "right_joint3", 
                          "right_joint4", 
                          "right_joint5", 
                          "right_joint6", 
                          "right_gripper"],
            },
        }

    @property
    def features(self):
        return {**self.motor_features, **self.camera_features}

    @property
    def has_camera(self):
        return len(self.cameras) > 0

    @property
    def num_cameras(self):
        return len(self.cameras)

    @property
    def available_arms(self):
        available_arms = []
        for name in self.follower_arms:
            arm_id = get_arm_id(name, "follower")
            available_arms.append(arm_id)
        for name in self.leader_arms:
            arm_id = get_arm_id(name, "leader")
            available_arms.append(arm_id)
        return available_arms

    def teleop_safety_stop(self):
        if self.robot_type in ["trossen_bimanual"]:
            for arms in self.follower_arms:
                self.follower_arms[arms].write("Reset", 1)
            time.sleep(2)
            for arms in self.follower_arms:
                self.follower_arms[arms].write("Torque_Enable", 1)

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "ManipulatorRobot is already connected. Do not run `robot.connect()` twice."
            )
            
                
        for name in self.follower_arms:
            print(f"Connecting {name} follower arm.")
            self.follower_arms[name].connect()
        for name in self.leader_arms:
            print(f"Connecting {name} leader arm.")
            self.leader_arms[name].connect()
        time.sleep(2)
        
        from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
        # for name in self.follower_arms:
        #     self.follower_arms[name].write("Torque_Enable", TorqueMode.DISABLED.value)
        for name in self.leader_arms:
            self.leader_arms[name].write("Torque_Enable", TorqueMode.DISABLED.value)
            
        self.activate_calibration()


        if self.config.leader_gripper_open_degree is not None:
            for name in self.leader_arms:
                self.leader_arms[name].write("Torque_Enable", 0, "gripper")  
                self.leader_arms[name].write("Operating_Mode", 5, "gripper")
                self.leader_arms[name].write("Current_Limit",  100, "gripper")

                self.leader_arms[name].write("Torque_Enable", 1, "gripper")
                self.leader_arms[name].write("Goal_Position", self.config.leader_gripper_open_degree, "gripper")
        
        
        for name in self.follower_arms:
            self.follower_arms[name].read("Present_Position")
        for name in self.leader_arms:
            self.leader_arms[name].read("Present_Position")
        
        for name in self.cameras:
            self.cameras[name].connect()

        self.is_connected = True   
    
    def activate_calibration(self):
        """After calibration all motors function in human interpretable ranges.
        Rotations are expressed in degrees in nominal range of [-180, 180],
        and linear motions (like gripper of Aloha) in nominal range of [0, 100].
        """

        def load_or_run_calibration_(name, arm, arm_type):
            arm_id = get_arm_id(name, arm_type)
            arm_calib_path = self.calibration_dir / f"{arm_id}.json"

            if arm_calib_path.exists():
                with open(arm_calib_path) as f:
                    calibration = json.load(f)
            else:
                print(f"Missing calibration file '{arm_calib_path}'")

                from lerobot.common.robot_devices.robots.dynamixel_calibration import run_arm_calibration
                calibration = run_arm_calibration(arm, self.robot_type, name, arm_type)

                print(f"Calibration is done! Saving calibration file '{arm_calib_path}'")
                arm_calib_path.parent.mkdir(parents=True, exist_ok=True)
                with open(arm_calib_path, "w") as f:
                    json.dump(calibration, f)

            return calibration

        for name, arm in self.leader_arms.items():
            calibration = load_or_run_calibration_(name, arm, "leader")
            arm.set_calibration(calibration)
        
        
    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

        # Prepare to assign the position of the leader to the follower
        leader_pos = {}
        for name in self.leader_arms:
            before_lread_t = time.perf_counter()
            leader_pos[name] = self.leader_arms[name].read("Present_Position")
            leader_pos[name] = torch.from_numpy(leader_pos[name])
            self.logs[f"read_leader_{name}_pos_dt_s"] = time.perf_counter() - before_lread_t

        # Send goal position to the follower
        follower_goal_pos = {}
        for name in self.follower_arms:
            before_fwrite_t = time.perf_counter()
            goal_pos = leader_pos[name]
            
            if self.config.max_relative_target is not None:
                present_pos = np.rad2deg(self.follower_arms[name].read("Present_Position"))
                present_pos = torch.from_numpy(present_pos)
                goal_pos[:6] = ensure_safe_goal_position(goal_pos[:6], present_pos[:6], self.config.max_relative_target)
            
            follower_goal_pos[name] = goal_pos

            # WRITE RAD
            goal_pos = goal_pos.numpy().astype(np.float32)
            goal_pos[:6] = np.deg2rad(goal_pos[:6])
            goal_pos[6] = map_range(goal_pos[6], 
                                    self.config.leader_gripper_open_degree, 
                                    self.config.leader_gripper_close_degree, 
                                    self.config.follower_gripper_open_m, 
                                    self.config.follower_gripper_close_m)
            
            self.follower_arms[name].write("Goal_Position", goal_pos)
        
            self.logs[f"write_follower_{name}_goal_pos_dt_s"] = time.perf_counter() - before_fwrite_t

        # Early exit when recording data is not requested
        if not record_data:
            return

        # Read follower position
        follower_pos = {}
        for name in self.follower_arms:
            before_fread_t = time.perf_counter()
            follower_pos[name] = np.rad2deg(self.follower_arms[name].read("Present_Position"))
            follower_pos[name] = torch.from_numpy(np.array(follower_pos[name], dtype=np.float32))
            self.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - before_fread_t

        # Create state by concatenating follower current position
        state = []
        for name in self.follower_arms:
            if name in follower_pos:
                state.append(follower_pos[name])
        state = torch.cat(state)

        # Create action by concatenating follower goal position
        action = []
        for name in self.follower_arms:
            if name in follower_goal_pos:
                action.append(follower_goal_pos[name])
        action = torch.cat(action)

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Populate output dictionaries
        obs_dict, action_dict = {}, {}
        obs_dict["observation.state"] = state
        action_dict["action"] = action
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]

        return obs_dict, action_dict

    def capture_observation(self):
        """The returned observations do not have a batch dimension."""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

        # Read follower position
        follower_pos = {}
        for name in self.follower_arms:
            before_fread_t = time.perf_counter()
            follower_pos[name] = np.rad2deg(self.follower_arms[name].read("Present_Position"))
            follower_pos[name] = torch.from_numpy(np.array(follower_pos[name], dtype=np.float32))
            self.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - before_fread_t

        # Create state by concatenating follower current position
        state = []
        for name in self.follower_arms:
            if name in follower_pos:
                state.append(follower_pos[name])
        state = torch.cat(state)

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Populate output dictionaries and format to pytorch
        obs_dict = {}
        obs_dict["observation.state"] = state
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]
        return obs_dict

    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        """Command the follower arms to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Args:
            action: tensor containing the concatenated goal positions for the follower arms.
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

        # from_idx = 0
        # to_idx = 0
        # action_sent = []
        for name in self.follower_arms:

            goal_pos = torch.from_numpy(action.numpy().astype(np.float32))            
            
            if self.config.max_relative_target is not None:
                present_pos = np.rad2deg(self.follower_arms[name].read("Present_Position"))
                present_pos = torch.from_numpy(present_pos)
                goal_pos[:6] = ensure_safe_goal_position(goal_pos[:6], present_pos[:6], self.config.max_relative_target)
        
            goal_pos = goal_pos.numpy().astype(np.float32)
            goal_pos[:6] = np.deg2rad(goal_pos[:6])
            goal_pos[6] = map_range(goal_pos[6], 
                                    self.config.leader_gripper_open_degree, 
                                    self.config.leader_gripper_close_degree, 
                                    self.config.follower_gripper_open_m, 
                                    self.config.follower_gripper_close_m)
            
            self.follower_arms[name].write("Goal_Position", goal_pos)
                
            action_sent=goal_pos
            
        return action_sent


    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()` before disconnecting."
            )

        for name in self.follower_arms:
            self.follower_arms[name].disconnect()

        for name in self.leader_arms:
            self.leader_arms[name].disconnect()
        time.sleep(2)

        for name in self.cameras:
            self.cameras[name].disconnect()

        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
