# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

########################################################################################
# Utilities
########################################################################################


import logging
import time
import traceback
from contextlib import nullcontext
from copy import copy
from functools import cache

import rerun as rr
import torch
from deepdiff import DeepDiff
from termcolor import colored

from lerobot.common.datasets.image_writer import safe_stop_image_writer
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import get_features_from_robot
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.utils.utils import get_safe_torch_device, has_method

# TRLC
import cv2
import numpy as np


def log_control_info(robot: Robot, dt_s, episode_index=None, frame_index=None, fps=None):
    log_items = []
    if episode_index is not None:
        log_items.append(f"ep:{episode_index}")
    if frame_index is not None:
        log_items.append(f"frame:{frame_index}")

    def log_dt(shortname, dt_val_s):
        nonlocal log_items, fps
        info_str = f"{shortname}:{dt_val_s * 1000:5.2f} ({1 / dt_val_s:3.1f}hz)"
        if fps is not None:
            actual_fps = 1 / dt_val_s
            if actual_fps < fps - 1:
                info_str = colored(info_str, "yellow")
        log_items.append(info_str)

    # total step time displayed in milliseconds and its frequency
    log_dt("dt", dt_s)

    # TODO(aliberts): move robot-specific logs logic in robot.print_logs()
    if not robot.robot_type.startswith("stretch"):
        for name in robot.leader_arms:
            key = f"read_leader_{name}_pos_dt_s"
            if key in robot.logs:
                log_dt("dtRlead", robot.logs[key])

        for name in robot.follower_arms:
            key = f"write_follower_{name}_goal_pos_dt_s"
            if key in robot.logs:
                log_dt("dtWfoll", robot.logs[key])

            key = f"read_follower_{name}_pos_dt_s"
            if key in robot.logs:
                log_dt("dtRfoll", robot.logs[key])

        for name in robot.cameras:
            key = f"read_camera_{name}_dt_s"
            if key in robot.logs:
                log_dt(f"dtR{name}", robot.logs[key])

    info_str = " ".join(log_items)
    logging.info(info_str)


@cache
def is_headless():
    """Detects if python is running without a monitor."""
    try:
        import pynput  # noqa

        return False
    except Exception:
        print(
            "Error trying to import pynput. Switching to headless mode. "
            "As a result, the video stream from the cameras won't be shown, "
            "and you won't be able to change the control flow with keyboards. "
            "For more info, see traceback below.\n"
        )
        traceback.print_exc()
        print()
        return True


def predict_action(observation, policy, device, use_amp):
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        for name in observation:
            if "image" in name:
                observation[name] = observation[name].type(torch.float32) / 255
                observation[name] = observation[name].permute(2, 0, 1).contiguous()
            observation[name] = observation[name].unsqueeze(0)
            observation[name] = observation[name].to(device)

        # Compute the next action with the policy
        # based on the current observation
        action = policy.select_action(observation)

        # Remove batch dimension
        action = action.squeeze(0)

        # Move to cpu, if not already the case
        action = action.to("cpu")

    return action


def init_keyboard_listener():
    # Allow to exit early while recording an episode or resetting the environment,
    # by tapping the right arrow key '->'. This might require a sudo permission
    # to allow your terminal to monitor keyboard events.
    events = {}
    events["exit_early"] = False
    events["rerecord_episode"] = False
    events["stop_recording"] = False

    if is_headless():
        logging.warning(
            "Headless environment detected. On-screen cameras display and keyboard inputs will not be available."
        )
        listener = None
        return listener, events

    # Only import pynput if not in a headless environment
    from pynput import keyboard

    def on_press(key):
        try:
            if key == keyboard.Key.right:
                print("Right arrow key pressed. Exiting loop...")
                events["exit_early"] = True
            elif key == keyboard.Key.left:
                print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                events["rerecord_episode"] = True
                events["exit_early"] = True
            elif key == keyboard.Key.esc:
                print("Escape key pressed. Stopping data recording...")
                events["stop_recording"] = True
                events["exit_early"] = True
        except Exception as e:
            print(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener, events


def warmup_record(
    robot,
    events,
    enable_teleoperation,
    warmup_time_s,
    display_data,
    fps,
):
    control_loop(
        robot=robot,
        control_time_s=warmup_time_s,
        display_data=display_data,
        events=events,
        fps=fps,
        teleoperate=enable_teleoperation,
    )


def record_episode(
    robot,
    dataset,
    events,
    episode_time_s,
    display_data,
    policy,
    fps,
    single_task,
    masking=False,
):
    control_loop(
        robot=robot,
        control_time_s=episode_time_s,
        display_data=display_data,
        dataset=dataset,
        events=events,
        policy=policy,
        fps=fps,
        teleoperate=policy is None,
        single_task=single_task,
        masking=masking,
    )


@safe_stop_image_writer
def control_loop(
    robot,
    control_time_s=None,
    teleoperate=False,
    display_data=False,
    dataset: LeRobotDataset | None = None,
    events=None,
    policy: PreTrainedPolicy = None,
    fps: int | None = None,
    single_task: str | None = None,
    masking: bool = False,
):
    # TODO(rcadene): Add option to record logs
    if not robot.is_connected:
        robot.connect()

    if events is None:
        events = {"exit_early": False}

    if control_time_s is None:
        control_time_s = float("inf")

    if teleoperate and policy is not None:
        raise ValueError("When `teleoperate` is True, `policy` should be None.")

    if dataset is not None and single_task is None:
        raise ValueError("You need to provide a task as argument in `single_task`.")

    if dataset is not None and fps is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset['fps']} != {fps}).")

    if masking:
        import matplotlib.pyplot as plt
        from sam2.build_sam import build_sam2_camera_predictor
        sam2_checkpoint = "/home/ubuntu/trlc/segment-anything-2-real-time/checkpoints/sam2.1_hiera_small.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
        predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)
        
        
        observation = robot.capture_observation()
        first_frame = observation["observation.images.cam_head"].numpy()

        clicked_point = {}

        def onclick(event):
            if event.xdata is not None and event.ydata is not None:
                clicked_point['x'] = int(event.xdata)
                clicked_point['y'] = int(event.ydata)
                plt.close()

        fig, ax = plt.subplots()
        # Convert BGR to RGB for matplotlib
        # img_rgb = first_frame[..., ::-1]
        ax.imshow(first_frame)
        ax.set_title("Click a point on the image")
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
        fig.canvas.mpl_disconnect(cid)

        if 'x' in clicked_point and 'y' in clicked_point:
            print(f"Clicked point: ({clicked_point['x']}, {clicked_point['y']})")
        else:
            print("No point was clicked.")
            
        predictor.load_first_frame(first_frame)
        
        ann_frame_idx = 0
        obj_id = 1
        points = np.array([[clicked_point['x'], clicked_point['y']]], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)
        _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
            frame_idx=ann_frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels
        )
        

        h, w = first_frame.shape[:2]
        mask_acc = np.zeros((h, w), dtype=np.uint8)
        for logit in out_mask_logits:
            m = (logit > 0).permute(1, 2, 0).cpu().numpy().astype(np.uint8).squeeze() * 255
            mask_acc = cv2.bitwise_or(mask_acc, m)

        alpha = 0.9  # how strongly red you want the tint: 0.0 = no tint, 1.0 = full red
    
        red_img = np.zeros_like(first_frame)
        red_img[..., 0] = 255  # BGR → red channel

        blended = cv2.addWeighted(first_frame, 1.0 - alpha, red_img, alpha, 0)
        mask_bool = mask_acc.astype(bool)               # shape (h, w), True where mask
        mask_3c  = np.repeat(mask_bool[:, :, None], 3, axis=2)

        overlay = first_frame.copy()
        overlay[mask_3c] = blended[mask_3c]
        
    
        fig, ax = plt.subplots()

        ax.imshow(overlay)
        ax.set_title("Click a point on the image")
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
        fig.canvas.mpl_disconnect(cid)
        
        


    timestamp = 0
    start_episode_t = time.perf_counter()

    # Controls starts, if policy is given it needs cleaning up
    if policy is not None:
        policy.reset()

    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if teleoperate:
            observation, action = robot.teleop_step(record_data=True)
        
            if masking:    
                frame_rgb = observation["observation.images.cam_head"].numpy()
                out_obj_ids, out_mask_logits = predictor.track(frame_rgb)
                
                h, w = frame_rgb.shape[:2]
                mask_acc = np.zeros((h, w), dtype=np.uint8)
                for logit in out_mask_logits:
                    m = (logit > 0).permute(1, 2, 0).cpu().numpy().astype(np.uint8).squeeze() * 255
                    mask_acc = cv2.bitwise_or(mask_acc, m)
                    
                alpha = 0.9  # how strongly red you want the tint: 0.0 = no tint, 1.0 = full red
    
                red_img = np.zeros_like(frame_rgb)
                red_img[..., 0] = 255  # BGR → red channel

                blended = cv2.addWeighted(frame_rgb, 1.0 - alpha, red_img, alpha, 0)
                mask_bool = mask_acc.astype(bool)             
                mask_3c  = np.repeat(mask_bool[:, :, None], 3, axis=2)

                overlay = frame_rgb.copy()
                overlay[mask_3c] = blended[mask_3c]
                
                observation["observation.images.cam_head"] = torch.from_numpy(overlay)

                print(type(observation["observation.images.cam_head"]))
                print(observation["observation.images.cam_head"].shape)
                print(observation["observation.images.cam_head"].dtype)
                print(type(observation["observation.images.cam_wrist"]))
                print(observation["observation.images.cam_wrist"].shape)
                print(observation["observation.images.cam_wrist"].dtype)
        
        
        else:
            observation = robot.capture_observation()
            
            if masking:    
                frame_rgb = observation["observation.images.cam_head"].numpy()
                out_obj_ids, out_mask_logits = predictor.track(frame_rgb)
                
                h, w = frame_rgb.shape[:2]
                mask_acc = np.zeros((h, w), dtype=np.uint8)
                for logit in out_mask_logits:
                    m = (logit > 0).permute(1, 2, 0).cpu().numpy().astype(np.uint8).squeeze() * 255
                    mask_acc = cv2.bitwise_or(mask_acc, m)
                    
                alpha = 0.9  # how strongly red you want the tint: 0.0 = no tint, 1.0 = full red
    
                red_img = np.zeros_like(frame_rgb)
                red_img[..., 0] = 255  # BGR → red channel

                blended = cv2.addWeighted(frame_rgb, 1.0 - alpha, red_img, alpha, 0)
                mask_bool = mask_acc.astype(bool)             
                mask_3c  = np.repeat(mask_bool[:, :, None], 3, axis=2)

                overlay = frame_rgb.copy()
                overlay[mask_3c] = blended[mask_3c]
                
                observation["observation.images.cam_head"] = torch.from_numpy(overlay)

                print(type(observation["observation.images.cam_head"]))
                print(observation["observation.images.cam_head"].shape)
                print(observation["observation.images.cam_head"].dtype)
                print(type(observation["observation.images.cam_wrist"]))
                print(observation["observation.images.cam_wrist"].shape)
                print(observation["observation.images.cam_wrist"].dtype)
            
            action = None

            if policy is not None:
                pred_action = predict_action(
                    observation, policy, get_safe_torch_device(policy.config.device), policy.config.use_amp
                )
                # Action can eventually be clipped using `max_relative_target`,
                # so action actually sent is saved in the dataset.
                action = robot.send_action(pred_action)
                action = {"action": action}

        if dataset is not None:
            frame = {**observation, **action, "task": single_task}
            dataset.add_frame(frame)

        # TODO(Steven): This should be more general (for RemoteRobot instead of checking the name, but anyways it will change soon)
        if (display_data and not is_headless()) or (display_data and robot.robot_type.startswith("lekiwi")):
            if action is not None:
                for k, v in action.items():
                    for i, vv in enumerate(v):
                        rr.log(f"sent_{k}_{i}", rr.Scalar(vv.numpy()))

            image_keys = [key for key in observation if "image" in key]
            for key in image_keys:
                rr.log(key, rr.Image(observation[key].numpy()), static=True)

        if fps is not None:
            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / fps - dt_s)

        dt_s = time.perf_counter() - start_loop_t
        log_control_info(robot, dt_s, fps=fps)

        timestamp = time.perf_counter() - start_episode_t
        if events["exit_early"]:
            events["exit_early"] = False
            break


def reset_environment(robot, events, reset_time_s, fps):
    # TODO(rcadene): refactor warmup_record and reset_environment
    if has_method(robot, "teleop_safety_stop"):
        robot.teleop_safety_stop()

    control_loop(
        robot=robot,
        control_time_s=reset_time_s,
        events=events,
        fps=fps,
        teleoperate=True,
    )


def stop_recording(robot, listener, display_data):
    robot.disconnect()

    if not is_headless() and listener is not None:
        listener.stop()


def sanity_check_dataset_name(repo_id, policy_cfg):
    _, dataset_name = repo_id.split("/")
    # either repo_id doesnt start with "eval_" and there is no policy
    # or repo_id starts with "eval_" and there is a policy

    # Check if dataset_name starts with "eval_" but policy is missing
    if dataset_name.startswith("eval_") and policy_cfg is None:
        raise ValueError(
            f"Your dataset name begins with 'eval_' ({dataset_name}), but no policy is provided ({policy_cfg.type})."
        )

    # Check if dataset_name does not start with "eval_" but policy is provided
    if not dataset_name.startswith("eval_") and policy_cfg is not None:
        raise ValueError(
            f"Your dataset name does not begin with 'eval_' ({dataset_name}), but a policy is provided ({policy_cfg.type})."
        )


def sanity_check_dataset_robot_compatibility(
    dataset: LeRobotDataset, robot: Robot, fps: int, use_videos: bool
) -> None:
    fields = [
        ("robot_type", dataset.meta.robot_type, robot.robot_type),
        ("fps", dataset.fps, fps),
        ("features", dataset.features, get_features_from_robot(robot, use_videos)),
    ]

    mismatches = []
    for field, dataset_value, present_value in fields:
        diff = DeepDiff(dataset_value, present_value, exclude_regex_paths=[r".*\['info'\]$"])
        if diff:
            mismatches.append(f"{field}: expected {present_value}, got {dataset_value}")

    if mismatches:
        raise ValueError(
            "Dataset metadata compatibility check failed with mismatches:\n" + "\n".join(mismatches)
        )
