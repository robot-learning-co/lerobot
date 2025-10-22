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
import traceback
from contextlib import nullcontext
from copy import copy
from functools import cache
from typing import Any

import numpy as np
import torch
from deepdiff import DeepDiff

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import DEFAULT_FEATURES
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.robots import Robot


import os
import sys
import threading
import time
from contextlib import contextmanager

@cache
def is_headless():
    """
    Detect whether we should run in headless/SSH mode.

    Heuristics:
    - No GUI display (no $DISPLAY on POSIX) -> headless
    - On Windows, assume non-headless (pynput should work) unless explicitly broken.
    - If pynput import works AND a display is present, treat as non-headless.
    """
    # POSIX: if there's no DISPLAY, we’re almost certainly headless/SSH
    if os.name != "nt" and not os.environ.get("DISPLAY"):
        return True

    try:
        import pynput  # noqa: F401
        return False
    except Exception:
        # Either import failed or no backend available -> headless
        return True

class _HeadlessKeyListener:
    """
    Minimal stdin-based key listener that mimics pynput.Listener enough for this script.

    - start(): starts a background thread reading from stdin
    - stop(): stops it and restores terminal state
    - Works in SSH/TTY sessions.
    """

    def __init__(self, on_press):
        self._on_press = on_press
        self._stop = threading.Event()
        self._thread = None
        self._is_tty = sys.stdin.isatty()
        self._restore_ctx = None

    def start(self):
        if not self._is_tty:
            # Nothing to do: no TTY to read from (e.g., piped)
            return

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        # best-effort restore (POSIX only)
        if self._restore_ctx is not None:
            try:
                self._restore_ctx.__exit__(None, None, None)
            except Exception:
                pass

    def _run(self):
        if os.name == "nt":
            self._run_windows()
        else:
            self._run_posix()

    # ---------- POSIX ----------
    @contextmanager
    def _raw_tty(self):
        import termios, tty  # POSIX only

        fd = sys.stdin.fileno()
        old_attrs = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)  # cbreak is more forgiving than raw and works fine here
            yield
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)

    def _read_posix_nonblocking(self):
        import select
        rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
        if rlist:
            return os.read(sys.stdin.fileno(), 32)  # read up to 32 bytes
        return b""

    def _run_posix(self):
        # Arrow keys arrive as escape sequences: ESC [ C (right), ESC [ D (left)
        # Esc alone is just b'\x1b'
        with self._raw_tty() as ctx:
            self._restore_ctx = ctx  # store for stop()
            buf = b""
            while not self._stop.is_set():
                chunk = self._read_posix_nonblocking()
                if not chunk:
                    continue
                buf += chunk

                # Consume sequences in buffer
                while buf:
                    # ESC sequences
                    if buf.startswith(b"\x1b["):
                        # Need at least 3 bytes for ESC [ X
                        if len(buf) < 3:
                            break  # wait for more
                        seq = buf[:3]
                        buf = buf[3:]
                        if seq == b"\x1b[C":  # Right arrow
                            self._safe_on_press(("ARROW_RIGHT",))
                        elif seq == b"\x1b[D":  # Left arrow
                            self._safe_on_press(("ARROW_LEFT",))
                        else:
                            # Unhandled CSI, swallow
                            pass
                    elif buf.startswith(b"\x1b"):
                        # Lone ESC
                        buf = buf[1:]
                        self._safe_on_press(("ESC",))
                    else:
                        # Other chars (Enter, letters, etc.) — we ignore.
                        # Consume one byte to avoid infinite loop.
                        buf = buf[1:]

    # ---------- Windows ----------
    def _run_windows(self):
        # Windows arrow keys: first a prefix (224) then a code:
        # 77 = right, 75 = left. ESC = 27.
        import msvcrt

        while not self._stop.is_set():
            if not msvcrt.kbhit():
                time.sleep(0.05)
                continue
            ch = msvcrt.getch()
            if not ch:
                continue
            c = ch[0] if isinstance(ch, bytes) else ch

            if c == 27:  # ESC
                self._safe_on_press(("ESC",))
            elif c in (0, 224):
                # Arrow/function sequence
                nxt = msvcrt.getch()
                if not nxt:
                    continue
                n = nxt[0] if isinstance(nxt, bytes) else nxt
                if n == 77:
                    self._safe_on_press(("ARROW_RIGHT",))
                elif n == 75:
                    self._safe_on_press(("ARROW_LEFT",))
                # else ignore others

    def _safe_on_press(self, key_tuple):
        try:
            self._on_press(key_tuple)
        except Exception as e:
            print(f"Error handling key press: {e}", file=sys.stderr)


def predict_action(
    observation: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    use_amp: bool,
    task: str | None = None,
    robot_type: str | None = None,
):
    """
    Performs a single-step inference to predict a robot action from an observation.

    This function encapsulates the full inference pipeline:
    1. Prepares the observation by converting it to PyTorch tensors and adding a batch dimension.
    2. Runs the preprocessor pipeline on the observation.
    3. Feeds the processed observation to the policy to get a raw action.
    4. Runs the postprocessor pipeline on the raw action.
    5. Formats the final action by removing the batch dimension and moving it to the CPU.

    Args:
        observation: A dictionary of NumPy arrays representing the robot's current observation.
        policy: The `PreTrainedPolicy` model to use for action prediction.
        device: The `torch.device` (e.g., 'cuda' or 'cpu') to run inference on.
        preprocessor: The `PolicyProcessorPipeline` for preprocessing observations.
        postprocessor: The `PolicyProcessorPipeline` for postprocessing actions.
        use_amp: A boolean to enable/disable Automatic Mixed Precision for CUDA inference.
        task: An optional string identifier for the task.
        robot_type: An optional string identifier for the robot type.

    Returns:
        A `torch.Tensor` containing the predicted action, ready for the robot.
    """
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        observation = prepare_observation_for_inference(observation, device, task, robot_type)
        observation = preprocessor(observation)

        # Compute the next action with the policy
        # based on the current observation
        action = policy.select_action(observation)

        action = postprocessor(action)

    return action


def init_keyboard_listener():
    """
    Initializes a non-blocking keyboard listener that works BOTH with a GUI (pynput)
    and headless SSH sessions (stdin-based).

    Returns:
        (listener, events)
        - listener: object with .start() and .stop() methods (pynput.Listener or _HeadlessKeyListener)
        - events: dict with flags: exit_early, rerecord_episode, stop_recording
    """
    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
    }

    def set_flag(key):
        """
        Normalized key handler. For GUI mode, `key` looks like pynput keyboard keys.
        For headless mode, `key` is a tuple like ("ARROW_RIGHT",) or ("ESC",).
        """
        # Headless normalized tuples
        if isinstance(key, tuple):
            if key == ("ARROW_RIGHT",):
                print("Right arrow key pressed. Exiting loop...")
                events["exit_early"] = True
            elif key == ("ARROW_LEFT",):
                print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                events["rerecord_episode"] = True
                events["exit_early"] = True
            elif key == ("ESC",):
                print("Escape key pressed. Stopping data recording...")
                events["stop_recording"] = True
                events["exit_early"] = True
            return

        # pynput mode
        try:
            from pynput import keyboard  # imported only if available

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
        except Exception:
            # If pynput import fails mid-run, ignore
            pass

    if is_headless():
        # Headless: use stdin-based listener
        listener = _HeadlessKeyListener(on_press=set_flag)
        listener.start()
        if not sys.stdin.isatty():
            logging.warning(
                "Headless mode detected but no TTY on stdin. Keyboard inputs won’t be available."
            )
        else:
            logging.info("Headless keyboard listener active (SSH/TTY).")
        return listener, events

    # GUI-capable: use pynput
    from pynput import keyboard
    listener = keyboard.Listener(on_press=set_flag)
    listener.start()
    return listener, events


def sanity_check_dataset_name(repo_id, policy_cfg):
    """
    Validates the dataset repository name against the presence of a policy configuration.

    This function enforces a naming convention: a dataset repository ID should start with "eval_"
    if and only if a policy configuration is provided for evaluation purposes.

    Args:
        repo_id: The Hugging Face Hub repository ID of the dataset.
        policy_cfg: The configuration object for the policy, or `None`.

    Raises:
        ValueError: If the naming convention is violated.
    """
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
    dataset: LeRobotDataset, robot: Robot, fps: int, features: dict
) -> None:
    """
    Checks if a dataset's metadata is compatible with the current robot and recording setup.

    This function compares key metadata fields (`robot_type`, `fps`, and `features`) from the
    dataset against the current configuration to ensure that appended data will be consistent.

    Args:
        dataset: The `LeRobotDataset` instance to check.
        robot: The `Robot` instance representing the current hardware setup.
        fps: The current recording frequency (frames per second).
        features: The dictionary of features for the current recording session.

    Raises:
        ValueError: If any of the checked metadata fields do not match.
    """
    fields = [
        ("robot_type", dataset.meta.robot_type, robot.robot_type),
        ("fps", dataset.fps, fps),
        ("features", dataset.features, {**features, **DEFAULT_FEATURES}),
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
