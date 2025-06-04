# sheeprl/envs/robosuite.py
from gymnasium import Env, spaces
from robomimic.envs.env_robosuite import EnvRobosuite
import numpy as np
import robomimic.utils.obs_utils as ObsUtils

class RobosuiteEnv(Env):
    def __init__(self, env_name="PickPlaceCan", camera_names=["agentview", "robot0_eye_in_hand"], camera_height=32, camera_width=32, frame_stack=5, channels_first=True, render_mode=0):
        super().__init__()
        self.channels_first = channels_first
        obs_specs = {
            "obs": {
                "rgb": ["agentview_image", "robot0_eye_in_hand_image"],
                "low_dim": [
                    "robot0_eef_pos",
                    "robot0_eef_quat",
                    "robot0_gripper_qpos",
                    "object"
                ]
            }
        }
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_specs)

        # Configure robomimic environment with both cameras
        self.env = EnvRobosuite(
            env_name=env_name,
            robots="Panda",
            controller_configs={
                "type": "OSC_POSE",
                "interpolation": "linear",
                "ramp_ratio": 0.6
            },
            use_image_obs=True,
            use_camera_obs=True,
            camera_names=camera_names,  # Enable both cameras
            camera_heights=camera_height,
            camera_widths=camera_width,
            reward_shaping=True,
            has_renderer=False,
            has_offscreen_renderer=True,
            ignore_done=False,
            control_freq=20,
            render_gpu_device_id=0,
        )

        self.render_mode = "rgb_array"
        self.frame_stack = frame_stack
        self.camera_names = camera_names

        # Set observation space shape based on channels_first
        if self.channels_first:
            img_shape = (3, camera_height, camera_width)
        else:
            img_shape = (camera_height, camera_width, 3)
        self.observation_space = spaces.Dict({
            "rgb_wrist": spaces.Box(0, 255, shape=img_shape, dtype=np.uint8),
            "rgb_third": spaces.Box(0, 255, shape=img_shape, dtype=np.uint8)
        })

        action_dim = self.env.action_dimension
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )

        #Set render mode: 0 is 1st person, 1 is 3rd person
        self.render_mode = render_mode

    def step(self, action):
        obs, reward, terminated, info = self.env.step(action)
        truncated = info.get("TimeLimit.truncated", False)
        processed_obs = self._process_observations(obs)
        self._last_obs = processed_obs
        return processed_obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs = self.env.reset()
        processed_obs = self._process_observations(obs)
        self._last_obs = processed_obs
        return processed_obs, {}

    def _process_observations(self, obs):
        image_keys = [k for k in obs.keys() if 'image' in k.lower()]
        fallback_shape = self.observation_space['rgb_wrist'].shape
        fallback = np.zeros(fallback_shape, dtype=np.uint8)

        def convert_img(img):
            if img is None or not isinstance(img, np.ndarray):
                return fallback
            arr = img
            # If (H, W, C) or (C, H, W), convert to desired format
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 1) if arr.max() <= 1.0 else np.clip(arr, 0, 255)
                arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
            if self.channels_first:
                if arr.shape == (3, fallback_shape[1], fallback_shape[2]):
                    return arr
                elif arr.shape[-1] == 3:
                    arr = np.transpose(arr, (2, 0, 1))
                return arrcfg.env.wrapper
            else:
                if arr.shape == (fallback_shape[0], fallback_shape[1], 3):
                    return arr
                elif arr.shape[0] == 3:
                    arr = np.transpose(arr, (1, 2, 0))
                return arr

        wrist_img = convert_img(obs.get("robot0_eye_in_hand_image", None))
        third_img = convert_img(obs.get("agentview_image", None))

        # Concatenate wrist and third-person views along the width (axis=2 for channels_last, axis=3 for channels_first)
        if self.channels_first:
            #TODO fix this logic since we cannot just concat the images
            concatenated_img = np.concatenate((wrist_img, third_img), axis=2)
        else:
            concatenated_img = np.concatenate((wrist_img, third_img), axis=1)

        return {
            "rgb_wrist": wrist_img,
            "rgb_third": third_img,
            # "rgb_concat": concatenated_img
        }

    def render(self, mode="rgb_array"):
        if hasattr(self, '_last_obs') and self._last_obs is not None:
            #Third person
            if self.render_mode:
                img = self._last_obs.get("rgb_third", None)
                if img is not None:
                    if img.shape[0] == 3 and len(img.shape) == 3:  # (C, H, W)
                        img = np.transpose(img, (1, 2, 0))
                    return img
            #First person
            else:
                img = self._last_obs.get("rgb_wrist", None)
                if img is not None:
                    if img.shape[0] == 3 and len(img.shape) == 3:  # (C, H, W)
                        img = np.transpose(img, (1, 2, 0))
                    return img
        #Else return none
        return None

    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()
