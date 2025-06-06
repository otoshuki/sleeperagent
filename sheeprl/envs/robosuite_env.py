# sheeprl/envs/robosuite.py
from gymnasium import Env, spaces
from robomimic.envs.env_robosuite import EnvRobosuite
import numpy as np
import robomimic.utils.obs_utils as ObsUtils
import cv2

class RobosuiteEnv(Env):
    def __init__(self, env_name="PickPlaceCan", camera_names=["agentview", "robot0_eye_in_hand"], camera_height=32, camera_width=32, frame_stack=5, channels_first=True, observation_type="rgb_wrist"):
        super().__init__()
        self.channels_first = channels_first
        self.observation_type = observation_type
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
        if self.observation_type == "rgb_concat":
            camera_width *= 2  # Double the width for concatenation
            
        if self.channels_first:
            img_shape = (3, camera_height, camera_width)
        else:
            img_shape = (camera_height, camera_width, 3)
        
        if self.observation_type == "rgb_wrist":
            self.observation_space = spaces.Dict({
                "rgb_wrist": spaces.Box(0, 255, shape=img_shape, dtype=np.uint8)
            })
        elif self.observation_type == "rgb_third":
            self.observation_space = spaces.Dict({
                "rgb_third": spaces.Box(0, 255, shape=img_shape, dtype=np.uint8)
            })
        elif self.observation_type == "rgb_concat":
            self.observation_space = spaces.Dict({
                "rgb_wrist": spaces.Box(0, 255, shape=img_shape, dtype=np.uint8),
                "rgb_third": spaces.Box(0, 255, shape=img_shape, dtype=np.uint8)
            })
        else:
            raise ValueError(f"Invalid observation_type: {self.observation_type}")
        
        action_dim = self.env.action_dimension
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )

        #Set render mode: 0 is 1st person, 1 is 3rd person
        # self.render_mode = render_mode

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
        
        if self.observation_type == "rgb_wrist":
            fallback_shape = self.observation_space['rgb_wrist'].shape
        elif self.observation_type == "rgb_third":
            fallback_shape = self.observation_space['rgb_third'].shape
        elif self.observation_type == "rgb_concat":
            fallback_shape = self.observation_space['rgb_third'].shape
        else:
            raise ValueError(f"Invalid observation_type: {self.observation_type}")
        
        fallback = np.zeros(fallback_shape, dtype=np.uint8)

        def convert_img(img):
            if img is None or not isinstance(img, np.ndarray):
                return fallback
            arr = img
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 1) if arr.max() <= 1.0 else np.clip(arr, 0, 255)
                arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
            if self.channels_first:
                if arr.shape == (3, fallback_shape[1], fallback_shape[2]):
                    return arr
                elif arr.shape[-1] == 3:
                    arr = np.transpose(arr, (2, 0, 1))
                return arr
            else:
                if arr.shape == (fallback_shape[0], fallback_shape[1], 3):
                    return arr
                elif arr.shape[0] == 3:
                    arr = np.transpose(arr, (1, 2, 0))
                return arr

        wrist_img = convert_img(obs.get("robot0_eye_in_hand_image", None))
        third_img = convert_img(obs.get("agentview_image", None))

        if self.observation_type == "rgb_wrist":
            return {"rgb_wrist": wrist_img}
        elif self.observation_type == "rgb_third":
            return {"rgb_third": third_img}
        elif self.observation_type == "rgb_concat":
            if wrist_img is None or third_img is None:
                raise ValueError("Missing wrist or third-person view for rgb_concat.")
            return {"rgb_wrist": wrist_img, "rgb_third": third_img}
        else:
            raise ValueError(f"Invalid observation_type: {self.observation_type}")

    def render(self, mode="rgb_array"):
        if hasattr(self, '_last_obs') and self._last_obs is not None:
            if self.observation_type == "rgb_concat":
                wrist_img = self._last_obs.get("rgb_wrist", None)
                third_img = self._last_obs.get("rgb_third", None)
                if wrist_img is not None and third_img is not None:
                    if self.channels_first:
                        concatenated_img = np.concatenate((wrist_img, third_img), axis=2)  # Concatenate along width
                    else:
                        concatenated_img = np.concatenate((wrist_img, third_img), axis=1)  # Concatenate along width
                    return concatenated_img.transpose(1, 2, 0) if self.channels_first else concatenated_img
            elif self.observation_type == "rgb_wrist":
                img = self._last_obs.get("rgb_wrist", None)
                if img is not None:
                    return img.transpose(1, 2, 0) if self.channels_first else img
            elif self.observation_type == "rgb_third":
                img = self._last_obs.get("rgb_third", None)
                if img is not None:
                    return img.transpose(1, 2, 0) if self.channels_first else img

        shape = self.observation_space[next(iter(self.observation_space.keys()))].shape
        if len(shape) == 3 and shape[0] == 3:
            shape = (shape[1], shape[2], shape[0])
        return np.zeros(shape, dtype=np.uint8)

    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()
