#!/usr/bin/env python
import os
os.environ["MUJOCO_GL"] = "glfw"  # GLFW 백엔드 사용

import gymnasium as gym
import numpy as np
import mujoco
from gymnasium import spaces
from gymnasium.utils import seeding

#* Bindings
import sys
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BUILD_DIR = os.path.join(_THIS_DIR, "build")
sys.path.append(_BUILD_DIR)
import ctrlbind

class OneLegEnv(gym.Env):

    # metadata = {"render_modes": ["human"]}

    def __init__(self, xml_file="xml/scene.xml"):
        super(OneLegEnv, self).__init__()

    #* Binding Controller

        self.ctrl = ctrlbind.Controller()
        self.error_old = np.zeros(2)


        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)

        # Action Space: 
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        pi = np.pi

        obs_high_constraint = np.array([1,  # z height
                                        pi,    # hip angle
                                        pi,    # knee angle
                                        np.inf,  # z vel          
                                        20,  # hip ang vel
                                        20,  # knee ang vel
                                        60,  # front_hip torque
                                        60,  # front_knee torque
                                        ])
        obs_low_constraint = np.array([0.4,  # z height
                                        0,    # hip angle
                                        0,    # knee angle
                                        -np.inf,  # z vel          
                                        -20,  # hip ang vel
                                        -20,  # knee ang vel
                                        -60,  # front_hip torque
                                        -60,  # front_knee torque
                                        ])
        obs_high_constraint = np.ones(8)*np.inf
        obs_low_constraint = -np.ones(8)*np.inf
        
        obs_high = np.array(obs_high_constraint, dtype=np.float32)
        obs_low = np.array(obs_low_constraint, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        self.dt = self.model.opt.timestep  # XML에 설정된 타임스텝
        self.max_time = 10                # 최대 에피소드 시간 (초)
        self.elapsed_time = 0.0
        self.done_flag = False
        self.noise_scale = 1.0

        # Observation space normalization
        obs_dim = self.observation_space.shape[0]
        self.obs_mean = np.zeros(obs_dim)
        self.obs_var = np.ones(obs_dim)
        self.alpha = 0.001

    def _get_obs(self):

        qpos = self.data.qpos
        qvel = self.data.qvel
        torque = self.data.ctrl
        obs = np.concatenate([qpos, qvel, torque]).astype(np.float32)
        return obs


    def step(self, action):



    #! Observation
        obs = self._get_obs()
        # Observation space normalization
        obs_norm = self.normalize_obs(obs)  # 여기서 normalization 수행
        

        z_height = obs[0]
        hip_angle = obs[1]
        knee_angle = obs[2]
        z_vel = obs[3]
        hip_angvel = obs[4]
        knee_angvel = obs[5]
        hip_torque = obs[6]
        knee_torque = obs[7]

        des_hip_angle = np.pi/4
        des_knee_angle = np.pi/2

    #! Reward

        alive_bonus = 0.5

    #! Weight Design
        w_pos = 0.1
        w_vel = 0.01
        w_torque = 0.001
        reward = (  - w_pos * (des_hip_angle - hip_angle)**2
                - w_pos * (des_knee_angle - knee_angle)**2 
                - w_vel * (hip_angvel**2 + knee_angvel**2)
                - w_torque * (hip_torque**2 + knee_torque**2)
                + alive_bonus)
        

    #! Fault Conditions
        done = False
        # (1) 루트 높이가 너무 낮으면 넘어짐
        if z_height < 0.15:
            done = True
            reward -= 50.0


    #! Actuate
        scaled_action = 60*action
        # print(scaled_action, '\n')

        # 학습 초기에 action에 noise를 추가
        scaled_action = scaled_action # + np.random.normal(0, self.noise_scale)
        self.noise_scale *= 0.995  # 점진적으로 줄임


        self.data.ctrl[0] = scaled_action[0]  # front_hip
        self.data.ctrl[1] = scaled_action[1]  # front_knee
        
        mujoco.mj_step(self.model, self.data)
        self.elapsed_time += self.dt
        
        return obs_norm, reward, done, False, {}




    def reset(self, **kwargs):
        mujoco.mj_resetData(self.model, self.data)
        # 초기 상태 설정: qpos와 qvel 초기화
        init_qpos = np.zeros(self.model.nq)
        init_qvel = np.zeros(self.model.nv)
        init_qpos[0] = 0.5 # np.random.uniform(-0.1, 0.1)       # z height
        init_qpos[1] = np.pi/4 # + np.random.uniform(0, 0.05) # hip angle
        init_qpos[2] = np.pi/2                                # knee angle

        self.data.qpos[:] = init_qpos
        self.data.qvel[:] = init_qvel
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.elapsed_time = 0.0
        self.done_flag = False

        obs = self._get_obs()
        
        return obs, {}

    def render(self, mode="human"):
        pass
    
        # Observation space normalization
    def normalize_obs(self, obs):
        self.obs_mean = (1 - self.alpha) * self.obs_mean + self.alpha * obs
        self.obs_var = (1 - self.alpha) * self.obs_var + self.alpha * (obs - self.obs_mean)**2
        return (obs - self.obs_mean) / (np.sqrt(self.obs_var) + 1e-8)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = np.random.RandomState(), seed
        return [seed]

if __name__ == "__main__":
    env = OneLegEnv()
    obs = env.reset()
    n_steps = 10
    for _ in range(n_steps):
    # Random action
        action = env.action_space.sample()
        obs, reward, done, A, info = env.step(action)
        
        if done:
            obs = env.reset()