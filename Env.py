# #!/usr/bin/env python
# import os
# os.environ["MUJOCO_GL"] = "glfw"  # GLFW 백엔드 사용

# import gymnasium as gym
# import numpy as np
# import mujoco
# from gymnasium import spaces
# from gymnasium.utils import seeding

# #* Bindings
# import sys
# _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# _BUILD_DIR = os.path.join(_THIS_DIR, "build")
# sys.path.append(_BUILD_DIR)
# import ctrlbind

# class OneLegEnv(gym.Env):

#     # metadata = {"render_modes": ["human"]}

#     def __init__(self, xml_file="xml/scene.xml"):
#         super(OneLegEnv, self).__init__()

#     #* Binding Controller

#         self.ctrl = ctrlbind.Controller()
#         self.error_old = np.zeros(2)


#         self.model = mujoco.MjModel.from_xml_path(xml_file)
#         self.data = mujoco.MjData(self.model)

#         # Action Space: 
#         self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

#         pi = np.pi

#         obs_high_constraint = np.array([1,  # z height
#                                         pi,    # hip angle
#                                         pi,    # knee angle
#                                         np.inf,  # z vel          
#                                         20,  # hip ang vel
#                                         20,  # knee ang vel
#                                         60,  # front_hip torque
#                                         60,  # front_knee torque
#                                         ])
#         obs_low_constraint = np.array([0.4,  # z height
#                                         0,    # hip angle
#                                         0,    # knee angle
#                                         -np.inf,  # z vel          
#                                         -20,  # hip ang vel
#                                         -20,  # knee ang vel
#                                         -60,  # front_hip torque
#                                         -60,  # front_knee torque
#                                         ])
#         obs_high_constraint = np.ones(8)*np.inf
#         obs_low_constraint = -np.ones(8)*np.inf
        
#         obs_high = np.array(obs_high_constraint, dtype=np.float32)
#         obs_low = np.array(obs_low_constraint, dtype=np.float32)
#         self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

#         self.dt = self.model.opt.timestep  # XML에 설정된 타임스텝
#         self.max_time = 10                # 최대 에피소드 시간 (초)
#         self.elapsed_time = 0.0
#         self.done_flag = False
#         self.noise_scale = 1.0

#         # Observation space normalization
#         obs_dim = self.observation_space.shape[0]
#         self.obs_mean = np.zeros(obs_dim)
#         self.obs_var = np.ones(obs_dim)
#         self.alpha = 0.001

#     def _get_obs(self):

#         qpos = self.data.qpos
#         qvel = self.data.qvel
#         torque = self.data.ctrl
#         obs = np.concatenate([qpos, qvel, torque]).astype(np.float32)
#         return obs


#     def step(self, action):



#     #! Observation
#         obs = self._get_obs()
#         # Observation space normalization
#         obs_norm = self.normalize_obs(obs)  # 여기서 normalization 수행
        

#         z_height = obs[0]
#         hip_angle = obs[1]
#         knee_angle = obs[2]
#         z_vel = obs[3]
#         hip_angvel = obs[4]
#         knee_angvel = obs[5]
#         hip_torque = obs[6]
#         knee_torque = obs[7]

#         des_hip_angle = np.pi/4
#         des_knee_angle = np.pi/2

#     #! Reward

#         alive_bonus = 0.5

#     #! Weight Design
#         w_pos = 0.1
#         w_vel = 0.01
#         w_torque = 0.001
#         reward = (  - w_pos * (des_hip_angle - hip_angle)**2
#                 - w_pos * (des_knee_angle - knee_angle)**2 
#                 - w_vel * (hip_angvel**2 + knee_angvel**2)
#                 - w_torque * (hip_torque**2 + knee_torque**2)
#                 + alive_bonus)
        

#     #! Fault Conditions
#         done = False
#         # (1) 루트 높이가 너무 낮으면 넘어짐
#         if z_height < 0.15:
#             done = True
#             reward -= 50.0


#     #! Actuate
#         scaled_action = 60*action
#         # print(scaled_action, '\n')

#         # 학습 초기에 action에 noise를 추가
#         scaled_action = scaled_action # + np.random.normal(0, self.noise_scale)
#         self.noise_scale *= 0.995  # 점진적으로 줄임


#         self.data.ctrl[0] = scaled_action[0]  # front_hip
#         self.data.ctrl[1] = scaled_action[1]  # front_knee
        
#         mujoco.mj_step(self.model, self.data)
#         self.elapsed_time += self.dt
        
#         return obs_norm, reward, done, False, {}




#     def reset(self, **kwargs):
#         mujoco.mj_resetData(self.model, self.data)
#         # 초기 상태 설정: qpos와 qvel 초기화
#         init_qpos = np.zeros(self.model.nq)
#         init_qvel = np.zeros(self.model.nv)
#         init_qpos[0] = 0.5 # np.random.uniform(-0.1, 0.1)       # z height
#         init_qpos[1] = np.pi/4 # + np.random.uniform(0, 0.05) # hip angle
#         init_qpos[2] = np.pi/2                                # knee angle

#         self.data.qpos[:] = init_qpos
#         self.data.qvel[:] = init_qvel
#         self.data.time = 0.0
#         mujoco.mj_forward(self.model, self.data)

#         self.elapsed_time = 0.0
#         self.done_flag = False

#         obs = self._get_obs()
        
#         return obs, {}

#     def render(self, mode="human"):
#         pass
    
#         # Observation space normalization
#     def normalize_obs(self, obs):
#         self.obs_mean = (1 - self.alpha) * self.obs_mean + self.alpha * obs
#         self.obs_var = (1 - self.alpha) * self.obs_var + self.alpha * (obs - self.obs_mean)**2
#         return (obs - self.obs_mean) / (np.sqrt(self.obs_var) + 1e-8)

#         self.seed()
#         self.reset()

#     def seed(self, seed=None):
#         self.np_random, seed = np.random.RandomState(), seed
#         return [seed]

# if __name__ == "__main__":
#     env = OneLegEnv()
#     obs = env.reset()
#     n_steps = 10
#     for _ in range(n_steps):
#     # Random action
#         action = env.action_space.sample()
#         obs, reward, done, A, info = env.step(action)
        
#         if done:
#             obs = env.reset()


import os
os.environ["MUJOCO_GL"] = "glfw"  # GLFW 백엔드 사용

import sys
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

#* Bindings
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BUILD_DIR = os.path.join(_THIS_DIR, "build")
sys.path.append(_BUILD_DIR)
import ctrlbind  # Controller + fk_pos/jacobian/task_pid_tau

class OneLegEnv(gym.Env):
    """
    Action: a in [-1,1]^3  -> (omega_n, zeta, k) for z-admittance
    Control law:
      - Fz from contacts (sum of contact-frame z)
      - dz = Admittance(omega_n, zeta, k, Fz)
      - x_ref = [x_des, z_des + dz]
      - (tau, e_x) = task_pid_tau(ctrl, q0_abs, q1_abs, x_ref, e_old)
      - bi-articular mapping: ctrl[0] = tau[0] + tau[1], ctrl[1] = tau[1]
    Observation (len=10):
      [ z, q0_abs, q1_abs, z_dot, q0_dot, q1_abs_dot, tau0, tau1, Fz, dz ]
    """
    metadata = {"render_modes": ["human"]}

    # -------------------- 사용자/제어 파라미터 --------------------
    ARM_LEN = 0.25
    TORQUE_LIMIT = 60.0
    X_DES = 0.0
    Z_DES = -0.48

    # 행동 -> 파라미터 범위 (필요시 조정)
    WN_MIN, WN_MAX = 5.0, 80.0         # rad/s
    ZETA_MIN, ZETA_MAX = 0.2, 2.0      # -
    K_MIN, K_MAX = 50.0, 3000.0        # 단위는 impl.에 따름

    # 보상 가중치
    W_X = 5.0
    W_XDOT = 0.1
    W_TAU = 0.001
    ALIVE_BONUS = 0.5

    def __init__(self, xml_file="xml/scene.xml"):
        super().__init__()

        # MuJoCo
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep

        # Controller
        self.ctrl = ctrlbind.Controller()
        self.e_old = np.zeros(2, dtype=float)  # task-space error buffer

        # Action: (omega_n, zeta, k)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Observation: len=10 (아래 _get_obs 에서 정의)
        obs_high = np.ones(10, dtype=np.float32) * np.inf
        obs_low  = -obs_high
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # Obs normalization state
        self.alpha = 0.001
        self.obs_mean = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        self.obs_var  = np.ones(self.observation_space.shape[0],  dtype=np.float32)

        # Episode state
        self.max_time = 10.0
        self.elapsed_time = 0.0

        # Targets
        self.x_des = float(self.X_DES)
        self.z_des = float(self.Z_DES)

    # --------------- 유틸: 행동 -> (omega_n, zeta, k) 매핑 ---------------
    @staticmethod
    def _affine(a, lo, hi):
        # a in [-1,1] -> [lo,hi]
        return lo + 0.5 * (a + 1.0) * (hi - lo)

    def _map_action_to_params(self, action):
        a = np.asarray(action, dtype=float)
        wn   = self._affine(a[0], self.WN_MIN,   self.WN_MAX)
        zeta = self._affine(a[1], self.ZETA_MIN, self.ZETA_MAX)
        k    = self._affine(a[2], self.K_MIN,    self.K_MAX)
        return float(wn), float(zeta), float(k)

    # --------------- 유틸: 접촉에서 Fz 합산 ---------------
    def get_Fz(self):
        Fz = 0.0

        f = np.zeros(6, dtype=np.float64)        # cf[6] 과 동일
        mujoco.mj_contactForce(self.model, self.data, 0, f)
        Fz = f[0]
        return Fz


    # --------------- 관측 구성 ---------------
    def _get_obs(self, tau0, tau1, Fz, dz):
        # bi-articular: q0_abs = qpos[1], q1_abs = qpos[1] + qpos[2]
        q0_abs = float(self.data.qpos[1])
        q1_abs = float(self.data.qpos[1] + self.data.qpos[2])
        z = float(self.data.qpos[0])

        q0d = float(self.data.qvel[1])
        q1d_abs = float(self.data.qvel[1] + self.data.qvel[2])
        zd = float(self.data.qvel[0])

        obs = np.array([z, q0_abs, q1_abs, zd, q0d, q1d_abs,
                        float(tau0), float(tau1), float(Fz), float(dz)],
                       dtype=np.float32)
        return obs

    def normalize_obs(self, obs):
        self.obs_mean = (1 - self.alpha) * self.obs_mean + self.alpha * obs
        self.obs_var  = (1 - self.alpha) * self.obs_var  + self.alpha * (obs - self.obs_mean)**2
        return (obs - self.obs_mean) / (np.sqrt(self.obs_var) + 1e-8)

    # --------------- 스텝 ---------------
    def step(self, action):
        # 1) 행동 -> admittance 파라미터
        omega_n, zeta, k = self._map_action_to_params(action)

        # 2) 상태 읽기 (bi-articular)
        q0_abs = float(self.data.qpos[1])
        q1_abs = float(self.data.qpos[1] + self.data.qpos[2])

    #! Controller
        #? Admittance
        Fz = self.get_Fz()  # 또는 self._get_Fz_worldZ()
        dz = float(self.ctrl.admittance(omega_n, zeta, k, Fz))
        x_ref = np.array([self.x_des, self.z_des + dz], dtype=float)

        #? PID
        tau, e_x = ctrlbind.task_pid_tau(self.ctrl, q0_abs, q1_abs, x_ref, self.e_old, self.ARM_LEN)
        self.e_old = np.array(e_x, dtype=float)

    #! Control Input
        tau = np.asarray(tau, dtype=float)
        tau = np.clip(tau, -self.TORQUE_LIMIT, self.TORQUE_LIMIT)
        tau0_cmd = float(tau[0] + tau[1])  # hip actuator
        tau1_cmd = float(tau[1])           # knee actuator
        self.data.ctrl[0] = tau0_cmd
        self.data.ctrl[1] = tau1_cmd

    #! Simulation
        mujoco.mj_step(self.model, self.data)
        self.elapsed_time += self.dt

    #! Observation
        obs = self._get_obs(tau0_cmd, tau1_cmd, Fz, dz)
        obs_norm = self.normalize_obs(obs)

    #! Reward
        J = ctrlbind.jacobian(q0_abs, q1_abs, self.ARM_LEN)
        q0d = float(self.data.qvel[1])
        q1d_abs = float(self.data.qvel[1] + self.data.qvel[2])
        xdot = J @ np.array([q0d, q1d_abs], dtype=float)

        w_x, w_xd, w_tau = self.W_X, self.W_XDOT, self.W_TAU
        alive = self.ALIVE_BONUS
        cost = (w_x * float(np.dot(e_x, e_x))
                + w_xd * float(np.dot(xdot, xdot))
                + w_tau * (tau0_cmd**2 + tau1_cmd**2))
        reward = -cost + alive

        # 9) 종료 조건
        z_height = float(self.data.qpos[0])
        terminated = False
        if z_height < 0.15:
            terminated = True
            reward -= 50.0

        truncated = False
        if self.elapsed_time >= self.max_time:
            truncated = True

        info = {
            "omega_n": omega_n, "zeta": zeta, "k": k,
            "Fz": Fz, "dz": dz,
            "e_x": np.array(e_x, dtype=float),
        }
        return obs_norm, float(reward), terminated, truncated, info

    # --------------- 리셋 ---------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # 원하는 초기 포즈: z, q0_abs, q1_abs
        z0 = 0.5
        L = self.ARM_LEN
        # z_des 기반 대칭 초기자세 (예시)
        th = np.arccos(np.clip(-self.z_des/(2*L), -1.0, 1.0))
        q0_abs = np.pi/2 - th
        q1_abs = np.pi/2 + th

        # bi-articular 저장 규칙: qpos[2] = q1_abs - q0_abs
        self.data.qpos[0] = z0
        self.data.qpos[1] = q0_abs
        self.data.qpos[2] = (q1_abs - q0_abs)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.e_old = np.zeros(2, dtype=float)
        self.elapsed_time = 0.0

        # 빈 토크/힘/보정으로 관측 구성
        obs = self._get_obs(0.0, 0.0, 0.0, 0.0).astype(np.float32)
        return obs, {}

    def render(self):
        pass
