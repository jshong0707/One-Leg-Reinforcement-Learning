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

    metadata = {"render_modes": ["human"]}

    # -------------------- 사용자/제어 파라미터 --------------------
    ARM_LEN = 0.25
    TORQUE_LIMIT = 60.0
    X_DES = 0.0
    Z_DES = -0.43

    # 행동 -> 파라미터 범위 (필요시 조정)
    WN_MIN, WN_MAX = 5.0, 40.0         # rad/s
    # ZETA_MIN, ZETA_MAX = 0.2, 2.0      # -
    ZETA_MIN, ZETA_MAX = 1.0, 1.0      # -
    K_MIN, K_MAX = 50.0, 200.0        # 

    # 보상 가중치
    W_X = 5.0
    W_XDOT = 0.01
    W_TAU = 0.001
    W_FZ = 10
    ALIVE_BONUS = 0.5

    def __init__(self, xml_file="xml/scene.xml"):
        super().__init__()

        # MuJoCo
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep

    #! Binding
        self.ctrl = ctrlbind.Controller()
        self.e_old = np.zeros(2, dtype=float)  # task-space error buffer
        self.fsm = ctrlbind.FSM()
        self.traj = ctrlbind.Trajectory(self.fsm)

        
    #! Define Action space (omega_n, zeta, k)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    #! Observation: len=10 (아래 _get_obs 에서 정의)
        obs_high = np.ones(10, dtype=np.float32) * np.inf
        obs_low  = -obs_high
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        #? Obs normalization state
        self.alpha = 0.001
        self.obs_mean = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        self.obs_var  = np.ones(self.observation_space.shape[0],  dtype=np.float32)

        #? Episode state
        self.max_time = 3.0
        self.elapsed_time = 0.0

        # Targets
        self.x_des = float(self.X_DES)
        self.z_des = float(self.Z_DES)

    # --------------- 유틸: 행동 -> (omega_n, zeta, k) 매핑 ---------------
    @staticmethod
    def _affine(a, lo, hi):
        # a in [-1,1] -> [lo,hi]
        return lo + 0.5 * (a + 1.0) * (hi - lo)

    @staticmethod
    def _log_affine(a, lo, hi):
        # lo, hi > 0 이어야 함 (주파수, 강성 등은 양수 범위)
        log_lo, log_hi = np.log(lo), np.log(hi)
        t = 0.5 * (a + 1.0)  # [-1,1] -> [0,1]
        return float(np.exp(log_lo + t * (log_hi - log_lo)))

    # --------------- 액션 -> (omega_n, zeta, k) 매핑 ---------------
    def _map_action_to_params(self, action):
        a = np.asarray(action, dtype=float)

        # 로그 매핑: omega_n, k
        wn   = self._log_affine(a[0], self.WN_MIN, self.WN_MAX)
        k    = self._log_affine(a[2], self.K_MIN,  self.K_MAX)

        # 선형 매핑: zeta (범위가 좁고 절대 변화가 중요)
        zeta = self._affine(a[1], self.ZETA_MIN, self.ZETA_MAX)

        return wn, zeta, k
    
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

        # 2) Bi-articular
        q0_abs = float(self.data.qpos[1])
        q1_abs = float(self.data.qpos[1] + self.data.qpos[2])

    #! Controller
        #? Admittance
        Fz = self.get_Fz()
        dz = float(self.ctrl.admittance(omega_n, zeta, k, Fz))
        # x_ref = np.array([self.x_des, self.z_des + dz], dtype=float)
        
    #! --- contact state update ---
        self.fsm.get_pos(ctrlbind.fk_pos(q0_abs, q1_abs, self.ARM_LEN))
        self.traj.contact_state_update(self.model, self.data)

    #! --- Trajectory ---
        z_vel = float(self.data.qvel[0])
        x_ref = np.array(self.traj.get_trajectory(float(self.data.time), z_vel), dtype=float)

        # post-contact라면 admittance dz 더해줌
        x_ref[1] += dz

        #? FK/Jacobian & Error
        # (작업공간 EE 위치)
        x = np.array(ctrlbind.fk_pos(q0_abs, q1_abs, self.ARM_LEN))
        e_x = x_ref - x

        #? PID (작업공간 힘 f)
        f_task = np.asarray(self.ctrl.pid(e_x, self.e_old), dtype=float)

        #? Torque 변환 (tau = J^T f)
        J = np.array(ctrlbind.jacobian(q0_abs, q1_abs, self.ARM_LEN))
        tau = J.T @ f_task

        #? error_old 업데이트
        self.e_old = e_x.copy()

    #! Control Input
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
        q0d = float(self.data.qvel[1])
        q1d_abs = float(self.data.qvel[1] + self.data.qvel[2])
        xdot = J @ np.array([q0d, q1d_abs], dtype=float)

        w_x, w_xd, w_tau, w_Fz = self.W_X, self.W_XDOT, self.W_TAU, self.W_FZ
        alive = self.ALIVE_BONUS
        cost = (w_x * float(np.dot(e_x, e_x))
                # + w_xd * float(np.dot(xdot, xdot))
                + w_tau * (tau0_cmd**2 + tau1_cmd**2)
                + w_Fz * (Fz**2))
        reward = -cost + alive

        # 종료 조건
        z_height = float(self.data.qpos[0])
        terminated = False
        if z_height < 0.1:
            terminated = True
            reward -= 300.0

        if Fz > 200:
            terminated = True
            reward -= 300.0

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
        self.fsm  = ctrlbind.FSM()
        self.traj = ctrlbind.Trajectory(self.fsm)
        
        # 원하는 초기 포즈: z, q0_abs, q1_abs
        z0 = 0.7 + np.random.uniform(-0.05, 0.6)
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
