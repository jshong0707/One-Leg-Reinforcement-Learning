import os
os.environ.setdefault("MUJOCO_GL", "egl")

import sys, time
import numpy as np
import mujoco
import mujoco.viewer
import scipy.io as sio  # .mat 저장

#* Bindings
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BUILD_DIR = os.path.join(_THIS_DIR, "build")
if _BUILD_DIR not in sys.path:
    sys.path.append(_BUILD_DIR)
import ctrlbind  # C++ Controller + task-space helper

# ===== 사용자 설정 =====
XML_PATH = "xml/scene.xml"
SIM_TIME = 50.0
RENDER   = True
TORQUE_LIMIT = 60.0
ARM_LEN = 0.25
# admittance parameters (C++과 동일)
OMEGA_N = 20.0
ZETA    = 1.0
K_ADM   = 150.0
# decimation: C++ RK4 콜백과 동기화가 아닌, 파이썬 메인루프 기준 제어 → 1 권장
CONTROL_DECIMATION = 1
# ======================

def clamp_vec(v, lim):
    return np.clip(v, -lim, lim)

def get_Fz(model, data):
    """C++과 동일: contact 0의 contact-frame 힘성분 f[0] 사용."""
    f = np.zeros(6, dtype=np.float64)
    try:
        mujoco.mj_contactForce(model, data, 0, f)
        return float(f[0])   # contact frame normal 첫 성분
    except Exception:
        return 0.0

# (참고) ctrlbind에도 fk_pos, jacobian 이 있지만, 여기서는 동일부호 정의를 로컬로 유지
def fk_and_jacobian_planar_2link(q0, q1_abs, L):
    """사용자 부호/정의와 일치하는 2R 평면팔 FK/Jacobian ([x,z])."""
    c0, s0 = np.cos(q0), np.sin(q0)
    c1, s1 = np.cos(q1_abs), np.sin(q1_abs)
    x = -L * (c0 + c1)
    z = -L * (s0 + s1)
    J = np.array([[ L*s0,  L*s1],
                  [-L*c0, -L*c1]], dtype=float)
    return x, z, J

def main():
    # --- 모델/데이터 ---
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)
    dt    = model.opt.timestep

    # Controller 인스턴스 (매 스텝 재생성 X)
    ctrl = ctrlbind.Controller()

    # --- 초기 자세: C++ 로직과 동일 (z_des 기반 역기구학) ---
    z_des = -0.43
    th = np.arccos(-z_des/(2*ARM_LEN))
    qm = np.pi/2 - th
    qb = np.pi/2 + th
    q0_init = qm
    q1_init = qb - qm
    data.qpos[0] = 0.9
    data.qpos[1] = q0_init
    data.qpos[2] = q1_init

    mujoco.mj_forward(model, data)

    # --- 로깅 버퍼 (pre-step 타이밍) ---
    time_log = []
    q0_log, q1_log = [], []
    tau0_log, tau1_log = [], []
    xref_log, zref_log = [], []
    Fz_log = []
    x_log, z_log = [], []
    xdot_log, zdot_log = [], []
    z_body_log, zd_body_log = [], []
    wn_log, zeta_log, k_log = [], [], []

    # --- 내부 상태 ---
    e_old = np.zeros(2, dtype=float)
    world_cnt = 0
    loop_index = 0

    def log_pre_step(q0, q1_abs, tau, Fz, x_ref):
        """'스텝 전(pre-step)' 상태를 기록 (C++ save_data_leg 타이밍과 동일)."""
        # 시간 (pre-step)
        time_log.append(float(data.time))

        # 관절/토크
        q0_log.append(float(q0))
        q1_log.append(float(q1_abs))
        tau0_log.append(float(tau[0]))
        tau1_log.append(float(tau[1]))

        # 몸통
        z_body_log.append(float(data.qpos[0]))
        zd_body_log.append(float(data.qvel[0]))

        # EE 상태 (pre-step)
        x_pre, z_pre, J_pre = fk_and_jacobian_planar_2link(q0, q1_abs, ARM_LEN)
        x_log.append(float(x_pre)); z_log.append(float(z_pre))
        q0d = float(data.qvel[1])
        q1d_abs = float(data.qvel[1] + data.qvel[2])
        v = J_pre @ np.array([q0d, q1d_abs], dtype=float)
        xdot_log.append(float(v[0])); zdot_log.append(float(v[1]))

        # 기준/외력/파라미터
        xref_log.append(float(x_ref[0])); zref_log.append(float(x_ref[1]))
        Fz_log.append(float(Fz))
        wn_log.append(float(OMEGA_N)); zeta_log.append(float(ZETA)); k_log.append(float(K_ADM))

    def control_and_log_if_needed():
        nonlocal e_old, world_cnt, loop_index

        # 현재(pre-step) 상태
        q0 = float(data.qpos[1])
        q1_abs = float(data.qpos[1] + data.qpos[2])

        if (world_cnt % CONTROL_DECIMATION) == 0:
            # 1) GRF (pre-step)
            Fz = get_Fz(model, data)

            # 2) 어드미턴스 → z_ref
            dz = ctrl.admittance(OMEGA_N, ZETA, K_ADM, Fz)
            x_ref = np.array([0.0, z_des + dz], dtype=float)

            # 3) FK/J (pre-step)  →  작업공간 오차 e
            x_pre, z_pre, J_pre = fk_and_jacobian_planar_2link(q0, q1_abs, ARM_LEN)
            e = np.array([x_ref[0] - x_pre, x_ref[1] - z_pre], dtype=float)

            # 4) ★ PID(error, error_old) = 작업공간 힘 f  (요청사항)
            f_task = np.asarray(ctrl.pid(e, e_old), dtype=float)

            # 5) tau = J^T f  (작업공간 → 관절공간)
            tau = J_pre.T @ f_task

            # 6) 다음 스텝을 위한 error_old 갱신
            e_old[:] = e

            # 7) 토크 제한 + qvel 포화(수치 안정)
            tau = clamp_vec(tau, TORQUE_LIMIT)
            if data.qvel[1] > 20:  data.qvel[1] = 20
            if data.qvel[1] < -20: data.qvel[1] = -20
            s = data.qvel[1] + data.qvel[2]
            if s > 20:   data.qvel[2] =  20 - data.qvel[1]
            if s < -20:  data.qvel[2] = -20 - data.qvel[1]

            # 8) biarticular torque mapping (pre-step → 이번 스텝에 적용)
            data.ctrl[0] = float(tau[0] + tau[1])   # HIP actuator
            data.ctrl[1] = float(tau[1])            # KNEE actuator

            # 9) 로깅(pre-step)
            log_pre_step(q0, q1_abs, tau, Fz, x_ref)
            loop_index += 1

        world_cnt += 1

    save_path = os.path.join("data", "maindata.mat")


    # === 메인 루프 ===
    try:
        if RENDER:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                while viewer.is_running() and (data.time < SIM_TIME):
                    t0 = time.time()

                    # 스텝 전 제어/로깅
                    control_and_log_if_needed()

                    # 스텝
                    mujoco.mj_step(model, data)

                    viewer.sync()
                    # (선택) 실시간 유지
                    frame = time.time() - t0
                    if dt > frame:
                        time.sleep(max(0.0, dt - frame))
        else:
            while data.time < SIM_TIME:
                t0 = time.time()
                control_and_log_if_needed()
                mujoco.mj_step(model, data)
                frame = time.time() - t0
                if dt > frame:
                    time.sleep(max(0.0, dt - frame))
    finally:
        # === 저장 ===
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        payload = {
            "time":    np.asarray(time_log,    dtype=np.float64),
            "q0":      np.asarray(q0_log,      dtype=np.float64),
            "q1":      np.asarray(q1_log,      dtype=np.float64),
            "tau0":    np.asarray(tau0_log,    dtype=np.float64),
            "tau1":    np.asarray(tau1_log,    dtype=np.float64),
            "x_ref":   np.asarray(xref_log,    dtype=np.float64),
            "z_ref":   np.asarray(zref_log,    dtype=np.float64),
            "Fz":      np.asarray(Fz_log,      dtype=np.float64),
            "x":       np.asarray(x_log,       dtype=np.float64),
            "z":       np.asarray(z_log,       dtype=np.float64),
            "xdot":    np.asarray(xdot_log,    dtype=np.float64),
            "zdot":    np.asarray(zdot_log,    dtype=np.float64),
            "z_body":  np.asarray(z_body_log,  dtype=np.float64),
            "zd_body": np.asarray(zd_body_log, dtype=np.float64),
            "wn":      np.asarray(wn_log,      dtype=np.float64),
            "zeta":    np.asarray(zeta_log,    dtype=np.float64),
            "k":       np.asarray(k_log,       dtype=np.float64),
        }
        sio.savemat(save_path, payload)
        print("[SAVE]", os.path.abspath(save_path))
        print("[len] time/wn/zeta/k =", len(time_log), len(wn_log), len(zeta_log), len(k_log))
        print("done.")

if __name__ == "__main__":
    main()
