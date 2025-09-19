#!/usr/bin/env python
import os, sys, time
import numpy as np
import mujoco
import mujoco.viewer

#* Bindings
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BUILD_DIR = os.path.join(_THIS_DIR, "build")
sys.path.append(_BUILD_DIR)
import ctrlbind  # C++ Controller + task-space helper

# ===== 사용자 설정 =====
XML_PATH = "xml/scene.xml"
SIM_TIME = 50.0
RENDER   = True
TORQUE_LIMIT = 60.0
ARM_LEN = 0.25
# admittance parameters
OMEGA_N = 20.0
ZETA    = 1.0
K_ADM   = 150.0
# ======================

def clamp_vec(v, lim):
    return np.clip(v, -lim, lim)


def get_Fz(model, data):
    Fz = 0.0

    f = np.zeros(6, dtype=np.float64)        # cf[6] 과 동일
    mujoco.mj_contactForce(model, data, 0, f)
    Fz = f[0]
    return Fz



def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)
    dt    = model.opt.timestep

    ctrl = ctrlbind.Controller()
    
    # Task-space 목표 & error buffer
    x_des = 0.0
    z_des = -0.48
    x_ref = np.array([x_des, z_des], dtype=float)
    e_old = np.zeros(2, dtype=float)
    
    
    th = np.arccos(-z_des/(2*0.25))

    qm = np.pi/2 - th
    qb = np.pi/2 + th

    q0 = qm
    q1 = qb - qm
    data.qpos[0] = 1.3
    data.qpos[1] = q0
    data.qpos[2] = q1
    

    # mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    elapsed = 0.0
    if RENDER:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            t_start = time.time()
            while viewer.is_running() and elapsed < SIM_TIME:
                t0 = time.time()

                # biarticular joint angles
                q0 = float(data.qpos[1])                   # hip
                q1 = float(data.qpos[1] + data.qpos[2])    # knee

                # --- Fz from contacts ---
                Fz = get_Fz(model, data)

                # --- admittance controller로 z_ref 보정 ---
                dz = ctrl.admittance(OMEGA_N, ZETA, K_ADM, Fz)
                x_ref = np.array([x_des, z_des + dz], dtype=float)

                # --- task-space PID → joint torques ---
                tau, e = ctrlbind.task_pid_tau(ctrl, q0, q1, x_ref, e_old, ARM_LEN)
                e_old = e

                # --- torque mapping for biarticular actuators ---
                tau = clamp_vec(tau, TORQUE_LIMIT)
                data.ctrl[0] = float(tau[0] + tau[1])   # hip actuator
                data.ctrl[1] = float(tau[1])            # knee actuator

                mujoco.mj_step(model, data)

                elapsed = time.time() - t_start
                viewer.sync()
                frame = time.time() - t0
                if dt > frame:
                    time.sleep(dt - frame)
                    
    else:
        t_start = time.time()
        while elapsed < SIM_TIME:

            q0 = float(data.qpos[1]); q1 = float(data.qpos[1] + data.qpos[2])
            Fz = get_Fz(model, data)
            dz = ctrl.admittance(OMEGA_N, ZETA, K_ADM, Fz)
            x_ref = np.array([x_des, z_des + dz], dtype=float)
            tau, e = ctrlbind.task_pid_tau(ctrl, q0, q1, x_ref, e_old, ARM_LEN)
            e_old = e
            tau = clamp_vec(tau, TORQUE_LIMIT)
            data.ctrl[0], data.ctrl[1] = float(tau[0] + tau[1]), float(tau[1])
            mujoco.mj_step(model, data)
            elapsed = time.time() - t_start

    print("done.")

if __name__ == "__main__":
    main()
