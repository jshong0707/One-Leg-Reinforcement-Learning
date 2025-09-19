#!/usr/bin/env python
import os, sys
import numpy as np
import mujoco
import mujoco.viewer
from Env import OneLegEnv
from stable_baselines3 import PPO

import matplotlib
matplotlib.use("Agg")  # GUI 없이 파일 저장
import matplotlib.pyplot as plt
import scipy.io as sio  # MATLAB .mat 저장

# ----------------- 플로팅 함수 -----------------
def plot_params(time, wn, zeta, kval, Fz, z_body, zd_body,
                out_path="params_timeseries.png"):
    fig, axs = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
    axs[0].plot(time, wn);      axs[0].set_ylabel(r"$\omega_n$ [rad/s]")
    axs[1].plot(time, zeta);    axs[1].set_ylabel(r"$\zeta$")
    axs[2].plot(time, kval);    axs[2].set_ylabel(r"$k$")
    axs[3].plot(time, Fz);      axs[3].set_ylabel(r"$F_z$")
    axs[4].plot(time, zd_body); axs[4].set_ylabel(r"$\dot{z}_{body}$")
    axs[5].plot(time, z_body);  axs[5].set_ylabel(r"$z_{body}$")
    axs[5].set_xlabel("time [s]")
    for ax in axs: ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_ee(time, xdot, zdot, x, z, out_path="ee_timeseries.png"):
    fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(time, xdot);  axs[0].set_ylabel(r"$\dot{x}$")
    axs[1].plot(time, zdot);  axs[1].set_ylabel(r"$\dot{z}$")
    axs[2].plot(time, x);     axs[2].set_ylabel(r"$x$")
    axs[3].plot(time, z);     axs[3].set_ylabel(r"$z$")
    axs[3].set_xlabel("time [s]")
    for ax in axs: ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

# ----------------- 실행 -----------------
def run(model_name: str):
    # 모델 경로 자동 처리 (models/ 접두사, .zip 확장자 보정)
    model_path = model_name
    if not os.path.exists(model_path):
        model_path = os.path.join("models", model_name)
    if not os.path.exists(model_path):
        if not model_path.endswith(".zip"):
            model_path += ".zip"
    print(f"[INFO] load model: {model_path}")

    env = OneLegEnv("xml/scene.xml")
    obs, _ = env.reset()
    agent = PPO.load(model_path, device="cpu")

    # ctrlbind import (야코비안, fk_pos)
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    _BUILD_DIR = os.path.join(_THIS_DIR, "build")
    if _BUILD_DIR not in sys.path:
        sys.path.append(_BUILD_DIR)
    import ctrlbind

    # ---- 로그 버퍼 ----
    t_log = []
    wn_log, zeta_log, k_log, Fz_log = [], [], [], []
    x_log, z_log, xdot_log, zdot_log = [], [], [], []
    z_body_log, zd_body_log = [], []   # 몸통 z, ż

    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            while viewer.is_running():
                # 정책 → step
                action, _ = agent.predict(obs, deterministic=True)
                obs, _, term, trunc, info = env.step(action)
                if term or trunc:
                    obs, _ = env.reset()

                # 시간
                t_log.append(float(env.data.time))

                # 파라미터/힘
                wn_log.append(float(info.get("omega_n", np.nan)))
                zeta_log.append(float(info.get("zeta", np.nan)))
                k_log.append(float(info.get("k", np.nan)))
                Fz_log.append(float(info.get("Fz", np.nan)))

                # EE 위치/속도
                q0 = float(env.data.qpos[1])
                q1_abs = float(env.data.qpos[1] + env.data.qpos[2])
                J = ctrlbind.jacobian(q0, q1_abs, env.ARM_LEN)
                pos = ctrlbind.fk_pos(q0, q1_abs, env.ARM_LEN)
                x_log.append(float(pos[0])); z_log.append(float(pos[1]))
                q0d = float(env.data.qvel[1])
                q1d_abs = float(env.data.qvel[1] + env.data.qvel[2])
                xdot, zdot = (J @ np.array([q0d, q1d_abs], dtype=float)).tolist()
                xdot_log.append(float(xdot)); zdot_log.append(float(zdot))

                # 몸통 z, ż (base dof)
                z_body_log.append(float(env.data.qpos[0]))
                zd_body_log.append(float(env.data.qvel[0]))

                viewer.sync()

    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        # ---- PNG 저장 ----
        plot_params(t_log, wn_log, zeta_log, k_log, Fz_log,
                    z_body_log, zd_body_log,
                    out_path="params_timeseries.png")
        plot_ee(t_log, xdot_log, zdot_log, x_log, z_log,
                out_path="ee_timeseries.png")
        print("[plot] saved: params_timeseries.png, ee_timeseries.png")

        # ---- MATLAB(.mat) 저장 ----
        sio.savemat("log_data.mat", {
            "time":    np.array(t_log),
            "wn":      np.array(wn_log),
            "zeta":    np.array(zeta_log),
            "k":       np.array(k_log),
            "Fz":      np.array(Fz_log),
            "x":       np.array(x_log),
            "z":       np.array(z_log),
            "xdot":    np.array(xdot_log),
            "zdot":    np.array(zdot_log),
            "z_body":  np.array(z_body_log),
            "zd_body": np.array(zd_body_log),
        })
        print("[mat] saved: log_data.mat")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str,
                        help="Model file (e.g., PPO_2000000_steps or PPO_2000000_steps.zip)")
    args = parser.parse_args()
    run(args.model_name)
