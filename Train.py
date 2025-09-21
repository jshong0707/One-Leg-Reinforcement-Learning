import os
os.environ.setdefault("MUJOCO_GL", "glfw")  

import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList

from Env import OneLegEnv


def save_env_stats(env: OneLegEnv, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, mean=env.obs_mean, var=env.obs_var, alpha=env.alpha)

def load_env_stats(env: OneLegEnv, path: str):
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        env.obs_mean = z["mean"].astype(np.float32)
        env.obs_var  = z["var"].astype(np.float32)
        env.alpha    = float(z["alpha"])
        print(f"[EnvStats] loaded: {path}")
    else:
        print(f"[EnvStats] not found (fresh stats): {path}")


class LoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # SB3 Gymnasium 호환: info 안에 'episode' 키가 있을 때 에피소드 종료 누적 보상 접근
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append((self.model.num_timesteps, info["episode"]["r"]))
        return True

# -------------------------
# 체크포인트 경로 해석
# -------------------------
def resolve_model_path(name_or_path: str) -> str:
    """
    - 'PPO_1000000_steps' => './models/PPO_1000000_steps.zip' 탐색
    - 'models/PPO_1000000_steps.zip' 같이 직접 경로도 허용
    """
    cands = []
    if name_or_path:
        cands += [name_or_path]
        if not name_or_path.endswith(".zip"):
            cands += [name_or_path + ".zip"]
        base = os.path.join("models", name_or_path)
        cands += [base, base + ".zip"]

    for c in cands:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"Checkpoint not found among candidates: {cands}")

# -------------------------
# 학습 루틴
# -------------------------
def train(args):
    # 1) Env 생성
    env = OneLegEnv("xml/scene.xml")

    # 2) 이어학습이면 env 정규화 상태 복원 (선택)
    env_stats_path = os.path.join("data", "env_stats.npz")
    if args.continue_training:
        load_env_stats(env, env_stats_path)

    # 3) 모델 생성/로드
    if args.continue_training:
        # 기존 체크포인트에서 이어학습
        if not args.model_name:
            raise ValueError("이어학습에는 체크포인트 이름/경로가 필요합니다. 예: "
                             "python train_ppo.py PPO_1000000_steps --continue-training")
        ckpt_path = resolve_model_path(args.model_name)
        print(f"[LOAD] {ckpt_path}")
        model = PPO.load(ckpt_path, env=env, device=args.device, print_system_info=True)
        reset_num_timesteps = False
    else:
        # 처음부터 학습
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            device=args.device,
            learning_rate=3e-4,
            n_steps=4096,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            seed=2,
        )
        reset_num_timesteps = True

    # 4) 콜백: 체크포인트 + 로깅
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=args.save_dir,
        name_prefix="PPO"
    )
    logging_callback = LoggingCallback(verbose=1)
    callback = CallbackList([checkpoint_callback, logging_callback])

    # 5) 학습
    model.learn(
        total_timesteps=args.total_timesteps,
        reset_num_timesteps=reset_num_timesteps,
        callback=callback
    )

    # 6) 마지막 모델 저장 + env 정규화 상태 저장
    os.makedirs("models", exist_ok=True)
    final_path = os.path.join("models", "continue_training_model")
    model.save(final_path)
    print(f"[SAVE] final model → {final_path}.zip")

    # Env stats 저장(선택)
    save_env_stats(env, env_stats_path)
    print(f"[SAVE] env stats → {env_stats_path}")

# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Train/Continue PPO on OneLegEnv")
    parser.add_argument(
        "model_name", nargs="?", default=None,
        help="이어학습할 체크포인트 이름 또는 경로 (예: PPO_1000000_steps 또는 models/PPO_1000000_steps.zip)"
    )
    parser.add_argument(
        "--continue-training", action="store_true",
        help="기존 체크포인트에서 이어학습"
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="PPO 네트워크 연산 장치 (물리 시뮬은 CPU, NN은 GPU 사용 가능)")
    parser.add_argument("--total-timesteps", type=int, default=10_000_000,
                        help="이번 run 에서 추가로 학습할 총 timesteps")
    parser.add_argument("--save-freq", type=int, default=200_000,
                        help="체크포인트 저장 간격 (timesteps)")
    parser.add_argument("--save-dir", type=str, default="./models/",
                        help="체크포인트 저장 폴더")
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()


# resume training : python Train.py PPO_1000000_steps --continue-training