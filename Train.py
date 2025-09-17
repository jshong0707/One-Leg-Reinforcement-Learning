#!/usr/bin/env python
import sys
import argparse
import os
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")  # Non-GUI backend
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from Env import OneLegEnv

class LoggingCallback(BaseCallback):
    """
    사용자 정의 콜백:
    - 에피소드가 끝날 때마다 누적 reward를 기록합니다.
    - 각 rollout 종료 시 PPO 내부 logger에서 (global timestep과 함께)
      'train/loss', 'train/policy_gradient_loss', 'train/value_loss',
      'train/approx_kl', 'train/clip_fraction', 'train/entropy_loss',
      'train/explained_variance' 등의 metric들을 기록합니다.
    - 학습 종료 시 기록된 데이터를 "data/training_logs.mat" 파일로 저장합니다.
    """
    def __init__(self, verbose=0):
        super(LoggingCallback, self).__init__(verbose)
        self.episode_rewards = []      # list of (global timestep, episode reward)
        self.total_losses = []         # list of (global timestep, total loss)
        self.policy_grad_losses = []   # list of (global timestep, policy gradient loss)
        self.value_losses = []         # list of (global timestep, value loss)
        self.approx_kls = []           # list of (global timestep, approx KL)
        self.clip_fractions = []       # list of (global timestep, clip fraction)
        self.entropy_losses = []       # list of (global timestep, entropy loss)
        self.explained_variances = []  # list of (global timestep, explained variance)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append((self.model.num_timesteps, info["episode"]["r"]))
        return True

    def _on_rollout_end(self) -> None:
        current_timestep = self.model.num_timesteps
        # PPO의 내부 logger에서 metric들을 가져옵니다.
        metrics = getattr(self.model.logger, "name_to_value", {})
        if "train/loss" in metrics:
            self.total_losses.append((current_timestep, metrics["train/loss"]))
        if "train/policy_gradient_loss" in metrics:
            self.policy_grad_losses.append((current_timestep, metrics["train/policy_gradient_loss"]))
        if "train/value_loss" in metrics:
            self.value_losses.append((current_timestep, metrics["train/value_loss"]))
        if "train/approx_kl" in metrics:
            self.approx_kls.append((current_timestep, metrics["train/approx_kl"]))
        if "train/clip_fraction" in metrics:
            self.clip_fractions.append((current_timestep, metrics["train/clip_fraction"]))
        if "train/entropy_loss" in metrics:
            self.entropy_losses.append((current_timestep, metrics["train/entropy_loss"]))
        if "train/explained_variance" in metrics:
            self.explained_variances.append((current_timestep, metrics["train/explained_variance"]))

    def _on_training_end(self) -> None:
        log_data = {
            "Episode_rewards": np.array(self.episode_rewards),
            "Total_losses": np.array(self.total_losses),
            "Policy_grad_losses": np.array(self.policy_grad_losses),
            "Value_losses": np.array(self.value_losses),
            "Approx_kls": np.array(self.approx_kls),
            "Clip_fractions": np.array(self.clip_fractions),
            "Entropy_losses": np.array(self.entropy_losses),
            "Explained_variances": np.array(self.explained_variances)
        }
        data_folder = "./data"
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        sio.savemat(os.path.join(data_folder, "training_logs.mat"), log_data)
        if self.verbose:
            print("Saved training logs to", data_folder)
        
        # 플롯 생성: 에피소드 reward, policy gradient loss, value loss 각각
        if log_data["episode_rewards"].size > 0:
            timesteps, rewards = log_data["episode_rewards"].T
            plt.figure()
            plt.plot(timesteps, rewards, marker='o', linestyle='-')
            plt.title("Episode Rewards")
            plt.xlabel("Global Timestep")
            plt.ylabel("Reward")
            plt.savefig(os.path.join(data_folder, "episode_rewards.png"))
            plt.close()
        if log_data["policy_grad_losses"].size > 0:
            timesteps, pg_losses = log_data["policy_grad_losses"].T
            plt.figure()
            plt.plot(timesteps, pg_losses, marker='o', linestyle='-')
            plt.title("Policy Gradient Loss")
            plt.xlabel("Global Timestep")
            plt.ylabel("Policy Gradient Loss")
            plt.savefig(os.path.join(data_folder, "policy_grad_losses.png"))
            plt.close()
        if log_data["value_losses"].size > 0:
            timesteps, v_losses = log_data["value_losses"].T
            plt.figure()
            plt.plot(timesteps, v_losses, marker='o', linestyle='-')
            plt.title("Value Loss")
            plt.xlabel("Global Timestep")
            plt.ylabel("Value Loss")
            plt.savefig(os.path.join(data_folder, "value_losses.png"))
            plt.close()

class TrainPPO:
    def __init__(self, reset_timesteps):
        self.reset_timesteps = reset_timesteps

    def run_training(self):
        env = OneLegEnv("xml/scene.xml")

        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            device = "cpu",
            learning_rate=3e-4,
            n_steps=4096,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            seed=2
        )

        total_timesteps = 10000000

        checkpoint_callback = CheckpointCallback(
            save_freq=50000,
            save_path='./models/',
            name_prefix='PPO'
        )

        logging_callback = LoggingCallback(verbose=1)

        callback = CallbackList([checkpoint_callback, logging_callback])

        model.learn(total_timesteps=total_timesteps,
                    reset_num_timesteps=self.reset_timesteps,
                    callback=callback)
        model.save("oneleg_ppo_model")
        print("Model saved as oneleg_ppo_model.zip")



def main():
    parser = argparse.ArgumentParser(description="Train PPO on Walker2DEnv with optional timestep reset.")
    parser.add_argument(
        "--continue-training",
        action="store_true",
        help="If continuing training from an existing model, do not reset num_timesteps."
    )
    args = parser.parse_args()
    train_instance = TrainPPO(reset_timesteps=not args.continue_training)
    train_instance.run_training()

if __name__ == "__main__":
    main()


# 이어서 학습하는 코드 =  python PPO_1000000_steps --continue-training

