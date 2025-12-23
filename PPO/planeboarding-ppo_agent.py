import gymnasium as gym
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from Environments.PlaneBoarding.planeboarding import PlaneEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement, StopTrainingOnRewardThreshold

import os

base_dir = Path(__file__).parent.parent
model_dir = base_dir / "models" / "planeboarding"
log_dir = base_dir / "logs" / "planeboarding"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

def train():
    env = make_vec_env(PlaneEnv, n_envs=12, env_kwargs={"nrows":10, "seats_per_row":5}, vec_env_cls=SubprocVecEnv)
    model = MaskablePPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    
    eval_callback = MaskableEvalCallback(
        env,
        eval_freq=10_000,
        verbose=1,
        best_model_save_path=os.path.join(model_dir, "MaskablePPO"),
    )
    model.learn(total_timesteps=int(1e10), callback=eval_callback)
    

def test_sb3(model_name, render=True):
    env = gym.make("PlaneBoarding-v0", nrows=10, seats_per_row=5, render_mode="human" if render else None)

    # Load model
    model = MaskablePPO.load(f"{model_dir}/MaskablePPO/{model_name}", env=env)

    rewards = 0
    # Run a test
    obs, _ = env.reset()
    terminated = False

    while True:
        action_masks = get_action_masks(env)
        # Turn on deterministic, so predict always returns the same behavior
        action, _ = model.predict(observation=obs, deterministic=True, action_masks=action_masks)
        obs, reward, terminated, _, _ = env.step(action)
        rewards += reward
        if terminated:
            break
    
    print(f"Total rewards: {rewards}")

if __name__ == "__main__":
    # train()
    test_sb3("best_model")