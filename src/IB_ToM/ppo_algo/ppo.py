import os
from typing import Optional, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

import numpy as np

from torch.utils.tensorboard import SummaryWriter

from networks import Actor, Critic, ConActor
from src.IB_ToM.utils.utils import ParameterManager
from src.IB_ToM.utils.utils import ReplayBuffer
from agents import PPOAgent

import torch.optim as optim

def collect_samples(env, agent, buffer, batch_size, state_norm):
    steps = 0
    episode_rewards = []
    state, _ = env.reset()
    done = False
    ep_reward = 0
    while steps < batch_size:
        state = state_norm(state)
        action, log_prob = agent.con_select_action(state)
        next_state, reward, done, next_done, _ = env.step(action[0])
        next_state = state_norm(next_state)
        steps += 1
        ep_reward += reward
        buffer.push(state, action[0], reward, next_state, done, log_prob)
        state = next_state
        if done or next_done:
            state, _ = env.reset()
            done = False
            episode_rewards.append(ep_reward)
            ep_reward = 0
    return steps, episode_rewards

def train(agent, buffer, writer, global_step):
    actor_loss, critic_loss = agent.update(buffer)
    writer.add_scalar('Loss/Actor', actor_loss, global_step)
    writer.add_scalar('Loss/Critic', critic_loss, global_step)
    return actor_loss, critic_loss


def get_run_log_dir(base_dir, prefix):
    """
    在 base_dir 目录下查找所有以 prefix 开头的目录，
    然后自动计算最新的序号，返回一个新的日志目录路径。
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # 查找以 prefix 开头的目录名称列表
    existing_dirs = [d for d in os.listdir(base_dir)
                     if os.path.isdir(os.path.join(base_dir, d)) and d.startswith(prefix)]

    # 从目录名称中提取数字后缀，构成一个列表，如果没有则从1开始
    run_numbers = []
    for d in existing_dirs:
        try:
            number = int(d.replace(prefix + '_', ''))
            run_numbers.append(number)
        except ValueError:
            pass

    # 自增计算下一个 run number
    run_number = max(run_numbers) + 1 if run_numbers else 1
    run_dir = os.path.join(base_dir, f"{prefix}_{run_number}")
    return run_dir


class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):  # 动态更新平均值和标准差可以用到在线算法（online algorithm），其中最常见的方法是Welford的算法
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x

def main():
    config = {
        'lr_actor': 3e-4,
        'lr_critic': 3e-4,
        'gamma': 0.99,
        'lambda': 0.95,
        'clip_epsilon': 0.2,
        'ppo_epochs': 10,
        'batch_size': 4096,
        'entropy_loss_coef': 0.01,
        'value_loss_coef': 1.0,
        'max_grad_norm': 0.5,
        'max_episodes': 1000,
        'max_timesteps': 10000000,
        'mini_batch_size': 64,
        'continuous': True,
    }
    param = ParameterManager(config)

    log_dir = get_run_log_dir('./logs/tensorboard_logs', 'ppo')

    writer = SummaryWriter(log_dir=log_dir)

    env = gym.make('MountainCarContinuous-v0', max_episode_steps=param.get("max_episodes"))
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    state_norm = Normalization(state_dim)

    agent = PPOAgent(state_dim, action_dim, 64, config)
    buffer = ReplayBuffer()

    total_timesteps = 0
    all_episode_rewards = []

    while total_timesteps < param.get("max_timesteps"):
        steps, episode_rewards = collect_samples(env, agent, buffer, param.get("batch_size"), state_norm)
        total_timesteps += steps
        all_episode_rewards.extend(episode_rewards)

        train(agent, buffer, writer, total_timesteps)

        if len(all_episode_rewards) >= 10:
            avg_reward = np.mean(all_episode_rewards[-10:])
            print(f"Total Timesteps: {total_timesteps}, Average Reward (last 10 episodes): {avg_reward:.2f}")
            writer.add_scalar("Reward/avg_last10", avg_reward, total_timesteps)

    writer.close()
    env.close()

if __name__ == "__main__":
    main()

