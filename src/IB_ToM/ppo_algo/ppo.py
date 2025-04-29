import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

import numpy as np

from torch.utils.tensorboard import SummaryWriter

from networks import Actor, Critic, ConActor
from src.IB_ToM.utils.utils import ParameterManager
from src.IB_ToM.utils.utils import ReplayBuffer

import torch.optim as optim

class PPOAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, config):
        self.config = config
        self.param = ParameterManager(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim, hidden_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, hidden_dim).to(self.device)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.param.get("lr_actor"))
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.param.get("lr_critic"))

    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        probs = self.actor(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()

    def con_select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        return self.actor.get_action(state_tensor)

    def update(self, buffer):
        states, actions, rewards, next_states, dones, old_log_probs = buffer.get_batch()
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            values = self.critic(states).squeeze()
            next_values = self.critic(next_states).squeeze()

            advantages = torch.zeros_like(rewards)
            gae = 0.0
            gamma = self.param.get("gamma")
            lamb = self.param.get("lambda")
            for t in reversed(range(len(rewards))):
                delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
                gae = delta + gamma * lamb * (1 - dones[t]) * gae
                advantages[t] = gae

        returns = advantages + values

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.param.get("ppo_epochs")):
            for state, action, old_log_prob, advantage, return_ in batch_generator(states, actions, old_log_probs, advantages, returns, self.param.get("mini_batch_size")):
                if self.param.get("continuous"):
                    mu, std = self.actor(state)
                    dist = torch.distributions.Normal(mu, std)
                else:
                    probs = self.actor(state)
                    dist = torch.distributions.Categorical(probs)
                new_log_prob = dist.log_prob(action)
                ratio = torch.exp(new_log_prob - old_log_prob)

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.param.get("clip_epsilon"),
                                    1.0 + self.param.get("clip_epsilon")) * advantage

                actor_loss = (-torch.min(surr1, surr2) - self.param.get("entropy_loss_coef") * dist.entropy()).mean()

                value = self.critic(state).float()

                value_loss = F.mse_loss(value, return_.unsqueeze(-1))

                critic_loss = 0.5* self.param.get("value_loss_coef") * value_loss.mean()

                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.param.get("max_grad_norm"))
                self.optimizer_actor.step()

                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.param.get("max_grad_norm"))
                self.optimizer_critic.step()

        buffer.clear()
        return actor_loss.item(), critic_loss.item()

    def save(self, ckpt_path: str, extra_info: dict | None = None) -> None:
        """
        Save actor/critic weights、optimizer states 以及可选的自定义信息。

        Args:
            ckpt_path (str): 存档文件完整路径（.pt or .pth 均可）。
            extra_info (dict, optional): 额外需要保存的内容，例如
                当前训练步数、环境名、评价指标等。
        """
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

        payload = {
            "actor_state_dict":  self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_actor_state":  self.optimizer_actor.state_dict(),
            "optimizer_critic_state": self.optimizer_critic.state_dict(),
            "config": self.config,            # 方便复现实验
            "param_table": self.param.table   # 若 ParameterManager 支持
        }
        if extra_info is not None:
            payload["extra_info"] = extra_info

        torch.save(payload, ckpt_path)
        print(f"[PPOAgent] Model saved → {ckpt_path}")

    def load(self, ckpt_path: str, map_location: str | torch.device | None = None) -> None:
        """
        Restore weights & optimizers.

        Args:
            ckpt_path (str): 与 save 对应的存档路径
            map_location (str | torch.device, optional): 设备映射
        """
        checkpoint = torch.load(ckpt_path, map_location=map_location or self.device)

        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])

        # 如果只做推理，可跳过以下两行
        self.optimizer_actor.load_state_dict(checkpoint["optimizer_actor_state"])
        self.optimizer_critic.load_state_dict(checkpoint["optimizer_critic_state"])

        # 若要同步 config / param，可按需覆盖
        print(f"[PPOAgent] Model loaded ← {ckpt_path}")

def batch_generator(states, actions, old_log_probs, advantages, returns, mini_batch_size):
    total_samples = len(states)
    indices = np.arange(total_samples)
    np.random.shuffle(indices)

    for start in range(0, total_samples, mini_batch_size):
        end = start + mini_batch_size
        batch_idx = indices[start:end]
        # 根据索引提取对应的 mini-batch 数据
        yield states[batch_idx], actions[batch_idx], old_log_probs[batch_idx], advantages[batch_idx], returns[batch_idx]


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

