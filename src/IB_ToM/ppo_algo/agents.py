import os
from collections import deque
from typing import Optional, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

import numpy as np

from torch.utils.tensorboard import SummaryWriter

from networks import Actor, Critic, ConActor, ToMActor
from src.IB_ToM.utils.utils import ParameterManager
from src.IB_ToM.utils.utils import batch_generator

import torch.optim as optim

class RandomAgent:
    def __init__(self, env):
        self.env = env

    def select_action(self, state):
        temp = 0
        return self.env.action_space.sample(), temp

    def tom_select_action(self, state, tom_latent):
        return self.env.action_space.sample(), 0


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

    def get_policy_entropy(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        probs = self.actor(state_tensor)
        dist = torch.distributions.Categorical(probs)
        return dist.entropy().mean()

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
            for state, action, old_log_prob, advantage, return_ in batch_generator(
                    states, actions, old_log_probs, advantages, returns, self.param.get("mini_batch_size")):
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

    def save(self, ckpt_path: str, extra_info: Optional[Dict[str, Any]] = None) -> None:
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

    def load(self, ckpt_path: str, map_location: Optional[Union[str, torch.device]] = None) -> None:
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

class ToMPPOAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, config):
        self.config = config
        self.param = ParameterManager(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = ToMActor(state_dim, hidden_dim, action_dim,
                              self.param.get("tom_input_size"),
                              self.param.get("tom_hidden_size")).to(self.device)
        self.critic = Critic(state_dim, hidden_dim).to(self.device)

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.param.get("lr_actor"))
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.param.get("lr_critic"))

    def tom_select_action(self, state, tom_latent):
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        probs = self.actor(state_tensor, tom_latent)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()

    def update(self, buffer, tom_model):
        states, actions, rewards, next_states, dones, log_probs = buffer.get_batch()
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        old_log_probs = torch.tensor(log_probs, dtype=torch.float32).to(self.device)

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

        # prepare for tom_latent
        partner_current_sa = deque(maxlen=2)

        for _ in range(self.param.get("ppo_epochs")):
            for state, action, old_log_prob, advantage, return_ in batch_generator(
                    states, actions, old_log_probs, advantages, returns, self.param.get("mini_batch_size")):
                # 使用队列，保证sa永远是两个
                partner_current_sa.append([state, action])
                # todo: set ToM Net here!
                if len(partner_current_sa) != 2:
                    tom_latent = torch.zeros(self.param.get("mini_batch_size"), self.param.get("seq_len"), len(state[0]) + 1).to('cuda')
                    tom_latent, _ = tom_model(tom_latent)
                else:
                    merged = []
                    for s, a in partner_current_sa:
                        a = a.unsqueeze(-1)
                        merged_step = torch.cat([s, a], dim=1)
                        merged.append(merged_step)
                    tom_latent, _ = tom_model(torch.stack(merged, dim=1))

                probs = self.actor(state, tom_latent.squeeze(0))
                dist = torch.distributions.Categorical(probs)
                new_log_prob = dist.log_prob(action)
                ratio = torch.exp(new_log_prob - old_log_prob)

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.param.get("clip_epsilon"),
                                  1 + self.param.get("clip_epsilon")) * advantage

                actor_loss = (-torch.min(surr1, surr2) - self.param.get("entropy_loss_coef") * dist.entropy()).mean()

                value = self.critic(state).squeeze(-1).float()

                value_loss = F.mse_loss(value, return_)

                critic_loss = 0.5 * self.param.get("value_loss_coef") * value_loss.mean()

                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.param.get("max_grad_norm"))
                self.optimizer_actor.step()

                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.param.get("max_grad_norm"))
                self.optimizer_critic.step()

        buffer.clear()
        return actor_loss.item(), critic_loss.item()

    def save(self, ckpt_path: str, extra_info: Optional[Dict[str, Any]] = None) -> None:
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

    def load(self, ckpt_path: str, map_location: Optional[Union[str, torch.device]] = None) -> None:
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
