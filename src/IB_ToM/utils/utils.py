import random
from typing import Any, List

from overcooked_ai_py.mdp.overcooked_env import (
    DEFAULT_ENV_PARAMS,
    OvercookedEnv,
)
from overcooked_ai_py.mdp.overcooked_mdp import (
    ObjectState,
    OvercookedGridworld,
    OvercookedState,
    PlayerState,
    Recipe,
    SoupState,
)
import gymnasium

import numpy as np

import torch

def env_maker(env_name="Overcooked-v0", layout_name="cramped_room"):
    base_mdp = OvercookedGridworld.from_layout_name(layout_name)
    base_env = OvercookedEnv.from_mdp(base_mdp, **DEFAULT_ENV_PARAMS, info_level=0)
    env = gymnasium.make(env_name, base_env=base_env, featurize_fn=base_env.featurize_state_mdp)
    return env

def overcooked_obs_process(state):
    """
    method for process the complicated state of overcooked env
    """
    assert isinstance(state, dict)
    # state是一个dict，三个值分别是：
    #   both_agent_obs: tuple，分别表示双方的obs
    #   overcooked_state: MDP视角下的state
    #   other_agent_env_idx: 另一个智能体的编号是什么
    # 其实第三项不重要，other_agent_env_idx是随机的，这是为了增加随机性。只要保证both_agent_obs中ego和partner的顺序不变就可以。
    both_agent_obs = state['both_agent_obs']
    obs_ego = both_agent_obs[0]
    obs_partner = both_agent_obs[1]
    return obs_ego, obs_partner

def print_generation_banner(generation_num):
    banner = f"""
╔══════════════════════════════════════════════════════════╗
║                  STARTING SELF-PLAY ROUND                ║
║                     GENERATION {generation_num:<4}                      ║
╚══════════════════════════════════════════════════════════╝
"""
    print(banner)

class ParameterManager:
    def __init__(self, config):
        self.config = config

    def get(self, key, default_value=None):
        return self.config.get(key, default_value)


class ReplayBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.clear()

    def push(self, state, action, reward, next_state, done, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []

    def get_batch(self):
        return (np.array(self.states),
                np.array(self.actions),
                np.array(self.rewards),
                np.array(self.next_states),
                np.array(self.dones),
                np.array(self.log_probs))

def batch_generator(states, actions, old_log_probs, advantages, returns, mini_batch_size):
    total_samples = len(states)
    indices = np.arange(total_samples)
    np.random.shuffle(indices)

    for start in range(0, total_samples, mini_batch_size):
        end = start + mini_batch_size
        batch_idx = indices[start:end]
        # 根据索引提取对应的 mini-batch 数据
        yield states[batch_idx], actions[batch_idx], old_log_probs[batch_idx], advantages[batch_idx], returns[batch_idx]

def one_rollout(env, agent, agent_partner, state_norm, obs_ego, obs_partner):
    """
    Collect one rollout for multi-env into the buffer

    @param env: The environment in which both agents interact. It must support `reset()` and `step()` functions.
    @param agent_ego: The learning agent (ego agent) whose policy is being optimized.
    @param agent_partner: The partner agent used for self-play, providing diverse training interactions for the ego agent.
    @param state_norm: A normalization function applied to states before feeding them into the policy networks.
    @param obs_ego: obs of ego from reset or step, already been normed.
    @param obs_partner: obs of partner from reset or step, already been normed.

    """
    action_ego, log_prob_ego = agent.select_action(obs_ego)
    action_partner, log_prob_partner = agent_partner.select_action(obs_partner)
    next_state, reward, done, _ = env.step([action_ego, action_partner])

    next_obs_ego, next_obs_partner = overcooked_obs_process(next_state)
    next_obs_ego = state_norm(next_obs_ego)
    next_obs_partner = state_norm(next_obs_partner)
    one_traj = (obs_ego, action_ego, reward, next_obs_ego, done, log_prob_ego)
    return next_obs_ego, next_obs_partner, reward, done, one_traj

def tom_one_rollout(env, agent, agent_partner, state_norm, obs_ego, obs_partner, tom_latent):

    action_ego, log_prob_ego = agent.tom_select_action(obs_ego, tom_latent)
    action_partner, log_prob_partner = agent_partner.tom_select_action(obs_partner, tom_latent)
    next_state, reward, done, _ = env.step([action_ego, action_partner])

    next_obs_ego, next_obs_partner = overcooked_obs_process(next_state)
    next_obs_ego = state_norm(next_obs_ego)
    next_obs_partner = state_norm(next_obs_partner)
    one_traj = (obs_ego, action_ego, reward, next_obs_ego, done, log_prob_ego)
    return next_obs_ego, next_obs_partner, action_partner, reward, done, one_traj

@torch.no_grad()
def evaluate_policy(env, agent_ego, partner, steps, state_norm):
    episode_rewards = []
    state = env.reset()
    obs_ego, obs_partner = overcooked_obs_process(state)
    obs_ego = state_norm(obs_ego)
    obs_partner = state_norm(obs_partner)
    done = False
    ep_reward = 0
    for _ in range(steps):
        next_obs_ego, next_obs_partner, reward, done, _ = one_rollout(
            env, agent_ego, partner, state_norm, obs_ego, obs_partner
        )
        ep_reward += reward
        obs_ego = next_obs_ego
        obs_partner = next_obs_partner
        if done:
            state = env.reset()
            obs_ego, obs_partner = overcooked_obs_process(state)
            obs_ego = state_norm(obs_ego)
            obs_partner = state_norm(obs_partner)
            done = False
            episode_rewards.append(ep_reward)
            ep_reward = 0
    return episode_rewards

@torch.no_grad()
def tom_evaluate_policy(env, agent_ego, partner, steps, state_norm, tom_model, dataset):
    episode_rewards = []
    state = env.reset()
    obs_ego, obs_partner = overcooked_obs_process(state)
    obs_ego = state_norm(obs_ego)
    obs_partner = state_norm(obs_partner)
    done = False
    ep_reward = 0
    dataset_item = []
    for _ in range(steps):
        # todo: '32' needed to be replaced by param
        tom_latent, _ = tom_model(dataset[0:32, :, :])
        next_obs_ego, next_obs_partner, action_partner, reward, done, _ = tom_one_rollout(
            env, agent_ego, partner, state_norm, obs_ego, obs_partner, tom_latent.mean(dim=1).squeeze(0)
        )
        dataset_item.append(
            torch.concat(
                [
                    torch.FloatTensor(obs_partner),
                    torch.FloatTensor([action_partner])
                ]
            )
        )
        # todo: '2' needed to be replaced by seq_len param
        if len(dataset_item) == 2:
            dataset = insert_dataset(dataset, dataset_item)
            dataset_item = []

        ep_reward += reward
        obs_ego = next_obs_ego
        obs_partner = next_obs_partner
        if done:
            state = env.reset()
            obs_ego, obs_partner = overcooked_obs_process(state)
            obs_ego = state_norm(obs_ego)
            obs_partner = state_norm(obs_partner)
            done = False
            episode_rewards.append(ep_reward)
            ep_reward = 0
    return episode_rewards

def build_eval_agent(env, eval_agent_choice="Random"):
    if eval_agent_choice == "Random":
        from src.IB_ToM.ppo_algo.agents import RandomAgent
        return RandomAgent(env)
    elif eval_agent_choice == "Human":
        from src.IB_ToM.ppo_algo.agents import HumanAgent
        return HumanAgent()
    else:
        raise ValueError("Invalid eval_agent_choice")

def insert_dataset(dataset, dataset_item: list):
    # dataset_item中是一个seq长度的s-a pair，以tensor格式
    # 需要将这个list插入到dataset中
    # list的长度是seq_len，每一项的长度是state_dim+action_dim
    # 首先将其转换为tensor(10, 63)
    # 然后将其插入到dataset中
    # 尾插法，后端的数据是新的
    dataset_item = torch.stack(dataset_item).to('cuda')
    if dataset_item.dim() == 2:
        dataset_item = dataset_item.unsqueeze(0)
    dataset = torch.concat((dataset, dataset_item), dim=0)
    return dataset

def random_choice(
    candidates: List[Any],
    strategy: str = "recency",
    recency_bias: float = 0.3,
) -> Any:
    """
    选择合作伙伴策略（partner）。

    Parameters
    ----------
    candidates : list
        策略对象列表（如 Policy、Agent 等）。
    strategy : {"uniform", "recency"}
        - "uniform" : 每个策略等概率抽取。
        - "recency":  最近策略出现概率为 `recency_bias`，
                      其余概率按指数衰减分配给更旧策略。
    recency_bias : float, optional
        当 strategy == "recency" 时，最新策略被选中的目标概率。
        0.5–0.9 通常效果较好。

    Returns
    -------
    Any
        被选中的策略对象。
    """
    if not candidates:
        raise ValueError("random_choice: `candidates` 不能为空。")

    # 只有一个候选时直接返回
    if len(candidates) == 1 or strategy == "uniform":
        return random.choice(candidates)

    if strategy == "recency":
        n = len(candidates)
        # 最旧 → 最新 的权重（指数衰减）
        # w_i ∝ (1 - recency_bias)^(n-1-i)       i = 0 … n-2
        # w_{n-1} = recency_bias
        weights = [(1.0 - recency_bias) ** (n - 1 - i) for i in range(n)]
        weights[-1] = recency_bias
        # 归一化
        total = sum(weights)
        weights = [w / total for w in weights]
        # random.choices 支持按概率抽样
        return random.choices(candidates, weights=weights, k=1)[0]

    raise ValueError(f"random_choice: 未知 strategy='{strategy}'")

def modify_tuple(t: tuple, index: int, value) -> tuple:
    """
    Return a new tuple with one element modified at the specified index.

    Parameters:
    - t: original tuple
    - index: index to modify (supports negative indexing)
    - value: new value to insert

    Returns:
    - new tuple with value at index replaced
    """
    if not isinstance(t, tuple):
        raise TypeError("Input must be a tuple.")
    if not (-len(t) <= index < len(t)):
        raise IndexError("Index out of range.")

    t_list = list(t)
    t_list[index] = value
    return tuple(t_list)
