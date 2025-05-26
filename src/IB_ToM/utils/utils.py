import random
from typing import Any, List

from torch.utils.data import TensorDataset, DataLoader

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

MAX_DATASET_NUM = 3200


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

    def set(self, key, value):
        self.config[key] = value


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
        tom_latent, _ = tom_model(dataset[-32:, :, :])
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
    if len(dataset) > MAX_DATASET_NUM:
        dataset = dataset[-MAX_DATASET_NUM:]
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


def mep_compute_avg_entropy(population, obs, tom_latent=None):
    if tom_latent is not None:
        all_probs = [agent.tom_get_policy_probs(obs, tom_latent) for agent in population]
    else:
        all_probs = [agent.get_policy_probs(obs) for agent in population]
    probs_stack = torch.stack(all_probs, dim=0)
    mean_probs = probs_stack.mean(dim=0).squeeze(0)
    mean_dist = torch.distributions.Categorical(mean_probs)
    avg_ent = mean_dist.entropy().item()
    return avg_ent


#############################
#                           #
#       Utiles for BC       #
#                           #
#############################

def bc_process_dataset(
        bc_data_addr: str,
        param: ParameterManager,
        bc_use_lstm: bool = False,
        train_mode: bool = True):
    trajs = torch.load(bc_data_addr, weights_only=False)
    all_states = []
    all_actions = []
    if bc_use_lstm:
        bc_seq_len = param.get("bc_seq_len")
        for ep_states, ep_actions in zip(trajs['ep_states'], trajs['ep_actions']):
            for i in range(len(ep_states) - bc_seq_len + 1):
                seq_states = ep_states[i:i + bc_seq_len]
                seq_actions = ep_actions[i + bc_seq_len - 1]
                all_states.append(seq_states)
                all_actions.append(seq_actions)
        state_tensor = [torch.tensor(seq_states, dtype=torch.float32).to('cuda') for seq_states in all_states]
        action_tensor = [torch.tensor(seq_actions, dtype=torch.long).to('cuda') for seq_actions in all_actions]
        dataset = TensorDataset(torch.stack(state_tensor), torch.stack(action_tensor))
        dataloader = DataLoader(dataset, batch_size=param.get("bc_batch_size"), shuffle=train_mode)

    else:
        for ep_states, ep_actions in zip(trajs['ep_states'], trajs['ep_actions']):
            all_states.extend(ep_states)
            all_actions.extend(ep_actions)
        state_tensor = torch.tensor(all_states, dtype=torch.float32).to('cuda')
        action_tensor = torch.tensor(all_actions, dtype=torch.long).to('cuda')
        dataset = TensorDataset(state_tensor, action_tensor)
        dataloader = DataLoader(dataset, batch_size=param.get("bc_batch_size"), shuffle=train_mode)
    return dataloader


if __name__ == "__main__":
    config = {
        'lr_actor': 3e-4,
        'lr_critic': 3e-4,
        'gamma': 0.99,
        'lambda': 0.95,
        'clip_epsilon': 0.2,
        'ppo_epochs': 10,
        'batch_size': 4096,
        'entropy_loss_coef': 0.0,
        'value_loss_coef': 1.0,
        'max_grad_norm': 0.5,
        'max_episodes': 1000,
        'max_timesteps': 5000000,
        'mini_batch_size': 64,
        'tom_input_size': 64,
        'tom_hidden_size': 64,
        'continuous': False,
        'target_reward': 180,
        'partners_num': 5,
        'checkpoint': 1e6,
        'bc_batch_size': 64,
        'bc_seq_len': 5,
    }
    param = ParameterManager(config)

    bc_process_dataset('../ppo_algo/bc/human_data/2019_hh_trials_all.pt', param, True)
