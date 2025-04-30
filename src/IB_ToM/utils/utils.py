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
        action_ego, log_prob_ego = agent_ego.select_action(obs_ego)
        action_partner, _ = partner.select_action(obs_partner)
        next_state, reward, done, _ = env.step([action_ego, action_partner])
        next_obs_ego, next_obs_partner = overcooked_obs_process(next_state)
        next_obs_ego = state_norm(next_obs_ego)
        next_obs_partner = state_norm(next_obs_partner)
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

