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