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

def test():
    base_mdp = OvercookedGridworld.from_layout_name("cramped_room")
    base_env = OvercookedEnv.from_mdp(base_mdp, **DEFAULT_ENV_PARAMS, info_level=0)
    env = gymnasium.make("Overcooked-v0", base_env=base_env, featurize_fn=base_env.featurize_state_mdp)
    print(env)

    epoch = 10

    a = env.reset()
    both_agent_obs = a['both_agent_obs']
    overcooked_state = a['overcooked_state']
    other_agent_env_idx = a['other_agent_env_idx']
    print(a)
    for i in range(epoch):
        action = env.action_space.sample()
        obs, reward, done, info = env.step([action, action])
        print(obs, reward, done, info)

    env.close()

if __name__ == "__main__":
    test()