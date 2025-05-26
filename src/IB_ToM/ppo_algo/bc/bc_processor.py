import json
import os

import pandas as pd
import torch

from human_aware_rl.human.process_dataframes import get_trajs_from_data
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import (
    ObjectState,
    OvercookedGridworld,
    OvercookedState,
    PlayerState,
)


def main():
    """
    Behavior Cloning Dataset Structure
    ==================================

    This dataset is a Python dictionary named `trajs`, representing human gameplay trajectories
    for behavior cloning training. It contains 126 episodes, each corresponding to one complete
    game session. Each episode consists of step-wise sequences for states, actions, and other metadata.

    Structure:
    ----------
    trajs: dict
    ├── ep_states     : List[List[state]]     -- List of episodes, each a list of per-step state vectors.
    ├── ep_actions    : List[List[action]]    -- List of episodes, each a list of actions taken by the human.
    ├── ep_rewards    : List[List[float]]     -- List of episodes, each a list of per-step rewards.
    ├── ep_dones      : List[List[bool]]      -- List of episodes, each a list of done flags (True if terminal).
    ├── ep_infos      : List[List[dict]]      -- (Optional) List of metadata dicts for each step.
    ├── ep_returns    : List[float]           -- Total return (sum of rewards) for each episode.
    ├── ep_lengths    : List[int]             -- Number of steps in each episode.

    Unused or optional metadata (not needed for BC training):
    ├── mdp_params    : Any                   -- MDP configuration used during data collection.
    ├── env_params    : Any                   -- Environment settings such as map layout, task config, etc.
    ├── metadatas     : Any                   -- Additional trajectory metadata.

    Example:
    --------
    - len(trajs['ep_states']) == 126            # 126 episodes
    - len(trajs['ep_states'][0]) == 1204        # First episode has 1204 steps
    - trajs['ep_states'][0][t] ∈ ℝ^n            # State at time t is an n-dimensional float vector
    """

    from pathlib import Path

    folder = Path('../../../human_aware_rl/static/human_data/cleaned/')
    file_names = [f.stem for f in folder.iterdir() if f.is_file()]

    for file_name in file_names:
        df = pd.read_pickle(f'../../../human_aware_rl/static/human_data/cleaned/{file_name}.pickle')
        layouts = df['layout_name'].unique().tolist()
        for layout in layouts:
            trajs, _ = get_trajs_from_data(
                f'../../../human_aware_rl/static/human_data/cleaned/{file_name}.pickle', [layout])
            os.makedirs(f'./human_data/{layout}', exist_ok=True)
            torch.save(trajs, f'./human_data/{layout}/{file_name}_{layout}.pt')
        print(f'{file_name} loaded.')
    print('All files loaded.')


    # trajs_after = torch.load('./human_data/2019_hh_trials_all.pt', weights_only=False)
    # print("test")


if __name__ == "__main__":
    main()
