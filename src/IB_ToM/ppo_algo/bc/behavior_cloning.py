"""
behavior_cloning.py

This file is used to train behavior cloning (BC) agents, including both MLP-based and LSTM-based models,
and to perform simple evaluations.
"""

from collections import deque

from prompt_toolkit.shortcuts import print_container

from IB_ToM.ppo_algo.bc.bc_agent import BCMLPAgent, BCLSTMAgent
from IB_ToM.utils.utils import ParameterManager, env_maker, overcooked_obs_process

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
        'bc_batch_size': 128,
        'bc_seq_len': 10,
        'bc_epoch': 4000,
        'bc_data_addr_train': "./human_data/2019_hh_trials_train.pt",
        'bc_data_addr_test': "./human_data/2019_hh_trials_test.pt",
    }

    param = ParameterManager(config)
    layouts_2019 = ["cramped_room",
                    "asymmetric_advantages",
                    "coordination_ring",
                    "random0",
                    "random3"]
    layouts_2020 = ["asymmetric_advantages_tomato",
                    "counter_circuit",
                    "cramped_corridor",
                    "inverse_marshmallow_experiment",
                    "marshmallow_experiment",
                    "marshmallow_experiment_coordination",
                    "soup_coordination",
                    "you_shall_not_pass"]
    for layout in layouts_2020:
        param.set('bc_data_addr_train', f'./human_data/{layout}/2020_hh_trials_train_{layout}.pt')
        param.set('bc_data_addr_test', f'./human_data/{layout}/2020_hh_trials_test_{layout}.pt')

        env = env_maker("Overcooked-v0", layout_name=layout)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        bc_lstm_agent = BCLSTMAgent(state_dim, action_dim, 128, config)
        # bc_lstm_agent.load("./trained_models/bc_lstm_agent.pth")
        print(f"Start training lstm agent on map from: {bc_lstm_agent.param.get('bc_data_addr_train')}.")
        bc_lstm_agent.update()
        bc_lstm_agent.evaluation()

        bc_lstm_agent.save(f"./trained_models/bc_lstm_agent_{layout}.pth")
        print(f"Saving lstm agent of {layout}...")

        # # evaluation by env
        # epoch = 100
        # for i in range(epoch):
        #     state = env.reset()
        #     obs_ego, obs_partner = overcooked_obs_process(state)
        #     ep_reward = 0
        #     while True:
        #         action_ego, _ = bc_lstm_agent.select_action(obs_ego)
        #         action_partner, _ = bc_lstm_agent.select_action(obs_partner)
        #         next_state, reward, done, _ = env.step([action_ego, action_partner])
        #         next_obs_ego, next_obs_partner = overcooked_obs_process(next_state)
        #         ep_reward += reward
        #         obs_ego = next_obs_ego
        #         obs_partner = next_obs_partner
        #         if done:
        #             break
        #     print(f"epoch:{i}, total_reward: {ep_reward}")

    # for i in range(epoch):
    #     state = env.reset()
    #     ep_reward = 0
    #     while True:
    #         action_ego = env.action_space.sample()
    #         action_partner = env.action_space.sample()
    #         next_state, reward, done, _ = env.step([action_ego, action_partner])
    #         ep_reward += reward
    #         if done:
    #             break
    #     print(f"epoch:{i}, total_reward: {ep_reward}")
    #
    #
    # env.close()

    # bc_mlp_agent = BCMLPAgent(state_dim, action_dim, 128, config)
    #
    # bc_mlp_agent.update()
    #
    # bc_mlp_agent.evaluation()
    #
    # bc_mlp_agent.save("./trained_models/bc_mlp_agent.pth")
    # epoch = 100
    # for i in range(epoch):
    #     state = env.reset()
    #     obs_ego, obs_partner = overcooked_obs_process(state)
    #     ep_reward = 0
    #     while True:
    #         action_ego, _ = bc_mlp_agent.select_action(obs_ego)
    #         action_partner, _ = bc_mlp_agent.select_action(obs_partner)
    #         next_state, reward, done, _ = env.step([action_ego, action_partner])
    #         next_obs_ego, next_obs_partner = overcooked_obs_process(next_state)
    #         ep_reward += reward
    #         obs_ego = next_obs_ego
    #         obs_partner = next_obs_partner
    #         if done:
    #             break
    #     print(f"epoch:{i}, total_reward: {ep_reward}")
    #
    # env.close()
