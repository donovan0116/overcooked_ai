import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from IB_ToM.ppo_algo.bc.bc_agent import BCMLPAgent, BCLSTMAgent
from IB_ToM.ppo_algo.ppo import Normalization
from IB_ToM.tom_net import ToMNet, make_fake_dataset, train_step1, train_step2, insert_dataset, train_tom_model
from IB_ToM.utils.utils import ParameterManager, env_maker, overcooked_obs_process
import torch

def compute_tsne_data(epoch_, max_episode_steps_, env_, state_norm_, bc_lstm_agent_, dataset_, tom_model_, seq_len):
    for i in range(epoch_):
        state = env_.reset()
        obs_ego, obs_partner = overcooked_obs_process(state)
        obs_ego = state_norm_(obs_ego)
        obs_partner = state_norm_(obs_partner)
        dataset_item = []
        ep_reward = 0
        for _ in range(max_episode_steps_):
            action_ego, _ = bc_lstm_agent_.select_action(obs_ego)
            action_partner, _ = bc_lstm_agent_.select_action(obs_partner)
            next_state, reward, done, _ = env_.step([action_ego, action_partner])
            next_obs_ego, next_obs_partner = overcooked_obs_process(next_state)

            dataset_item.append(
                torch.concat(
                    [
                        torch.FloatTensor(obs_partner),
                        torch.FloatTensor([action_partner])
                    ]
                )
            )
            if len(dataset_item) == seq_len:
                dataset_ = insert_dataset(dataset_, dataset_item)
                dataset_item = []

            ep_reward += reward
            obs_ego = next_obs_ego
            obs_partner = next_obs_partner
            if done:
                state = env_.reset()
                obs_ego, obs_partner = overcooked_obs_process(state)
                obs_ego = state_norm_(obs_ego)
                obs_partner = state_norm_(obs_partner)
        train_tom_model(tom_model_, dataset_, param)
    inferred_latent_ = []
    with torch.no_grad():
        for i in range(len(dataset_)):
            latent, _ = tom_model_(dataset_[i:i + 1, :, :])
            inferred_latent_.append(latent.squeeze(0).squeeze(0))
    inferred_latent_ = torch.stack(inferred_latent_)
    return inferred_latent_, dataset_


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
        'bc_epoch': 10,
        'bc_data_addr_train': "./human_data/2019_hh_trials_train.pt",
        'bc_data_addr_test': "./human_data/2019_hh_trials_test.pt",
        'seq_len': 10,
    }

    # print(os.getcwd())

    param = ParameterManager(config)
    layout = "cramped_corridor"
    param.set('bc_data_addr_train', f'./human_data/{layout}/2020_hh_trials_train_{layout}.pt')
    param.set('bc_data_addr_test', f'./human_data/{layout}/2020_hh_trials_test_{layout}.pt')
    env = env_maker("Overcooked-v0", layout_name=layout)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    bc_lstm_agent = BCLSTMAgent(state_dim, action_dim, 128, config)
    # bc_lstm_agent.update()
    # bc_lstm_agent.evaluation()
    # bc_lstm_agent.load("../bc/trained_models/bc_lstm_agent.pth")
    bc_lstm_agent.load(f"../bc/trained_models/bc_lstm_agent_{layout}.pth")

    state_norm = Normalization(state_dim)

    tom_model = ToMNet(input_size=state_dim + 1,
                       hidden_size=[64, 256, state_dim + 1],
                       output_size=(state_dim + 1) * param.get("seq_len")).to('cuda')

    fake_dataset = make_fake_dataset(env, param.get("mini_batch_size") * 10, param.get("seq_len"))
    train_step1(tom_model, fake_dataset, param.get("mini_batch_size"), 10)
    train_step2(tom_model, fake_dataset, param.get("mini_batch_size"), 10)
    dataset = fake_dataset
    epoch = 20
    max_episode_steps = 4096

    inferred_latent, dataset = compute_tsne_data(epoch, max_episode_steps, env, state_norm, bc_lstm_agent, dataset,
                                                 tom_model, param.get("seq_len"))

    # os.makedirs("./tsne_data", exist_ok=True)
    # torch.save(inferred_latent, f"./tsne_data/inferred_latent_{layout}.pt")
    # torch.save(dataset, f"./tsne_data/dataset_{layout}.pt")
    dataset_mean = dataset[:, 1, :]

    sa_np = dataset_mean.cpu().numpy()
    z_np = inferred_latent.cpu().numpy()
    np.save(f'./tsne_data/sa_seq2_{layout}.npy', sa_np)
    np.save(f'./tsne_data/latent_z_{layout}.npy', z_np)
    print("Saving tsne data...")

    sa_np_reduced = PCA(n_components=64).fit_transform(sa_np)
    x = np.concatenate([sa_np_reduced, z_np], axis=0)

    tsne = TSNE(n_components=2, init='pca', random_state=0)
    x_tsne = tsne.fit_transform(x)
    sa_tsne = x_tsne[:len(sa_np)]
    z_tsne = x_tsne[len(sa_np):]

    plt.figure(figsize=(7, 6))
    plt.scatter(sa_tsne[:, 0], sa_tsne[:, 1], c='blue', marker='o', s=12, alpha=0.6, label='(s,a) mean')
    plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c='red', marker='*', s=30, alpha=0.6, label='ToM latent z')

    plt.title("t-SNE of (s,a) Sequences and ToM Latents")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



