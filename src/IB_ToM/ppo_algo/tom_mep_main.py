from copy import deepcopy

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from IB_ToM.ppo_algo.agents import PPOAgent, ToMPPOAgent
from IB_ToM.ppo_algo.ppo import Normalization, get_run_log_dir
from IB_ToM.ppo_algo.self_play_main import sp_collect_samples
from IB_ToM.ppo_algo.tom_fcp_main import tom_fcp_collect_samples
from IB_ToM.tom_net import ToMNet, make_fake_dataset, train_step1, train_step2, train_tom_model
from IB_ToM.utils.utils import ParameterManager, env_maker, ReplayBuffer, overcooked_obs_process, \
    build_eval_agent, evaluate_policy, print_generation_banner, modify_tuple, tom_one_rollout, insert_dataset, \
    tom_evaluate_policy, mep_compute_avg_entropy


def tom_mep_collect_samples(env, agent_ego, agent_partner, buffer, batch_size, state_norm, population, param, tom_model, dataset):
    """
    Similar to the function sp_collect_samples(), which add the augment reward compute.
    """
    steps = 0
    episode_rewards = []
    state = env.reset()
    obs_ego, obs_partner = overcooked_obs_process(state)
    obs_ego = state_norm(obs_ego)
    obs_partner = state_norm(obs_partner)
    done = False
    ep_reward = 0
    dataset_item = []
    while steps < batch_size:
        # todo: '32' needed to be replaced by param
        tom_latent, _ = tom_model(dataset[-32:, :, :])
        next_obs_ego, next_obs_partner, action_partner, reward, done, one_traj = tom_one_rollout(
            env, agent_ego, agent_partner, state_norm, obs_ego, obs_partner, tom_latent.mean(dim=1).squeeze(0)
        )
        # compute augment reward
        avg_ent = mep_compute_avg_entropy(population, obs_partner, tom_latent.mean(dim=1).squeeze(0))

        augment_reward = (avg_ent * param.get("alpha") + reward)
        one_traj = modify_tuple(one_traj, 2, augment_reward)
        reward = augment_reward

        buffer.push(*one_traj)

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
        steps += 1
        if done:
            state = env.reset()
            obs_ego, obs_partner = overcooked_obs_process(state)
            obs_ego = state_norm(obs_ego)
            obs_partner = state_norm(obs_partner)
            done = False
            episode_rewards.append(ep_reward)
            ep_reward = 0

    return steps, episode_rewards

def tom_train_mep_population(env, pop, param, state_norm, tom_model, dataset):
    for i in range(param.get("pop_size")):
        agent_ego = pop[i]
        agent_partner = deepcopy(agent_ego)
        buffer = ReplayBuffer()
        total_timesteps = 0
        all_episode_rewards = []

        while total_timesteps < param.get("max_timesteps"):
            steps, episode_rewards = tom_mep_collect_samples(
                env, agent_ego, agent_partner, buffer, param.get("batch_size"), state_norm, pop, param, tom_model, dataset
            )
            total_timesteps += steps
            all_episode_rewards.extend(episode_rewards)

            agent_ego.update(buffer, tom_model)
            train_tom_model(tom_model, dataset, param)
            agent_partner = deepcopy(agent_ego)
            if len(all_episode_rewards) >= 10:
                avg_reward = np.mean(all_episode_rewards[-10:])
                if avg_reward > param.get("target_reward"):
                    pop[i] = deepcopy(agent_ego)
                    print(f"Agent {i} in the population was trained.")
                    break
    return pop


def tom_rollout_and_evaluate(env, agent_br, partner, param, state_norm, tom_model, dataset):
    eval_times = param.get("evaluate_times")
    max_episodes = param.get("max_episodes")
    returns = []
    for _ in range(eval_times):
        ep_reward = 0
        state = env.reset()
        obs_ego, obs_partner = overcooked_obs_process(state)
        obs_ego = state_norm(obs_ego)
        obs_partner = state_norm(obs_partner)
        dataset_item = []
        for _ in range(max_episodes):
            tom_latent, _ = tom_model(dataset[-32:, :, :])
            next_obs_ego, next_obs_partner, action_partner, reward, done, _ = tom_one_rollout(
                env, agent_br, partner, state_norm, obs_ego, obs_partner, tom_latent.mean(dim=1).squeeze(0)
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
                break
        returns.append(ep_reward)

    return sum(returns) / len(returns)


def tom_mep_prior_choice(env, agent_br, pop, param, state_norm, tom_model, dataset):
    beta = param.get("beta_mep")
    returns = []
    for partner in pop:
        returns.append(tom_rollout_and_evaluate(env, agent_br, partner, param, state_norm, tom_model, dataset))

    inv_returns = 1.0 / (np.array(returns) + 1e-8)
    sorted_indices = np.argsort(inv_returns)
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(1, len(returns) + 1)
    probs = ranks ** beta
    probs = probs / np.sum(probs)
    selected_index = np.random.choice(len(pop), p=probs)
    selected_agent = pop[selected_index]
    return selected_agent


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
        'max_timesteps': 5000000,
        'mini_batch_size': 64,
        'tom_input_size': 64,
        'tom_hidden_size': 64,
        'continuous': False,
        'target_reward': 180,
        'seq_len': 2,
        # mep settings
        'pop_size': 5,
        'alpha': 0.01,
        'beta_mep': 3,
        'evaluate_times': 100,
    }

    param = ParameterManager(config)

    layout = "cramped_room"
    env = env_maker("Overcooked-v0", layout_name=layout)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    state_norm = Normalization(state_dim)

    # build tom model and do the initial training
    tom_model = ToMNet(input_size=state_dim + 1,
                       hidden_size=[64, 256, state_dim + 1],
                       output_size=(state_dim + 1) * param.get("seq_len")).to('cuda')

    fake_dataset = make_fake_dataset(env, param.get("mini_batch_size") * 10, param.get("seq_len"))
    train_step1(tom_model, fake_dataset, param.get("mini_batch_size"), 10)
    train_step2(tom_model, fake_dataset, param.get("mini_batch_size"), 10)
    dataset = fake_dataset

    # Stage 1: train population of ME
    pop = []
    for _ in range(param.get("pop_size")):
        agent = ToMPPOAgent(state_dim, action_dim, 128, param)
        pop.append(agent)

    pop = tom_train_mep_population(env, pop, param, state_norm, tom_model, dataset)
    # Stage 2: train BR agent from MEP population (for 10 times).
    agent_ego = ToMPPOAgent(state_dim, action_dim, 128, param)
    zsc_agent = build_eval_agent(env, config, "Random")

    buffer = ReplayBuffer()

    generation = 0

    while True:
        log_dir = get_run_log_dir('./logs/tensorboard_logs/ppo_test', 'generation')
        writer = SummaryWriter(log_dir=log_dir)
        generation += 1
        print_generation_banner(generation)
        total_timesteps = 0
        all_episode_rewards = []
        all_episode_rewards_eval = []

        while total_timesteps < param.get("max_timesteps"):
            agent_partner = tom_mep_prior_choice(env, agent_ego, pop, param, state_norm, tom_model, dataset)
            #  呃呃借用一下fcp的采样函数，因为mep在训练种群时候的采样函数把名字占了，而mep训练br的采样和fcp又完全一致...
            steps, episode_rewards = tom_fcp_collect_samples(
                env, agent_ego, agent_partner, buffer, param.get("batch_size"), state_norm, tom_model, dataset)
            total_timesteps += steps
            all_episode_rewards.extend(episode_rewards)

            # train
            actor_loss, critic_loss = agent_ego.update(buffer, tom_model)
            writer.add_scalar('Loss/Actor', actor_loss, total_timesteps)
            writer.add_scalar('Loss/Critic', critic_loss, total_timesteps)
            train_tom_model(tom_model, dataset, param)
            # todo: change partner in eval to a human policy as zero_shot
            episode_rewards_eval = tom_evaluate_policy(env, agent_ego, zsc_agent, param.get("batch_size"), state_norm,
                                                   tom_model, dataset)
            all_episode_rewards_eval.extend(episode_rewards_eval)

            if len(all_episode_rewards) >= 10 and len(all_episode_rewards_eval) >= 10:
                avg_reward = np.mean(all_episode_rewards[-10:])
                print(f"Total Timesteps: {total_timesteps}, Average Reward (last 10 episodes): {avg_reward:.2f}")
                writer.add_scalar("Reward/avg_last10", avg_reward, total_timesteps)
                avg_reward_eval = np.mean(all_episode_rewards_eval[-10:])
                writer.add_scalar("Reward/eval", avg_reward_eval, total_timesteps)
                # if avg_reward > param.get("target_reward"):
                #     break
        if np.mean(all_episode_rewards[-10:]) < 1e-2:
            print("training finished early")
            break

    writer.close()
    env.close()


if __name__ == "__main__":
    main()
