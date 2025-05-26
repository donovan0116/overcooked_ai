import time
from copy import deepcopy
import torch
import random

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from IB_ToM.ppo_algo.agents import PPOAgent, ToMPPOAgent
from IB_ToM.ppo_algo.ppo import Normalization, get_run_log_dir
from IB_ToM.tom_net import ToMNet, make_fake_dataset, train_step1, train_step2, insert_dataset, train_tom_model
from IB_ToM.utils.utils import ParameterManager, env_maker, ReplayBuffer, build_eval_agent, evaluate_policy, \
    tom_evaluate_policy, random_choice, overcooked_obs_process, tom_one_rollout


def tom_fcp_collect_samples(env, agent_ego, agent_partner, buffer, batch_size, state_norm, tom_model, dataset):
    """
    Collect samples for self-play training.

    @param env: The environment in which both agents interact. It must support `reset()` and `step()` functions.
    @param agent_ego: The learning agent (ego agent) whose policy is being optimized.
    @param agent_partner: The partner agent used for self-play, providing diverse training interactions for the ego agent.
    @param buffer: A replay buffer or data storage object that supports a `.push()` method to store transition tuples.
    @param batch_size: The number of environment steps to collect before terminating sampling.
    @param state_norm: A normalization function applied to states before feeding them into the policy networks.

    @return:
        - steps: Total number of steps collected.
        - episode_rewards: A list of accumulated rewards per episode during sampling.
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
            env, agent_ego, agent_partner, state_norm, obs_ego, obs_partner, tom_latent.mean(dim=1).squeeze(0))
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

def tom_fcp_train(agent, buffer, writer, global_step, tom_model):
    actor_loss, critic_loss = agent.update(buffer, tom_model)
    writer.add_scalar('Loss/Actor', actor_loss, global_step)
    writer.add_scalar('Loss/Critic', critic_loss, global_step)
    return actor_loss, critic_loss


def fcp_build_population(env, agent_ego, agent_partner, buffer, state_norm, param, tom_model, dataset, seed):
    torch.manual_seed(seed + int(time.time() % 1000000))
    random.seed(seed + int(time.time() % 1000000))
    result_agent_pop = [deepcopy(agent_ego)]
    total_timesteps = 0
    all_episode_rewards = []
    while  total_timesteps < param.get("max_timesteps"):
        steps, episode_rewards = tom_fcp_collect_samples(
            env, agent_ego, agent_partner, buffer, param.get("batch_size"), state_norm, tom_model, dataset)
        total_timesteps += steps
        all_episode_rewards.extend(episode_rewards)
        agent_ego.update(buffer, tom_model)
        agent_partner = deepcopy(agent_ego)

        if len(all_episode_rewards) >= 10:
            avg_reward = np.mean(all_episode_rewards[-10:])
            print(f"Total Timesteps: {total_timesteps}, Average Reward (last 10 episodes): {avg_reward:.2f}")
            # todo: save model
            if total_timesteps // param.get("checkpoint") > (total_timesteps - steps) // param.get("checkpoint"):
                print(f"Save checkpoint at {total_timesteps}.")
                # agent_ego.save(f"")
                result_agent_pop.append(deepcopy(agent_ego))
            if avg_reward > param.get("target_reward"):
                result_agent_pop.append(deepcopy(agent_ego))
                break

    return result_agent_pop

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
        'partners_num': 5,
        'checkpoint': 1e6,
        # tom hyper param
        'seq_len': 2,
    }
    # Stage 1: Train diverse partner population
    param = ParameterManager(config)
    layout = "cramped_room"
    env = env_maker("Overcooked-v0", layout_name=layout)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    state_norm = Normalization(state_dim)
    buffer = ReplayBuffer()

    # build tom model and do the initial training
    tom_model = ToMNet(input_size=state_dim + 1,
                       hidden_size=[64, 256, state_dim + 1],
                       output_size=(state_dim + 1) * param.get("seq_len")).to('cuda')

    fake_dataset = make_fake_dataset(env, param.get("mini_batch_size") * 10, param.get("seq_len"))
    train_step1(tom_model, fake_dataset, param.get("mini_batch_size"), 10)
    train_step2(tom_model, fake_dataset, param.get("mini_batch_size"), 10)
    dataset = fake_dataset

    population = []
    for i in range(param.get("partners_num")):
        agent_ego = ToMPPOAgent(state_dim, action_dim, 128, config)
        agent_partner = deepcopy(agent_ego)
        sp_pop = fcp_build_population(
            env, agent_ego, agent_partner, buffer, state_norm, param, tom_model, dataset, seed=i
        )
        population.extend(sp_pop)
        print("========>There are ", len(population), " agents in the population. <========")
    # Stage 2: train FCP agent
    agent_fcp = ToMPPOAgent(state_dim, action_dim, 128, config)
    zsc_agent = build_eval_agent(env, "Random")

    # regular training fcp_agent by population
    log_dir = get_run_log_dir('./logs/tensorboard_logs/ppo_13', 'generation')
    writer = SummaryWriter(log_dir=log_dir)
    total_timesteps = 0
    all_episode_rewards = []
    all_episode_rewards_eval = []

    while total_timesteps < param.get("max_timesteps"):
        partner = random_choice(population)
        steps, episode_rewards = tom_fcp_collect_samples(
            env, agent_fcp, partner, buffer, param.get("batch_size"), state_norm, tom_model, dataset)
        total_timesteps += steps
        all_episode_rewards.extend(episode_rewards)

        tom_fcp_train(agent_fcp, buffer, writer, total_timesteps, tom_model)
        train_tom_model(tom_model, dataset, param)
        episode_rewards_eval = tom_evaluate_policy(env, agent_fcp, zsc_agent, param.get("batch_size"), state_norm,
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
    writer.close()
    env.close()



if __name__  == '__main__':
    main()