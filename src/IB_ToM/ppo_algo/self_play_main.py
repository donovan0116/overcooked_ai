from collections import deque
from copy import deepcopy
from typing import List, Any

import numpy as np
import random

import torch

from IB_ToM.utils.utils import ParameterManager, env_maker, print_generation_banner, one_rollout
from IB_ToM.utils.utils import ReplayBuffer, overcooked_obs_process, evaluate_policy, build_eval_agent
from agents import PPOAgent
from ppo import collect_samples, get_run_log_dir, Normalization
from torch.utils.tensorboard import SummaryWriter

def train(agent, buffer, writer, global_step):
    actor_loss, critic_loss = agent.update(buffer)
    writer.add_scalar('Loss/Actor', actor_loss, global_step)
    writer.add_scalar('Loss/Critic', critic_loss, global_step)
    return actor_loss, critic_loss



def sp_collect_samples(env, agent_ego, agent_partner, buffer, batch_size, state_norm):
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
    while steps < batch_size:
        next_obs_ego, next_obs_partner, reward, done, one_traj = one_rollout(
            env, agent_ego, agent_partner, state_norm, obs_ego, obs_partner)
        buffer.push(*one_traj)
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
        'bc_batch_size': 128,
        'bc_seq_len': 10,
        'bc_epoch': 10,
    }
    param = ParameterManager(config)

    layout = "cramped_room"
    env = env_maker("Overcooked-v0", layout_name=layout)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    state_norm = Normalization(state_dim)
    agent_ego = PPOAgent(state_dim, action_dim, 128, config)
    # agent_partner = deepcopy(agent_ego)
    agent_partner = agent_ego
    agent_pop = []
    zsc_agent = build_eval_agent(env, config, "Human_LSTM")

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
            # partner = random_choice(agent_pop + [agent_ego])
            steps, episode_rewards = sp_collect_samples(env, agent_ego, agent_partner, buffer, param.get("batch_size"), state_norm)
            total_timesteps += steps
            all_episode_rewards.extend(episode_rewards)

            train(agent_ego, buffer, writer, total_timesteps)
            # agent_partner = deepcopy(agent_ego)
            agent_partner = agent_ego
            episode_rewards_eval = evaluate_policy(env, agent_ego, zsc_agent, param.get("batch_size"), state_norm)
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

        # agent_pop.append(deepcopy(agent_ego))
        agent_pop.append(agent_ego)
        agent_ego = PPOAgent(state_dim, action_dim, 128, config)
        # agent_partner = deepcopy(agent_ego)
        agent_partner = agent_ego

        if generation > 10:
            print("training finished")
            break

    writer.close()
    env.close()


if __name__ == "__main__":
    main()