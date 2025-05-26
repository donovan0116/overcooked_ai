from copy import deepcopy

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from IB_ToM.ppo_algo.agents import PPOAgent
from IB_ToM.ppo_algo.ppo import Normalization, get_run_log_dir
from IB_ToM.ppo_algo.self_play_main import sp_collect_samples
from IB_ToM.utils.utils import ParameterManager, env_maker, ReplayBuffer, overcooked_obs_process, one_rollout, \
    build_eval_agent, evaluate_policy, print_generation_banner, modify_tuple, mep_compute_avg_entropy


def mep_collect_samples(env, agent_ego, agent_partner, buffer, batch_size, state_norm, population, param):
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
    while steps < batch_size:
        next_obs_ego, next_obs_partner, reward, done, one_traj = one_rollout(
            env, agent_ego, agent_partner, state_norm, obs_ego, obs_partner
        )
        # compute augment reward
        avg_ent = mep_compute_avg_entropy(population, obs_partner)

        augment_reward = (avg_ent * param.get("alpha") + reward)
        one_traj = modify_tuple(one_traj, 2, augment_reward)
        reward = augment_reward

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

def train_mep_population_old(env, pop, param, state_norm):
    for i in range(param.get("pop_size")):
        agent_ego = pop[i]
        agent_partner = deepcopy(agent_ego)
        buffer = ReplayBuffer()
        total_timesteps = 0
        all_episode_rewards = []

        while total_timesteps < param.get("max_timesteps"):
            steps, episode_rewards = mep_collect_samples(
                env, agent_ego, agent_partner, buffer, param.get("batch_size"), state_norm, pop, param
            )
            total_timesteps += steps
            all_episode_rewards.extend(episode_rewards)

            agent_ego.update(buffer)
            agent_partner = deepcopy(agent_ego)
            if len(all_episode_rewards) >= 10:
                avg_reward = np.mean(all_episode_rewards[-10:])
                print(f"Total Timesteps: {total_timesteps}, Average Reward (last 10 episodes): {avg_reward:.2f}")
                if avg_reward > param.get("target_reward"):
                    pop[i] = deepcopy(agent_ego)
                    break
        print(f"Agent {i} in the population was trained.")
    return pop

def train_mep_population(env, pop, param, state_norm):
    total_timesteps = 0
    buffer = ReplayBuffer()
    # record all episode rewards for each agent in pop
    all_episode_reward_set = []
    pop_trained = []
    for i in range(param.get("pop_size")):
        all_episode_reward_set.append([])
        pop_trained.append(False)

    while total_timesteps < param.get("max_timesteps"):
        steps_already_updated = False
        for i in range(param.get("pop_size")):
            if pop_trained[i]:
                continue
            agent_ego = pop[i]
            agent_partner = deepcopy(agent_ego)
            steps, episode_rewards = mep_collect_samples(
                env, agent_ego, agent_partner, buffer, param.get("batch_size"), state_norm, pop, param
            )
            if steps_already_updated is not True:
                total_timesteps += steps
                steps_already_updated = True
            all_episode_reward_set[i].extend(episode_rewards)
            agent_ego.update(buffer)
            pop[i] = deepcopy(agent_ego)
            if len(all_episode_reward_set[i]) >= 10:
                avg_reward = np.mean(all_episode_reward_set[i][-10:])
                print(f"[***Agent {i}***] Total Timesteps: {total_timesteps}, Average Reward (last 10 episodes): {avg_reward:.2f}")
                if avg_reward > param.get("target_reward"):
                    pop_trained[i] = True
                    print(f"Agent {i} trained!")

    return pop

def rollout_and_evaluate(env, agent_br, partner, param, state_norm):
    eval_times = param.get("evaluate_times")
    max_episodes = param.get("max_episodes")
    returns = []
    for _ in range(eval_times):
        ep_reward = 0
        state = env.reset()
        obs_ego, obs_partner = overcooked_obs_process(state)
        obs_ego = state_norm(obs_ego)
        obs_partner = state_norm(obs_partner)
        for _ in range(max_episodes):
            next_obs_ego, next_obs_partner, reward, done, _ = one_rollout(
                env, agent_br, partner, state_norm, obs_ego, obs_partner)
            ep_reward += reward
            obs_ego = next_obs_ego
            obs_partner = next_obs_partner
            if done:
                break
        returns.append(ep_reward)

    return sum(returns) / len(returns)


def mep_prior_choice(env, agent_br, pop, param, state_norm):
    beta = param.get("beta_mep")
    returns = []
    for partner in pop:
        returns.append(rollout_and_evaluate(env, agent_br, partner, param, state_norm))

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
        'continuous': False,
        'target_reward': 180,
        # mep settings
        'pop_size': 5,
        'alpha': 0.01,
        'beta_mep': 1,
        'evaluate_times': 100,
    }

    param = ParameterManager(config)

    layout = "cramped_room"
    env = env_maker("Overcooked-v0", layout_name=layout)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    state_norm = Normalization(state_dim)
    # Stage 1: train population of ME
    pop = []
    for _ in range(param.get("pop_size")):
        agent = PPOAgent(state_dim, action_dim, 128, param)
        pop.append(agent)

    pop = train_mep_population(env, pop, param, state_norm)
    # Stage 2: train BR agent from MEP population (for 10 times).
    agent_ego = PPOAgent(state_dim, action_dim, 128, param)
    zsc_agent = build_eval_agent(env, config, "Random")

    buffer = ReplayBuffer()

    generation = 0

    while True:
        log_dir = get_run_log_dir('./logs/tensorboard_logs/ppo_9', 'generation')
        writer = SummaryWriter(log_dir=log_dir)
        generation += 1
        print_generation_banner(generation)
        total_timesteps = 0
        all_episode_rewards = []
        all_episode_rewards_eval = []

        while total_timesteps < param.get("max_timesteps"):
            agent_partner = mep_prior_choice(env, agent_ego, pop, param, state_norm)
            steps, episode_rewards = sp_collect_samples(env, agent_ego, agent_partner, buffer, param.get("batch_size"), state_norm)
            total_timesteps += steps
            all_episode_rewards.extend(episode_rewards)

            # train
            actor_loss, critic_loss = agent_ego.update(buffer)
            writer.add_scalar('Loss/Actor', actor_loss, total_timesteps)
            writer.add_scalar('Loss/Critic', critic_loss, total_timesteps)
            # todo: change partner in eval to a human policy as zero_shot
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

    writer.close()
    env.close()


if __name__ == "__main__":
    main()
