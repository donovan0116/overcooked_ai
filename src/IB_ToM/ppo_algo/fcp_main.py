from copy import deepcopy

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from IB_ToM.ppo_algo.agents import PPOAgent
from IB_ToM.ppo_algo.ppo import Normalization, get_run_log_dir
from IB_ToM.ppo_algo.self_play_main import sp_collect_samples, train
from IB_ToM.utils.utils import ParameterManager, env_maker, ReplayBuffer, build_eval_agent, evaluate_policy, \
    random_choice


def fcp_build_population(env, agent_ego, agent_partner, buffer, state_norm, param, seed):
    result_agent_pop = [deepcopy(agent_ego)]
    total_timesteps = 0
    all_episode_rewards = []
    while  total_timesteps < param.get("max_timesteps"):
        steps, episode_rewards = sp_collect_samples(
            env, agent_ego, agent_partner, buffer, param.get("batch_size"), state_norm)
        total_timesteps += steps
        all_episode_rewards.extend(episode_rewards)
        agent_ego.update(buffer)
        agent_partner = deepcopy(agent_ego)

        if len(all_episode_rewards) >= 10:
            avg_reward = np.mean(all_episode_rewards[-10:])
            print(f"Total Timesteps: {total_timesteps}, Average Reward (last 10 episodes): {avg_reward:.2f}")
            # todo: save model
            if total_timesteps // param.get("checkpoint") > (total_timesteps - steps) // param.get("checkpoint"):
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
        'target_reward': 200,
        'partners_num': 1,
        'checkpoint': 1e5,
    }
    # Stage 1: Train diverse partner population
    param = ParameterManager(config)
    layout = "cramped_room"
    env = env_maker("Overcooked-v0", layout_name=layout)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    state_norm = Normalization(state_dim)
    buffer = ReplayBuffer()

    population = []
    for i in range(param.get("partners_num")):
        agent_ego = PPOAgent(state_dim, action_dim, 128, config)
        agent_partner = deepcopy(agent_ego)
        sp_pop = fcp_build_population(env, agent_ego, agent_partner, buffer, state_norm, param, seed=i)
        population.extend(sp_pop)
        print("========>There are ", len(population), " agents in the population. <========")
    # Stage 2: train FCP agent
    agent_fcp = PPOAgent(state_dim, action_dim, 128, config)
    zsc_agent = build_eval_agent(env, "Random")

    # regular training fcp_agent by population
    log_dir = get_run_log_dir('./logs/tensorboard_logs/ppo_6', 'generation')
    writer = SummaryWriter(log_dir=log_dir)
    total_timesteps = 0
    all_episode_rewards = []
    all_episode_rewards_eval = []

    while total_timesteps < param.get("max_timesteps"):
        partner = random_choice(population)
        steps, episode_rewards = sp_collect_samples(
            env, agent_fcp, partner, buffer, param.get("batch_size"), state_norm)
        total_timesteps += steps
        all_episode_rewards.extend(episode_rewards)

        train(agent_fcp, buffer, writer, total_timesteps)
        episode_rewards_eval = evaluate_policy(env, agent_fcp, zsc_agent, param.get("batch_size"), state_norm)
        all_episode_rewards_eval.extend(episode_rewards_eval)

        if len(all_episode_rewards) >= 10 and len(all_episode_rewards_eval) >= 10:
            avg_reward = np.mean(all_episode_rewards[-10:])
            print(f"Total Timesteps: {total_timesteps}, Average Reward (last 10 episodes): {avg_reward:.2f}")
            writer.add_scalar("Reward/avg_last10", avg_reward, total_timesteps)
            avg_reward_eval = np.mean(all_episode_rewards_eval[-10:])
            writer.add_scalar("Reward/eval", avg_reward_eval, total_timesteps)
            if avg_reward > param.get("target_reward"):
                break
    writer.close()
    env.close()



if __name__  == '__main__':
    main()