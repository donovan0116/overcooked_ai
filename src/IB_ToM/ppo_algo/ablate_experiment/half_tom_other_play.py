import os
import sys
from copy import deepcopy
from typing import List, Any

import numpy as np
import random

import torch

from IB_ToM.ppo_algo.agents import ToMPPOAgent
from IB_ToM.ppo_algo.ppo import Normalization, get_run_log_dir
from IB_ToM.tom_net import ToMNet, make_fake_dataset, train_step1, train_step2, insert_dataset, train_tom_model, \
    MLPToMNet
from IB_ToM.utils.utils import ParameterManager, env_maker, print_generation_banner, tom_one_rollout, \
    tom_evaluate_policy
from IB_ToM.utils.utils import ReplayBuffer, overcooked_obs_process, evaluate_policy, build_eval_agent
from torch.utils.tensorboard import SummaryWriter

def tom_op_train(agent, buffer, writer, global_step, tom_model):
    actor_loss, critic_loss = agent.update(buffer, tom_model)
    writer.add_scalar('Loss/Actor', actor_loss, global_step)
    writer.add_scalar('Loss/Critic', critic_loss, global_step)
    return actor_loss, critic_loss


def tom_sp_collect_samples(env, agent_ego, agent_partner, buffer, batch_size, state_norm, tom_model, dataset):
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


def random_choice(
    candidates: List[Any],
    strategy: str = "recency",
    recency_bias: float = 0.3,
) -> Any:
    """
    选择合作伙伴策略（partner）。

    Parameters
    ----------
    candidates : list
        策略对象列表（如 Policy、Agent 等）。
    strategy : {"uniform", "recency"}
        - "uniform" : 每个策略等概率抽取。
        - "recency":  最近策略出现概率为 `recency_bias`，
                      其余概率按指数衰减分配给更旧策略。
    recency_bias : float, optional
        当 strategy == "recency" 时，最新策略被选中的目标概率。
        0.5–0.9 通常效果较好。

    Returns
    -------
    Any
        被选中的策略对象。
    """
    if not candidates:
        raise ValueError("random_choice: `candidates` 不能为空。")

    # 只有一个候选时直接返回
    if len(candidates) == 1 or strategy == "uniform":
        return random.choice(candidates)

    if strategy == "recency":
        n = len(candidates)
        # 最旧 → 最新 的权重（指数衰减）
        # w_i ∝ (1 - recency_bias)^(n-1-i)       i = 0 … n-2
        # w_{n-1} = recency_bias
        weights = [(1.0 - recency_bias) ** (n - 1 - i) for i in range(n)]
        weights[-1] = recency_bias
        # 归一化
        total = sum(weights)
        weights = [w / total for w in weights]
        # random.choices 支持按概率抽样
        return random.choices(candidates, weights=weights, k=1)[0]

    raise ValueError(f"random_choice: 未知 strategy='{strategy}'")

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
        # tom hyper param
        'seq_len': 2,
        'tom_batch_size': 32,
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
    agent_ego = ToMPPOAgent(state_dim, action_dim, 128, config)
    agent_partner = deepcopy(agent_ego)
    agent_pop = []
    zsc_agent = build_eval_agent(env, config, "Human_LSTM")

    buffer = ReplayBuffer()

    generation = 0

    # build tom model and do the initial training
    tom_model = ToMNet(input_size=state_dim + 1,
                       hidden_size=[64, 256, state_dim + 1],
                       output_size=(state_dim + 1) * param.get("seq_len")).to('cuda')

    fake_dataset = make_fake_dataset(env, param.get("mini_batch_size") * 10, param.get("seq_len"))
    train_step1(tom_model, fake_dataset, param.get("mini_batch_size"), 10)
    # train_step2(tom_model, fake_dataset, param.get("mini_batch_size"), 10)
    dataset = fake_dataset

    while True:
        log_dir = get_run_log_dir('../logs/tensorboard_logs/ppo_18', 'generation')

        writer = SummaryWriter(log_dir=log_dir)

        generation += 1
        print_generation_banner(generation)

        total_timesteps = 0
        all_episode_rewards = []
        all_episode_rewards_eval = []

        while total_timesteps < param.get("max_timesteps"):
            partner = random_choice(agent_pop + [agent_partner])
            steps, episode_rewards = tom_sp_collect_samples(
                env, agent_ego, partner, buffer,
                param.get("batch_size"), state_norm, tom_model, dataset)
            total_timesteps += steps
            all_episode_rewards.extend(episode_rewards)

            tom_op_train(agent_ego, buffer, writer, total_timesteps, tom_model)
            agent_partner = deepcopy(agent_ego)
            # train_tom_model(tom_model, dataset, param)
            train_step1(tom_model, fake_dataset, param.get("mini_batch_size"), 10)
            episode_rewards_eval = tom_evaluate_policy(env, agent_ego, zsc_agent, param.get("batch_size"), state_norm,
                                                       tom_model, dataset, param.get("tom_batch_size"), param.get("seq_len"))
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

        agent_pop.append(agent_ego)
        agent_ego = ToMPPOAgent(state_dim, action_dim, 128, config)
        agent_partner = deepcopy(agent_ego)

        if generation > 10:
            print("training finished")
            break

    writer.close()
    env.close()


if __name__ == "__main__":
    main()