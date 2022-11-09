"""Trains RESVO agents on Escape Room game.
"""

from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import random

import numpy as np
import tensorflow.compat.v1 as tf

from resvo.alg import config_ipd_resvo
from resvo.alg import config_room_resvo_n3m2
from resvo.alg import evaluate
from resvo.env import ipd_wrapper
from resvo.env import room_symmetric

import wandb

def train(config):

    group_name = '-'.join([config.alg.name, config.env.name])
    if config.env.name == 'er':
        group_name = '-'.join([group_name, 'n{}m{}'.format(
            config.env.n_agents, config.env.min_at_lever)])
    if config.resvo.svo_ego:
        group_name = '-'.join([group_name, str(config.resvo.low_rank)])
    run = wandb.init(project="svo", config=config, group=group_name)

    seed = config.main.seed
    np.random.seed(seed)
    random.seed(seed)   
    tf.random.set_random_seed(seed)

    dir_name = config.main.dir_name
    exp_name = config.main.exp_name
    log_path = os.path.join('..', 'results', exp_name, dir_name)
    model_name = config.main.model_name
    save_period = config.main.save_period

    os.makedirs(log_path, exist_ok=True)

    # Keep a record of parameters used for this run
    with open(os.path.join(log_path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4, sort_keys=True)

    svo = 'svo' in config.alg.name
    svo_ego = 'ego' in config.alg.name

    n_episodes = int(config.alg.n_episodes)
    n_eval = config.alg.n_eval
    period = config.alg.period

    epsilon = config.resvo.epsilon_start
    epsilon_step = (
        epsilon - config.resvo.epsilon_end) / config.resvo.epsilon_div

    if config.env.name == 'er':
        env = room_symmetric.Env(config.env)
    elif config.env.name == 'ipd':
        env = ipd_wrapper.IPD(config.env)

    if config.resvo.use_actor_critic:
        from resvo.alg.resvo_ac import RESVO
    else:
        from resvo.alg.resvo_agent import RESVO

    list_agents = []
    for agent_id in range(env.n_agents):
        list_agents.append(RESVO(config.resvo, env.l_obs, env.l_action,
                                config.nn, 'agent_%d' % agent_id,
                                config.env.r_multiplier, env.n_agents,
                                agent_id))        

    for agent_id in range(env.n_agents):
        list_agents[agent_id].receive_list_of_agents(list_agents)
        list_agents[agent_id].create_policy_gradient_op()
        list_agents[agent_id].create_update_op()
        if config.resvo.use_actor_critic:
            list_agents[agent_id].create_critic_train_op()

    for agent_id in range(env.n_agents):
        list_agents[agent_id].create_reward_train_op()

    config_proto = tf.ConfigProto()
    if config.main.use_gpu:
        config_proto.device_count['GPU'] = 1
        config_proto.gpu_options.allow_growth = True
    else:
        config_proto.device_count['GPU'] = 0
    sess = tf.Session(config=config_proto)
    sess.run(tf.global_variables_initializer())

    if config.resvo.use_actor_critic:
        for agent in list_agents:
            sess.run(agent.list_initialize_v_ops)

    list_agent_meas = []
    if config.env.name == 'er':
        list_suffix = ['reward_total', 'n_lever', 'n_door',
                       'received', 'given', 'r-lever', 'r-start', 'r-door']
    elif config.env.name == 'ipd':
        list_suffix = ['given', 'received', 'reward_env',
                       'reward_total']
    for agent_id in range(1, env.n_agents + 1):
        for suffix in list_suffix:
            list_agent_meas.append('A%d_%s' % (agent_id, suffix))

    saver = tf.train.Saver(max_to_keep=config.main.max_to_keep)

    header = 'episode,step_train,step,'
    header += ','.join(list_agent_meas)
    if config.env.name == 'er':
        header += ',steps_per_eps\n'
    else:
        header += '\n'
    with open(os.path.join(log_path, 'log.csv'), 'w') as f:
        f.write(header)    

    step = 0
    step_train = 0
    for idx_episode in range(1, n_episodes + 1):

        list_buffers = run_episode(sess, env, list_agents, epsilon,
                                   prime=False)
        step += len(list_buffers[0].obs)

        # update policy according to policy gradient calculated by 
        # external_reward + reward_from_others
        for idx, agent in enumerate(list_agents):
            agent.update(sess, list_buffers[idx], epsilon)

        # sample new episodes based on updated policies
        list_buffers_new = run_episode(sess, env, list_agents,
                                       epsilon, prime=True)
        step += len(list_buffers_new[0].obs)

        for agent in list_agents:
            if agent.can_give:
                agent.train_reward(sess, list_buffers,
                                   list_buffers_new, epsilon)

        for idx, agent in enumerate(list_agents):
            agent.update_main(sess)

        step_train += 1

        if idx_episode % period == 0:

            if config.env.name == 'er':
                (reward_total, n_move_lever, n_move_door, rewards_received,
                        rewards_given, steps_per_episode, r_lever,
                        r_start, r_door, avg_rank) = evaluate.test_room_symmetric(
                            n_eval, env, sess, list_agents, alg='resvo')
                ind_rst = [reward_total, n_move_lever, n_move_door,
                            rewards_received, rewards_given,
                            r_lever, r_start, r_door]
                wandb_rst = {}
                for ind, suffix in enumerate(list_suffix):
                    for i in range(len(list_agents)):
                        wandb_rst['A{}_{}'.format(i, suffix)] = ind_rst[ind][i]
                wandb_rst['steps_per_episode'] = steps_per_episode
                if svo_ego:
                    wandb_rst['avg_rank'] = avg_rank
                wandb.log(wandb_rst)
                matrix_combined = np.stack([reward_total, n_move_lever, n_move_door,
                                            rewards_received, rewards_given,
                                            r_lever, r_start, r_door])
            elif config.env.name == 'ipd':
                given, received, reward_env, reward_total = evaluate.test_ipd(
                            n_eval, env, sess, list_agents, alg='resvo')
                ind_rst = [given, received, reward_env, reward_total]
                wandb_rst = {}
                for ind, suffix in enumerate(list_suffix):
                    for i in range(len(list_agents)):
                        wandb_rst['A{}_{}'.format(i, suffix)] = ind_rst[ind][i]
                wandb.log(wandb_rst)
                matrix_combined = np.stack([given, received, reward_env,
                                            reward_total])

            s = '%d,%d,%d' % (idx_episode, step_train, step)
            for idx in range(env.n_agents):
                s += ','
                if config.env.name == 'er':
                    s += ('{:.3e},{:.3e},{:.3e},{:.3e},{:.3e},'
                          '{:.3e},{:.3e},{:.3e}').format(
                              *matrix_combined[:, idx])
                elif config.env.name == 'ipd':
                    s += '{:.3e},{:.3e},{:.3e},{:.3e}'.format(
                        *matrix_combined[:, idx])
            if config.env.name == 'er':
                s += ',%.2f\n' % steps_per_episode
            else:
                s += '\n'
            with open(os.path.join(log_path, 'log.csv'), 'a') as f:
                f.write(s)

        if idx_episode % save_period == 0:
            saver.save(sess, os.path.join(log_path, '%s.%d'%(
                model_name, idx_episode)))

        if epsilon > config.resvo.epsilon_end:
            epsilon -= epsilon_step

    saver.save(sess, os.path.join(log_path, model_name))

    wandb.finish()
    

def run_episode(sess, env, list_agents, epsilon, prime=False):
    list_buffers = [Buffer(env.n_agents) for _ in range(env.n_agents)]
    list_obs = env.reset()
    done = False

    while not done:
        list_actions = []
        for agent in list_agents:
            action = agent.run_actor(list_obs[agent.agent_id], sess,
                                     epsilon, prime)
            list_actions.append(action)

        if list_agents[0].svo is True:
            if env.name == 'er':
                list_r4svo = env.get_reward(list_actions)
            elif env.name == 'ipd':
                ac0, ac1 = list_actions[0], list_actions[1]
                list_r4svo = [env.payout_mat[ac1][ac0], env.payout_mat[ac0][ac1]]
            else:
                raise NotImplementedError

        list_rewards = []
        ego_list_rewards = []
        ego_rewards = []
        total_reward_given_to_each_agent = np.zeros(env.n_agents)
        for agent in list_agents:
            if agent.can_give:
                reward = agent.give_reward(list_obs[agent.agent_id],
                                               list_actions, list_r4svo[agent.agent_id], 
                                               sess)
            else:
                reward = np.zeros(env.n_agents)
            ego_rewards.append(reward[agent.agent_id])
            reward[agent.agent_id] = 0
            total_reward_given_to_each_agent += reward
            reward = np.delete(reward, agent.agent_id)
            list_rewards.append(reward)
            if agent.svo_ego:
                reward.append(ego_rewards[-1])
                ego_list_rewards.append(reward)

        if env.name == 'er':
            list_obs_next, env_rewards, done = env.step(list_actions, list_rewards)
        elif env.name == 'ipd':
            list_obs_next, env_rewards, done = env.step(list_actions)

        for idx, buf in enumerate(list_buffers):
            buf.add([list_obs[idx], list_actions[idx], env_rewards[idx],
                     list_obs_next[idx], done])
            if list_agents[idx].svo_ego:
                buf.add_r_from_ego(ego_rewards[idx])
            buf.add_r_from_others(total_reward_given_to_each_agent[idx])
            buf.add_action_all(list_actions)
            if list_agents[idx].include_cost_in_chain_rule:
                if not list_agents[idx].svo_ego:
                    buf.add_r_given(np.sum(list_rewards[idx]))
                else:
                    buf.add_r_given(np.sum(ego_list_rewards[idx]))

        list_obs = list_obs_next

    return list_buffers


class Buffer(object):

    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.reset()

    def reset(self):
        self.obs = []
        self.action = []
        self.reward = []
        self.obs_next = []
        self.done = []
        self.r_from_ego = []
        self.r_from_others = []
        self.r_given = []
        self.action_all = []

    def add(self, transition):
        self.obs.append(transition[0])
        self.action.append(transition[1])
        self.reward.append(transition[2])
        self.obs_next.append(transition[3])
        self.done.append(transition[4])

    def add_r_from_ego(self, r):
        self.r_from_ego.append(r)

    def add_r_from_others(self, r):
        self.r_from_others.append(r)

    def add_action_all(self, list_actions):
        self.action_all.append(list_actions)

    def add_r_given(self, r):
        self.r_given.append(r)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('exp', type=str, choices=['er', 'ipd'])
    args = parser.parse_args()

    if args.exp == 'er':
        config = config_room_resvo_n3m2.get_config()
    elif args.exp == 'ipd':
        config = config_ipd_resvo.get_config()

    train(config)
