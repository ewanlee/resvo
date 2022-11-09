"""Tests a trained model on Escape Room."""

import argparse
import os
import random

from copy import deepcopy

import numpy as np
import tensorflow as tf

import resvo.alg.config_room_resvo_n3m2 as config_room_resvo
import resvo.alg.evaluate as evaluate
import resvo.env.room_symmetric as room_symmetric


def test_resvo(config):

    seed = config.main.seed
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)

    dir_name = config.main.dir_name
    exp_name = config.main.exp_name
    log_path = os.path.join('..', 'results', exp_name, dir_name)
    model_name = config.main.model_name

    n_test = config.alg.n_test

    env = room_symmetric.Env(config.env)

    if config.resvo.use_actor_critic:
        from resvo.alg.resvo_ac import RESVO
    else:
        from resvo.alg.resvo_agent import RESVO

    list_agents = []
    for agent_id in range(env.n_agents):
        list_agents.append(
            RESVO(config.resvo, env.l_obs, env.l_action,
                config.nn, 'agent_%d' % agent_id,
                config.env.r_multiplier, env.n_agents,
                agent_id))

    for agent_id in range(env.n_agents):
        list_agents[agent_id].receive_list_of_agents(list_agents)
        list_agents[agent_id].create_policy_gradient_op()
        list_agents[agent_id].create_update_op()

    for agent_id in range(env.n_agents):
        list_agents[agent_id].create_reward_train_op()

    if config.resvo.asymmetric:
        assert config.env.n_agents == 2
        for agent_id in range(env.n_agents):
            list_agents[agent_id].set_can_give(
                agent_id != config.resvo.idx_recipient)    

    config_proto = tf.ConfigProto()
    if config.main.use_gpu:
        config_proto.device_count['GPU'] = 1
        config_proto.gpu_options.allow_growth = True
    else:
        config_proto.device_count['GPU'] = 0
    sess = tf.Session(config=config_proto)

    saver = tf.train.Saver()
    print("Restoring variables from %s" % dir_name)
    saver.restore(sess, os.path.join(log_path, model_name))

    _ = evaluate.test_room_symmetric(
        n_test, env, sess, list_agents,
        log=True, log_path=log_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, default='resvo')
    parser.add_argument('--multiple', action='store_true')
    parser.add_argument('--n_seeds', default=20)
    parser.add_argument('--seed_base', default=12340)
    parser.add_argument('--seed_min', default=12340)
    args = parser.parse_args()

    if args.multiple:
        config = config_room_resvo.get_config()
        dir_name_base = config.main.dir_name
        for idx_run in range(args.n_seeds):
            config_copy = deepcopy(config)
            config_copy['main']['seed'] = args.seed_base + idx_run
            config_copy.main.dir_name = (dir_name_base + '_{:1d}'.format(
                args.seed_base+idx_run - args.seed_min))
            test_resvo(config_copy)
            tf.reset_default_graph()
    else:
        if args.alg == 'resvo':
            config = config_room_resvo.get_config()
            test_resvo(config)
    
