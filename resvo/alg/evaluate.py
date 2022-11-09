from __future__ import print_function
from scipy import stats

import os
import time
import numpy as np
import tensorflow as tf

from resvo.alg import scripted_agents

# Map from name of map to the largest column position
# where a cleaning beam fired from that position can clear waste
cleanup_map_river_boundary = {'cleanup_small_sym': 2,
                              'cleanup_10x10_sym': 3}


def test_room_symmetric(n_eval, env, sess, list_agents,
                        alg='resvo', log=False, log_path=''):
    """Eval episodes.

    Args:
        n_eval: number of episodes to run
        env: env object
        sess: TF session
        list_agents: list of agent objects
        alg: 'resvo'. 
        log: if True, measure rewards given/received at each state

    If alg=='pg', then agents must be the version of PG with 
    continuous reward-giving actions
    """
    rewards_total = np.zeros(env.n_agents)
    n_move_lever = np.zeros(env.n_agents)
    n_move_door= np.zeros(env.n_agents)
    rewards_received= np.zeros(env.n_agents)
    r_lever = np.zeros(env.n_agents)
    r_start = np.zeros(env.n_agents)
    r_door = np.zeros(env.n_agents)
    rewards_given= np.zeros(env.n_agents)

    if log:
        header = ('episode,g-lever,g-start,g-door,'
                  'r-lever,r-start,r-door\n')
        with open(os.path.join(log_path, 'test.csv'), 'w') as f:
            f.write(header)

    total_steps = 0
    epsilon = 0
    avg_rank = 0
    for idx_episode in range(1, n_eval + 1):

        if log:
            given_at_state = np.zeros(3)
            received_at_state = np.zeros(3)
        list_obs = env.reset()
        done = False
        avg_step_rank = 0
        while not done:
            list_actions = []
            for idx, agent in enumerate(list_agents):
                action = agent.run_actor(list_obs[idx], sess, epsilon)
                list_actions.append(action)
                if action == 0:
                    n_move_lever[idx] += 1
                elif action == 2:
                    n_move_door[idx] += 1

            if alg == 'resvo':
                if env.name == 'er':
                    list_r4svo = env.get_reward(list_actions)
                else:
                    raise NotImplementedError

            list_rewards = []
            svos = []
            # entry (i,j) is reward that agent i gives to agent j
            matrix_given = np.zeros((env.n_agents, env.n_agents))
            for idx, agent in enumerate(list_agents):
                if alg == 'resvo':
                    reward, svo = agent.give_reward(list_obs[idx], list_actions, 
                                                    list_r4svo, sess, return_svo=True)
                    svos.append(svo[0])
                reward[idx] = 0
                rewards_received += reward
                rewards_given[idx] += np.sum(reward)
                list_rewards.append(np.delete(reward, idx))
                matrix_given[idx] = reward
            
            if len(svos) > 0:
                rank = np.linalg.matrix_rank(np.array(svos))
                avg_step_rank += rank

            for idx, agent in enumerate(list_agents):
                received = np.sum(matrix_given[:, idx])
                if list_actions[idx] == 0:
                    r_lever[idx] += received
                elif list_actions[idx] == 1:
                    r_start[idx] += received
                else:
                    r_door[idx] += received
                if log:
                    given_at_state[list_actions[idx]] += np.sum(matrix_given[idx, :])
                    received_at_state[list_actions[idx]] += np.sum(matrix_given[:, idx])

            list_obs_next, env_rewards, done = env.step(list_actions, list_rewards)

            rewards_total += env_rewards

            for idx in range(env.n_agents):
                # add rewards received from others
                rewards_total[idx] += np.sum(matrix_given[:, idx])
                # subtract rewards given to others
                rewards_total[idx] -= np.sum(matrix_given[idx, :])

            list_obs = list_obs_next

        avg_step_rank /= env.steps
        avg_rank += avg_step_rank
        total_steps += env.steps
        if log:
            s = '%d,' % idx_episode
            s += '{:.3e},{:.3e},{:.3e},'.format(*given_at_state)
            s += '{:.3e},{:.3e},{:.3e}'.format(*received_at_state)
            s += '\n'
            with open(os.path.join(log_path, 'test.csv'), 'a') as f:
                f.write(s)

    rewards_total /= n_eval
    n_move_lever /= n_eval
    n_move_door /= n_eval
    rewards_received /= n_eval
    rewards_given /= n_eval
    steps_per_episode = total_steps / n_eval
    r_lever /= n_eval
    r_start /= n_eval
    r_door /= n_eval
    avg_rank /= n_eval

    if alg != 'resvo':
        return (rewards_total, n_move_lever, n_move_door, rewards_received,
                rewards_given, steps_per_episode, r_lever, r_start, r_door)
    else:
        return (rewards_total, n_move_lever, n_move_door, rewards_received,
                rewards_given, steps_per_episode, r_lever, r_start, r_door, 
                avg_rank)


def test_room_symmetric_baseline(n_eval, env, sess, list_agents):
    """Eval episodes.

    Args:
        n_eval: number of episodes to run
        env: env object
        sess: TF session
        list_agents: list of agent objects
    """
    rewards_total = np.zeros(env.n_agents)
    n_move_lever = np.zeros(env.n_agents)
    n_move_door= np.zeros(env.n_agents)
    total_steps = 0
    epsilon = 0
    for _ in range(1, n_eval + 1):

        list_obs = env.reset()
        done = False
        while not done:
            list_actions = []
            for idx, agent in enumerate(list_agents):
                action = agent.run_actor(list_obs[idx], sess, epsilon)
                list_actions.append(action)
                if action % 3 == 0:
                    n_move_lever[idx] += 1
                elif action % 3 == 2:
                    n_move_door[idx] += 1

            list_obs_next, env_rewards, done, info = env.step(list_actions)

            # When using discrete reward-giving actions,
            # env_rewards may not account for the full cost, since
            # we may have cost = reward_coeff * reward_value
            # (see room_symmetric_baseline.py)
            # The original env reward is recorded in info
            # rewards_total += env_rewards
            rewards_total += info['rewards_env']
            list_obs = list_obs_next

        total_steps += env.steps

    rewards_total /= n_eval
    n_move_lever /= n_eval
    n_move_door /= n_eval
    steps_per_episode = total_steps / n_eval

    return (rewards_total, n_move_lever, n_move_door, 
            steps_per_episode)


def test_ssd(n_eval, env, sess, list_agents, alg='resvo',
             log=False, log_path='', render=False):
    """Runs test episodes for sequential social dilemma.
    
    Args:
        n_eval: number of eval episodes
        env: ssd env
        sess: TF session
        list_agents: list of resvo agents
        alg: 'resvo'
        log: only used for testing a trained model
        log_path: path to log location
        render: only used for testing a trained model

    Returns:
        np.arrays of given rewards, received rewards, env rewards,
        total rewards, waste cleared
    """
    rewards_env = np.zeros((n_eval, env.n_agents))
    rewards_given = np.zeros((n_eval, env.n_agents))
    rewards_received = np.zeros((n_eval, env.n_agents))
    rewards_total = np.zeros((n_eval, env.n_agents))
    waste_cleared = np.zeros((n_eval, env.n_agents))
    received_riverside = np.zeros((n_eval, env.n_agents))
    received_beam = np.zeros((n_eval, env.n_agents))
    received_cleared = np.zeros((n_eval, env.n_agents))
    
    if log:
        list_agent_meas = []
        list_suffix = ['given', 'received', 'reward_env',
                       'reward_total', 'waste_cleared']
        for agent_id in range(1, env.n_agents + 1):
            for suffix in list_suffix:
                list_agent_meas.append('A%d_%s' % (agent_id, suffix))

        header = 'episode,'
        header += ','.join(list_agent_meas)
        header += '\n'
        with open(os.path.join(log_path, 'test.csv'), 'w') as f:
            f.write(header)

    epsilon = 0
    avg_rank = 0
    for idx_episode in range(1, n_eval + 1):

        list_obs = env.reset()
        budgets = np.zeros(env.n_agents)
        done = False
        if render:
            env.render()
            input('Episode %d. Press enter to start: ' % idx_episode)

        avg_step_rank = 0
        while not done:
            list_actions = []
            list_binary_actions = []
            for idx, agent in enumerate(list_agents):
                action = agent.run_actor(list_obs[idx], sess, epsilon)
                list_actions.append(action)
                list_binary_actions.append(
                    1 if action == env.cleaning_action_idx else 0)

            # These are the positions seen by the incentive function
            list_agent_positions = env.env.agent_pos

            list_obs_next, env_rewards, done, info = env.step(list_actions)

            svos = []
            # (i, j) is reward from i to j
            matrix_given = np.zeros((env.n_agents, env.n_agents))
            for idx, agent in enumerate(list_agents):
                if agent.can_give:
                    if not agent.svo:
                        if env.obs_cleaned_1hot:
                            if alg == 'resvo':
                                reward, _ = agent.give_reward(
                                    list_obs[idx], list_binary_actions, None, sess)
                        else:
                            if alg == 'resvo':
                                reward, _ = agent.give_reward(list_obs[idx], list_actions, None, sess)
                    else:
                        if env.obs_cleaned_1hot:
                            if alg == 'resvo':
                                reward, svo = agent.give_reward(
                                    list_obs[idx], list_binary_actions, env_rewards[idx], sess, budgets[idx], True)
                                svos.append(svo[0])
                            else:
                                raise NotImplementedError
                        else:
                            if alg == 'resvo':
                                reward, svo = agent.give_reward(
                                    list_obs[idx], list_actions, env_rewards[idx], sess, budgets[idx], True)
                                svos.append(svo[0])
                            else:
                                raise NotImplementedError
                else:
                    reward = np.zeros(env.n_agents)
                reward[idx] = 0
                rewards_received[idx_episode-1] += reward
                rewards_given[idx_episode-1, idx] += np.sum(reward)
                matrix_given[idx] = reward
            
            if render:
                env.render()
                time.sleep(0.1)

            rewards_env[idx_episode-1] += env_rewards
            budgets += env_rewards
            rewards_total[idx_episode-1] += env_rewards

            for idx in range(env.n_agents):
                # add rewards received from others
                rewards_total[idx_episode-1, idx] += np.sum(matrix_given[:, idx])
                # subtract amount given to others
                rewards_total[idx_episode-1, idx] -= np.sum(matrix_given[idx, :])

            waste_cleared[idx_episode-1] += np.array(info['n_cleaned_each_agent'])

            if len(svos) > 0:
                rank = np.linalg.matrix_rank(np.array(svos))
                avg_step_rank += rank
            
            for idx in range(env.n_agents):
                received = np.sum(matrix_given[:, idx])
                if (list_agent_positions[idx][1] <=
                    cleanup_map_river_boundary[env.config.map_name]):
                    received_riverside[idx_episode-1, idx] += received
                if list_binary_actions[idx] == 1:
                    received_beam[idx_episode-1, idx] += received
                if info['n_cleaned_each_agent'][idx] > 0:
                    received_cleared[idx_episode-1, idx] += received

            list_obs = list_obs_next
        
        avg_step_rank /= env.steps
        avg_rank += avg_step_rank

        if log:
            temp = idx_episode - 1
            combined = np.stack([rewards_given[temp], rewards_received[temp],
                                 rewards_env[temp], rewards_total[temp],
                                 waste_cleared[temp]])
            s = '%d' % idx_episode
            for idx in range(env.n_agents):
                s += ','
                s += '{:.3e},{:.3e},{:.3e},{:.3e},{:.2f}'.format(
                    *combined[:, idx])
            s += '\n'
            with open(os.path.join(log_path, 'test.csv'), 'a') as f:
                f.write(s)

    rewards_env = np.average(rewards_env, axis=0)
    rewards_given = np.average(rewards_given, axis=0)
    rewards_received = np.average(rewards_received, axis=0)
    rewards_total = np.average(rewards_total, axis=0)
    waste_cleared = np.average(waste_cleared, axis=0)
    received_riverside = np.average(received_riverside, axis=0)
    received_beam = np.average(received_beam, axis=0)
    received_cleared = np.average(received_cleared, axis=0)
    avg_rank /= n_eval

    if log:
        s = '\nAverage\n'
        combined = np.stack([rewards_given, rewards_received,
                             rewards_env, rewards_total,
                             waste_cleared])
        for idx in range(env.n_agents):
            s += ','
            s += '{:.3e},{:.3e},{:.3e},{:.3e},{:.2f}'.format(
                *combined[:, idx])
        with open(os.path.join(log_path, 'test.csv'), 'a') as f:
            f.write(s)

    
    if alg != 'resvo':
        return (rewards_given, rewards_received, rewards_env,
                rewards_total, waste_cleared, received_riverside,
                received_beam, received_cleared)
    else:
        return (rewards_given, rewards_received, rewards_env,
                rewards_total, waste_cleared, received_riverside,
                received_beam, received_cleared, avg_rank)


def test_ssd_baseline(n_eval, env, sess, list_agents,
                      render=False, allow_giving=False,
                      ia=None):
    """Runs test episodes for actor-critic baseline on SSD.
    
    Args:
        n_eval: number of eval episodes
        env: ssd env
        sess: TF session
        list_agents: list of agents
        render: only used for testing a trained model
        allow_giving: True only for baseline with 
                      discrete reward-giving actions
        ia: inequity_aversion object

    Returns:
        np.arrays of env rewards, waste cleared
    """
    rewards_env = np.zeros(env.n_agents)
    waste_cleared = np.zeros(env.n_agents)
    if allow_giving:
        rewards_given = np.zeros(env.n_agents)
        rewards_received = np.zeros(env.n_agents)
        rewards_total = np.zeros(env.n_agents)
        received_riverside = np.zeros(env.n_agents)
        received_beam = np.zeros(env.n_agents)
        received_cleared = np.zeros(env.n_agents)
    
    epsilon = 0
    for idx_episode in range(1, n_eval + 1):

        list_obs = env.reset()
        done = False
        if render:
            env.render()
            input('Episode %d. Press enter to start: ' % idx_episode)
        if ia:
            ia.reset()
            obs_v = np.array(ia.traces)
        
        while not done:
            list_actions = []
            for idx, agent in enumerate(list_agents):
                if ia:
                    action = agent.run_actor(list_obs[idx], sess, epsilon, obs_v)
                else:
                    action = agent.run_actor(list_obs[idx], sess, epsilon)
                list_actions.append(action)

            if allow_giving:
                list_agent_positions = env.env.agent_pos
                received = np.zeros(env.n_agents)
                for idx, agent in enumerate(list_agents):
                    if env.config.asymmetric and idx == env.idx_recipient:
                        continue
                    if list_actions[idx] >= env.l_action_base:
                        received[1-idx] = env.reward_value  # assumes N=2
                        rewards_given[idx] += env.reward_value
                        rewards_received[1-idx] += env.reward_value

            list_obs_next, env_rewards, done, info = env.step(list_actions)
            if render:
                env.render()
                time.sleep(0.1)
            if ia:
                # Don't need to modify actual rewards since only the
                # traces are needed for agents' observations
                _ = ia.compute_rewards(env_rewards)
                obs_v_next = np.array(ia.traces)

            waste_cleared += np.array(info['n_cleaned_each_agent'])
            if allow_giving:
                # Note that when using discrete reward-giving actions,
                # env_rewards does not account for the full cost, since
                # we may have cost = reward_coeff * reward_value (see ssd_discrete_reward.py)
                # so the actual extrinsic reward is recorded in info
                rewards_env += info['rewards_env']
                rewards_total += info['rewards_env']
                for idx in range(env.n_agents):
                    rewards_total[idx] += received[idx]  # assuming N=2
                    rewards_total[idx] -= received[1-idx]  # given, assuming N=2
                    if (list_agent_positions[idx][1] <=
                        cleanup_map_river_boundary[env.config.map_name]):
                        received_riverside[idx] += received[idx]
                    if list_actions[idx] >= env.l_action_base:
                        received_beam[idx] += received[idx]
                    if info['n_cleaned_each_agent'][idx] > 0:
                        received_cleared[idx] += received[idx]
            else:
                rewards_env += env_rewards

            list_obs = list_obs_next
            if ia:
                obs_v = obs_v_next

    rewards_env /= n_eval
    waste_cleared /= n_eval
    if allow_giving:
        rewards_given /= n_eval
        rewards_received /= n_eval
        rewards_total /= n_eval
        received_riverside /= n_eval
        received_beam /= n_eval
        received_cleared /= n_eval
        return (rewards_given, rewards_received, rewards_env,
                rewards_total, waste_cleared, received_riverside,
                received_beam, received_cleared)
    else:
        return (rewards_env, waste_cleared)

def test_ipd(n_eval, env, sess, list_agents, alg='resvo'):
    """Eval episodes on IPD."""

    n_c = np.zeros((n_eval, env.n_agents))  # count of cooperation
    n_d = np.zeros((n_eval, env.n_agents))  # count of defection
    rewards_env = np.zeros((n_eval, env.n_agents))
    rewards_given = np.zeros((n_eval, env.n_agents))
    rewards_received = np.zeros((n_eval, env.n_agents))
    rewards_total = np.zeros((n_eval, env.n_agents))

    epsilon = 0
    for idx_episode in range(1, n_eval + 1):

        list_obs = env.reset()
        done = False
        while not done:
            list_actions = []
            for idx, agent in enumerate(list_agents):
                action = agent.run_actor(list_obs[idx], sess, epsilon)
                list_actions.append(action)
                if action == 0:
                    n_c[idx_episode-1, idx] += 1
                elif action == 1:
                    n_d[idx_episode-1, idx] += 1

            if alg == 'resvo':
                if env.name == 'ipd':
                    ac0, ac1 = list_actions[0], list_actions[1]
                    list_r4svo = [env.payout_mat[ac1][ac0], env.payout_mat[ac0][ac1]]
                else:
                    raise NotImplementedError

            matrix_given = np.zeros((env.n_agents, env.n_agents))
            for idx, agent in enumerate(list_agents):
                if agent.can_give:
                    if alg == 'resvo':
                        reward = agent.give_reward(list_obs[idx], list_actions, list_r4svo, sess)
                    else:
                        raise NotImplementedError
                else:
                    reward = np.zeros(env.n_agents)
                reward[idx] = 0
                rewards_received[idx_episode-1] += reward
                rewards_given[idx_episode-1, idx] += np.sum(reward)
                matrix_given[idx] = reward

            list_obs_next, env_rewards, done = env.step(list_actions)

            rewards_env[idx_episode-1] += env_rewards
            rewards_total[idx_episode-1] += env_rewards

            for idx in range(env.n_agents):
                rewards_total[idx_episode-1, idx] += np.sum(matrix_given[:, idx])
                rewards_total[idx_episode-1, idx] -= np.sum(matrix_given[idx, :])

            list_obs = list_obs_next

    rewards_env = np.average(rewards_env, axis=0) / env.max_steps
    rewards_given = np.average(rewards_given, axis=0) / env.max_steps
    rewards_received = np.average(rewards_received, axis=0) / env.max_steps
    rewards_total = np.average(rewards_total, axis=0) / env.max_steps

    return (rewards_given, rewards_received, rewards_env,
            rewards_total)
