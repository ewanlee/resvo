"""RESVO with policy gradient for policy optimization."""
import numpy as np
import tensorflow as tf

import resvo.alg.networks as networks
import resvo.utils.util as util


class RESVO(object):

    def __init__(self, config, l_obs, l_action, nn, agent_name,
                 r_multiplier=2, n_agents=1, agent_id=0, l_action_for_r=None):
        self.alg_name = 'resvo'
        self.l_obs = l_obs
        self.dim_obs = l_obs
        self.image_obs = isinstance(self.dim_obs, list)
        self.l_action = l_action
        self.nn = nn
        self.agent_name = agent_name
        self.r_multiplier = r_multiplier
        self.n_agents = n_agents
        self.agent_id = agent_id
        self.l_action_for_r = l_action_for_r if l_action_for_r else l_action

        self.list_other_id = list(range(0, self.n_agents))
        del self.list_other_id[self.agent_id]

        # Default is allow the agent to give rewards
        self.can_give = True

        self.entropy_coeff = config.entropy_coeff
        self.gamma = config.gamma
        self.include_cost_in_chain_rule = config.include_cost_in_chain_rule
        self.lr_actor = config.lr_actor
        self.lr_cost = config.lr_cost
        self.lr_reward = config.lr_reward
        if 'optimizer' in config:
            self.optimizer = config.optimizer
        else:
            self.optimizer = 'sgd'
        self.reg = config.reg
        self.reg_coeff = config.reg_coeff
        self.separate_cost_optimizer = config.separate_cost_optimizer
        self.svo = config.svo
        self.svo_ego = config.svo_ego
        if self.svo_ego:
            self.lr_svo = config.lr_svo
            self.low_rank = config.low_rank
            self.activate_r = config.activate_r

        assert not (self.separate_cost_optimizer and self.include_cost_in_chain_rule)

        self.create_networks()
        self.policy_new = networks.PolicyNewCNN if self.image_obs else PolicyNew 
        
    def create_networks(self):
        self.action_others = tf.placeholder(
            tf.float32, [None, self.l_action_for_r * (self.n_agents - 1)])
        self.epsilon = tf.placeholder(tf.float32, None, 'epsilon')
        if self.svo is True:
            self.r_svo = tf.placeholder(tf.float32, None, 'r_svo')

        # Observation is either 1D or 3D
        if self.image_obs:
            self.obs = tf.placeholder(
                tf.float32, [None, self.dim_obs[0], self.dim_obs[1], self.dim_obs[2]],
                'obs')
            actor_net = networks.actor
            reward_net = networks.reward
        else:
            self.obs = tf.placeholder(tf.float32, [None, self.dim_obs], 'obs')
            actor_net = networks.actor_mlp
            reward_net = networks.reward_mlp

        with tf.variable_scope(self.agent_name):
            with tf.variable_scope('policy_main'):
                with tf.variable_scope('policy'):
                    probs = actor_net(self.obs, self.l_action, self.nn)
                with tf.variable_scope('eta'):
                    if self.svo is not True:
                        self.reward_function = reward_net(self.obs, self.action_others,
                                                                self.nn, n_recipients=self.n_agents)
                    else:
                        self.svo_embedding = reward_net(self.obs, self.action_others,
                                                             self.nn, n_recipients=self.n_agents)
                        self.reward_function = self.r_svo[:, None] * self.svo_embedding
                self.probs = (1 - self.epsilon) * probs + self.epsilon / self.l_action
                self.log_probs = tf.log(self.probs)
                self.action_samples = tf.multinomial(self.log_probs, 1)
                # self.action_samples = tf.random.categorical(self.log_probs, 1)

            with tf.variable_scope('policy_prime'):
                with tf.variable_scope('policy'):
                    probs = actor_net(self.obs, self.l_action, self.nn)
                self.probs_prime = (1-self.epsilon)*probs + self.epsilon/self.l_action
                self.log_probs_prime = tf.log(self.probs_prime)
                self.action_samples_prime = tf.multinomial(self.log_probs_prime, 1)
                # self.action_samples_prime = tf.random.categorical(self.log_probs_prime, 1)

        self.policy_params = tf.trainable_variables(
            self.agent_name + '/policy_main/policy')
        self.policy_prime_params = tf.trainable_variables(
            self.agent_name + '/policy_prime/policy')
        self.svo_params = tf.trainable_variables(
            self.agent_name + '/policy_main/eta')

        self.list_copy_main_to_prime_ops = []
        for idx, var in enumerate(self.policy_prime_params):
            self.list_copy_main_to_prime_ops.append(
                var.assign(self.policy_params[idx]))

        self.list_copy_prime_to_main_ops = []
        for idx, var in enumerate(self.policy_params):
            self.list_copy_prime_to_main_ops.append(
                var.assign(self.policy_prime_params[idx]))

    def receive_list_of_agents(self, list_of_agents):
        self.list_of_agents = list_of_agents

    def run_actor(self, obs, sess, epsilon, prime=False):
        feed = {self.obs: np.array([obs]), self.epsilon: epsilon}
        if prime:
            action = sess.run(self.action_samples_prime, feed_dict=feed)[0][0]
        else:
            action = sess.run(self.action_samples, feed_dict=feed)[0][0]
        return action

    def give_reward(self, obs, action_all, rew, sess, budgets=None, return_svo=False):
        try:
            action_others_1hot = util.get_action_others_1hot(action_all, self.agent_id,
                                                             self.l_action_for_r)
        except Exception as e:
            # https://github.com/tensorflow/tensorflow/commit/de6ce1de08ea97d599687fbbe5196ca4af5232ae
            # Previously, large logit values were not properly handled.  
            # This is now fixed by subtracting a maximum from logits before taking exponentials.
            # (Note that the fix is applied to the CPU functor only, as the GPU functor uses
            # a different algorithm that already admits large logits.)
            print(action_all, self.agent_id, self.l_action_for_r)
            raise e

        if not self.svo:
            feed = {self.obs: np.array([obs]),
                    self.action_others: np.array([action_others_1hot])}
            reward = sess.run(self.reward_function, feed_dict=feed)
            reward = reward.flatten() * self.r_multiplier
            return reward
        else:
            feed = {self.obs: np.array([obs]),
                    self.action_others: np.array([action_others_1hot]),
                    self.r_svo: np.array([rew])}
            if not self.svo_ego:
                reward = sess.run(self.reward_function, feed_dict=feed)
                reward = reward.flatten()
                return reward
            else:
                if not return_svo:
                    reward = sess.run(self.reward_function, feed_dict=feed)
                    reward = reward.flatten()
                    return reward
                else:
                    reward, svo = sess.run([self.reward_function, self.svo_embedding], feed_dict=feed)
                    reward = reward.flatten()
                    return reward, svo


    def create_policy_gradient_op(self):
        self.r_ext = tf.placeholder(tf.float32, [None], 'r_ext')
        if self.svo_ego:
            self.r_from_ego = tf.placeholder(tf.float32, [None], 'r_from_ego')
            r2 = self.r_from_ego # + self.activate_r + self.r_ext
        else:
            r2 = self.r_ext
        this_agent_1hot = tf.one_hot(indices=self.agent_id, depth=self.n_agents)
        for other_id in self.list_other_id:
            r2 += self.r_multiplier * tf.reduce_sum(
                tf.multiply(self.list_of_agents[other_id].reward_function,
                            this_agent_1hot), axis=1)

        if self.include_cost_in_chain_rule:
            # for this agent j, subtract the rewards given to all other agents
            # i.e. minus \sum_{i=1}^{N-1} r^i_{eta^j}
            reverse_1hot = 1 - tf.one_hot(indices=self.agent_id,
                                          depth=self.n_agents)
            r2 -= self.r_multiplier * tf.reduce_sum(
                tf.multiply(self.reward_function, reverse_1hot), axis=1)

        self.ones = tf.placeholder(tf.float32, [None], 'ones')
        self.gamma_prod = tf.math.cumprod(self.ones * self.gamma)
        returns = tf.reverse(
            tf.math.cumsum(tf.reverse(r2 * self.gamma_prod, axis=[0])), axis=[0])
        returns = returns / self.gamma_prod

        self.action_taken = tf.placeholder(tf.float32, [None, self.l_action],
                                           'action_taken')
        self.log_probs_taken = tf.log(tf.reduce_sum(
            tf.multiply(self.probs, self.action_taken), axis=1) + 1e-15)

        self.entropy = -tf.reduce_sum(tf.multiply(self.probs, self.log_probs))

        self.policy_loss = -tf.reduce_sum(
            tf.multiply(self.log_probs_taken, returns))
        self.loss = self.policy_loss - self.entropy_coeff * self.entropy

        self.policy_grads = tf.gradients(self.loss, self.policy_params)
        grads_and_vars = list(zip(self.policy_grads, self.policy_params))
        self.policy_opt = tf.train.GradientDescentOptimizer(self.lr_actor)
        self.policy_op = self.policy_opt.apply_gradients(grads_and_vars)

    def create_update_op(self):
        self.r_from_others = tf.placeholder(tf.float32, [None], 'r_from_others')
        if self.svo_ego:
            r2_val = self.r_from_ego + self.r_from_others # + self.activate_r + self.r_ext
        else:
            r2_val = self.r_ext + self.r_from_others
        if self.include_cost_in_chain_rule:
            self.r_given = tf.placeholder(tf.float32, [None], 'r_given')
            r2_val -= self.r_given
        returns_val = tf.reverse(tf.math.cumsum(
            tf.reverse(r2_val * self.gamma_prod, axis=[0])), axis=[0])
        returns_val = returns_val / self.gamma_prod

        log_probs_taken = tf.log(tf.reduce_sum(
            tf.multiply(self.probs_prime, self.action_taken), axis=1) + 1e-15)
        entropy = -tf.reduce_sum(
            tf.multiply(self.probs_prime, self.log_probs_prime))
        policy_loss = -tf.reduce_sum(
            tf.multiply(log_probs_taken, returns_val))
        loss = policy_loss - self.entropy_coeff * entropy

        policy_opt_prime = tf.train.GradientDescentOptimizer(self.lr_actor)
        self.policy_op_prime = policy_opt_prime.minimize(loss)

    def create_reward_train_op(self):
        list_reward_loss = []
        self.list_policy_new = [0 for x in range(self.n_agents)]
        self.returns = tf.placeholder(tf.float32, [None], 'returns')

        for agent in self.list_of_agents:
            if agent.agent_id == self.agent_id and \
                not self.include_cost_in_chain_rule and \
                not self.svo_ego:
                # In this case, cost for giving is not accounted in chain rule,
                # so the agent can skip over itself
                continue
            other_policy_params_new = {}
            for grad, var in zip(agent.policy_grads, agent.policy_params):
                other_policy_params_new[var.name] = var - agent.lr_actor * grad
            other_policy_new = agent.policy_new(
                other_policy_params_new, agent.l_obs, agent.l_action,
                agent.agent_name)
            self.list_policy_new[agent.agent_id] = other_policy_new

            log_probs_taken = tf.log(
                tf.reduce_sum(tf.multiply(other_policy_new.probs,
                                          other_policy_new.action_taken), axis=1))
            loss_term = -tf.reduce_sum(tf.multiply(log_probs_taken, self.returns))
            list_reward_loss.append(loss_term)

        if self.svo_ego:
            svo_matries = [agent.svo_embedding for agent in self.list_of_agents]
            svo_matries = tf.concat(svo_matries, 1) # batch_size x (n_agents x n_agents)
            # reshape to: batch_szie x n_agents x n_agents
            svo_matries = tf.reshape(svo_matries, [-1, len(self.list_of_agents), len(self.list_of_agents)])
            s, u, v = tf.svd(tf.transpose(svo_matries, perm=(0,2,1)), full_matrices=True)
            # directly constraints the nuclear norm is very unstable
            # nuclear_norm = tf.reduce_sum(s, axis=-1)
            # self.svo_loss = tf.reduce_sum(nuclear_norm)
            s, u, v = s[..., :self.low_rank], u[..., :, :self.low_rank], v[..., :, :self.low_rank]
            low_rank_svo_matries = tf.matmul(u, tf.matmul(tf.linalg.diag(s), v, adjoint_b=True))
            low_rank_svo_matries = tf.transpose(low_rank_svo_matries, perm=(0,2,1))
            list_svo_loss = tf.reduce_sum(tf.square(svo_matries-tf.stop_gradient(low_rank_svo_matries)), [-2, -1])
            if self.low_rank > 2:
                list_svo_loss *= 0.

        if self.include_cost_in_chain_rule:
            self.reward_loss = tf.reduce_sum(list_reward_loss)
        else:
            if not self.svo_ego:
                reverse_1hot = 1 - tf.one_hot(indices=self.agent_id, depth=self.n_agents)
                if self.separate_cost_optimizer or self.reg == 'l1':
                    given_each_step = tf.reduce_sum(tf.abs(
                        tf.multiply(self.reward_function, reverse_1hot)), axis=1)
                    total_given = tf.reduce_sum(tf.multiply(
                        given_each_step, self.gamma_prod/self.gamma))
                elif self.reg == 'l2':
                    total_given = tf.reduce_sum(tf.square(
                        tf.multiply(self.reward_function, reverse_1hot)))
                if self.separate_cost_optimizer:
                    self.reward_loss = tf.reduce_sum(list_reward_loss)
                else:
                    self.reward_loss = (tf.reduce_sum(list_reward_loss) +
                                        self.reg_coeff * total_given)
            else:
                if self.separate_cost_optimizer or self.reg == 'l1':
                    given_each_step = tf.reduce_sum(tf.abs(self.reward_function), axis=1)
                    total_given = tf.reduce_sum(tf.multiply(
                        given_each_step, self.gamma_prod/self.gamma))
                elif self.reg == 'l2':
                    total_given = tf.reduce_sum(tf.square(self.reward_function))
                if self.separate_cost_optimizer:
                    self.reward_loss = tf.reduce_sum(list_reward_loss)
                    self.svo_loss = tf.reduce_sum(list_svo_loss)
                else:
                    self.reward_loss = (tf.reduce_sum(list_reward_loss) +
                                        self.reg_coeff * total_given + 
                                        self.lr_svo * tf.reduce_sum(list_svo_loss))
        
        # In TensorFlow v1, the var_list param of minimize function is optional
        # Defaults (var_list=None) to the list of variables collected in the graph 
        # under the key GraphKeys.TRAINABLE_VARIABLES.    
        if self.optimizer == 'sgd':
            reward_opt = tf.train.GradientDescentOptimizer(self.lr_reward)
            if self.separate_cost_optimizer:
                cost_opt = tf.train.GradientDescentOptimizer(self.lr_cost)
                if self.svo_ego:
                    svo_opt = tf.train.GradientDescentOptimizer(self.lr_svo)
        elif self.optimizer == 'adam':
            reward_opt = tf.train.AdamOptimizer(self.lr_reward)
            if self.separate_cost_optimizer:
                cost_opt = tf.train.AdamOptimizer(self.lr_cost)
                if self.svo_ego:
                    svo_opt = tf.train.AdamOptimizer(self.lr_svo)
        self.reward_op = reward_opt.minimize(self.reward_loss)
        if self.separate_cost_optimizer:
            self.cost_op = cost_opt.minimize(total_given)
            if self.svo_ego:
                self.svo_op = svo_opt.minimize(self.svo_loss)

    def update(self, sess, buf, epsilon):
        sess.run(self.list_copy_main_to_prime_ops)

        n_steps = len(buf.obs)
        actions_1hot = util.process_actions(buf.action, self.l_action)
        ones = np.ones(n_steps)
        feed = {self.obs: buf.obs,
                self.action_taken: actions_1hot,
                self.r_ext: buf.reward,
                self.ones: ones,
                self.epsilon: epsilon}

        feed[self.r_from_others] = buf.r_from_others
        if self.include_cost_in_chain_rule:
            feed[self.r_given] = buf.r_given
        if self.svo_ego:
            feed[self.r_from_ego] = buf.r_from_ego

        _ = sess.run(self.policy_op_prime, feed_dict=feed)

    def train_reward(self, sess, list_buf, list_buf_new, epsilon,
                     summarize=False, writer=None):
        buf_self = list_buf[self.agent_id]
        buf_self_new = list_buf_new[self.agent_id]

        n_steps = len(buf_self.obs)
        ones = np.ones(n_steps)

        feed = {}

        for agent in self.list_of_agents:
            other_id = agent.agent_id
            if other_id == self.agent_id:
                continue
            buf_other = list_buf[other_id]
            actions_other_1hot = util.process_actions(buf_other.action, self.l_action)
            feed[agent.obs] = buf_other.obs
            feed[agent.action_taken] = actions_other_1hot
            feed[agent.r_ext] = buf_other.reward
            feed[agent.ones] = ones
            feed[agent.epsilon] = epsilon
            if agent.svo:
                feed[agent.r_svo] = buf_other.reward
            if agent.svo_ego:
                feed[agent.r_from_ego] = buf_other.r_from_ego

            # This is needed for the case N > 2. From an agent i's perspective,
            # another agent j will receive reward from a third agent k, 
            # so to compute j's policy update we need to input agent k's observation
            # of all other agents' actions (from agent k's perspective).
            # So in general we just feed action_others from all agents' perspectives.
            feed[agent.action_others] = util.get_action_others_1hot_batch(
                buf_other.action_all, other_id, agent.l_action_for_r)

            buf_other_new = list_buf_new[other_id]
            actions_other_1hot_new = util.process_actions(buf_other_new.action,
                                                          self.l_action)
            other_policy_new = self.list_policy_new[other_id]
            feed[other_policy_new.obs] = buf_other_new.obs
            feed[other_policy_new.action_taken] = actions_other_1hot_new

        if self.include_cost_in_chain_rule or self.svo_ego:
            # Needed to compute the chain rule,
            # These are for the update from \theta to \hat{\theta}
            action_self_1hot = util.process_actions(buf_self.action, self.l_action)
            feed[self.action_taken] = action_self_1hot
            feed[self.r_ext] = buf_self.reward
            feed[self.epsilon] = epsilon
            # These are needed for the factor
            # \nabla_{\hat{\theta}^j} J^i(\hat{\tau}, \hat{\theta}) when i == j
            action_self_1hot_new = util.process_actions(buf_self_new.action,
                                                        self.l_action)
            self_policy_new = self.list_policy_new[self.agent_id]
            feed[self_policy_new.obs] = buf_self_new.obs
            feed[self_policy_new.action_taken] = action_self_1hot_new

        n_steps = len(buf_self_new.obs)
        if self.include_cost_in_chain_rule:
            total_reward = [buf_self_new.reward[idx] + buf_self_new.r_from_others[idx]
                            - buf_self_new.r_given[idx] for idx in range(n_steps)]
        elif self.svo_ego:
            total_reward = [buf_self_new.r_from_ego[idx] \
                            # + buf_self_new.r_from_others[idx] \
                            # + self.activate_r + buf_self_new.reward[idx]
                            for idx in range(n_steps)]
        else:
            total_reward = buf_self_new.reward
        returns_new = util.process_rewards(total_reward, self.gamma)
        feed[self.obs] = buf_self.obs
        feed[self.action_others] = util.get_action_others_1hot_batch(
            buf_self.action_all, self.agent_id, self.l_action_for_r)
        feed[self.ones] = ones

        if self.svo:
            feed[self.r_svo] = buf_self.reward
        if self.svo_ego:
            feed[self.r_from_ego] = buf_self.r_from_ego

        feed[self.returns] = returns_new

        if self.separate_cost_optimizer:
            # fetches = [self.reward_op, self.cost_op, self.svo_embedding]
            # for agent in self.list_of_agents:
            #     other_id = agent.agent_id
            #     if other_id == self.agent_id:
            #         continue
            #     fetches.append(agent.svo_embedding)
            # outputs = sess.run(fetches, feed_dict=feed)
            # _ = sess.run([self.reward_op, self.cost_op], feed_dict=feed)
            if not self.svo_ego:
                _ = sess.run([self.reward_op, self.cost_op], feed_dict=feed)
            else:
                try:
                    _ = sess.run([self.reward_op, self.cost_op, self.svo_op], feed_dict=feed)
                except Exception as e:
                    print(buf_self.obs)
                    raise e
        else:
            try:
                _ = sess.run(self.reward_op, feed_dict=feed)
            except Exception as e:
                print(buf_self.obs)
                raise e

    def update_main(self, sess):
        sess.run(self.list_copy_prime_to_main_ops)

    def set_can_give(self, can_give):
        self.can_give = can_give


class PolicyNew(object):
    def __init__(self, params, l_obs, l_action, agent_name):
        self.obs = tf.placeholder(tf.float32, [None, l_obs], 'obs_new')
        self.action_taken = tf.placeholder(tf.float32, [None, l_action],
                                           'action_taken')
        prefix = agent_name + '/policy_main/policy/'
        with tf.variable_scope('policy_new'):
            h1 = tf.nn.relu(
                tf.nn.xw_plus_b(self.obs, params[prefix + 'actor_h1/kernel:0'],
                                params[prefix + 'actor_h1/bias:0']))
            h2 = tf.nn.relu(
                tf.nn.xw_plus_b(h1, params[prefix + 'actor_h2/kernel:0'],
                                params[prefix + 'actor_h2/bias:0']))
            out = tf.nn.xw_plus_b(h2, params[prefix + 'actor_out/kernel:0'],
                                params[prefix + 'actor_out/bias:0'])
        self.probs = tf.nn.softmax(out)

        
