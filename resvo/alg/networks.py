import numpy as np
import tensorflow as tf


def conv(t_input, scope, n_filters=6, k=(3, 3), s=(1, 1), data_format='NHWC'):

    if data_format == 'NHWC':
        channel_axis = 3
        strides = [1, s[0], s[1], 1]
        b_shape = [1, 1, 1, n_filters]
    elif data_format == 'NCHW':
        channel_axis = 1
        strides = [1, 1, s[0], s[1]]
        b_shape = [1, n_filters, 1, 1]
    n_in = t_input.get_shape()[channel_axis].value
    w_shape = [k[0], k[1], n_in, n_filters]
    with tf.variable_scope(scope):
        w = tf.get_variable('w', w_shape)
        b = tf.get_variable('b', b_shape)
        out = b + tf.nn.conv2d(t_input, w, strides=strides, padding='SAME',
                               data_format=data_format)

    return out


def convnet(t_input, f=6, k=(3, 3), s=(1, 1)):
    h = tf.nn.relu(conv(t_input, 'c1', f, k, s))
    size = np.prod(h.get_shape().as_list()[1:])
    conv_flat = tf.reshape(h, [-1, size])

    return conv_flat


def actor(obs, n_actions, config, return_logits=False):
    conv_out = convnet(obs, config.n_filters, config.kernel,
                       config.stride)
    h1 = tf.layers.dense(inputs=conv_out, units=config.n_h1,
                         activation=tf.nn.relu, use_bias=True, name='actor_h1')
    h2 = tf.layers.dense(inputs=h1, units=config.n_h2,
                         activation=tf.nn.relu, use_bias=True, name='actor_h2')
    out = tf.layers.dense(inputs=h2, units=n_actions, activation=None,
                          use_bias=True, name='actor_out')
    probs = tf.nn.softmax(out, name='actor_probs')
    if return_logits:
        return out, probs
    else:
        return probs


def actor_mlp(obs, n_actions, config, return_logits=False):
    h1 = tf.layers.dense(inputs=obs, units=config.n_h1,
                         activation=tf.nn.relu,
                         use_bias=True, name='actor_h1')
    h2 = tf.layers.dense(inputs=h1, units=config.n_h2,
                         activation=tf.nn.relu,
                         use_bias=True, name='actor_h2')
    out = tf.layers.dense(inputs=h2, units=n_actions, activation=None,
                          use_bias=True, name='actor_out')
    probs = tf.nn.softmax(out, name='actor_softmax')
    if return_logits:
        return out, probs
    else:
        return probs


def actor_image_vec(obs_image, obs_vec, n_actions, config):
    """Actor network for inequity aversion agents.

    Args:
        obs_image: image part of observation
        obs_vec: agents observe vector of all agents' smoothed rewards
        n_actions: size of action space
        config: ConfigDict object
    """
    conv_out = convnet(obs_image, config.n_filters, config.kernel,
                       config.stride)
    h1 = tf.layers.dense(inputs=conv_out, units=config.n_h1,
                         activation=tf.nn.relu, use_bias=True, name='actor_h1')
    h1 = tf.concat([h1, obs_vec], axis=1)
    h2 = tf.layers.dense(inputs=h1, units=config.n_h2,
                         activation=tf.nn.relu, use_bias=True, name='actor_h2')
    out = tf.layers.dense(inputs=h2, units=n_actions, activation=None,
                          use_bias=True, name='actor_out')
    probs = tf.nn.softmax(out, name='actor_probs')

    return probs


def reward(obs, a_others, config, n_recipients=1,
           output_nonlinearity=tf.nn.sigmoid):
    """Computes reward that this agent gives to (all) agents.

    Uses a convolutional net to process image obs.

    Args:
        obs: TF placeholder
        a_others: TF placeholder for observation of other agents' actions
        config: configDict
        n_recipients: number of output nodes
        output_nonlinearity: None or a TF function

    Returns: TF tensor
    """
    conv_out = convnet(obs, config.n_filters, config.kernel,
                       config.stride)
    conv_reduced = tf.layers.dense(inputs=conv_out, units=config.n_h1,
                                   activation=tf.nn.relu, use_bias=True,
                                   name='reward_conv_reduced')
    concated = tf.concat([conv_reduced, a_others], axis=1)
    h2 = tf.layers.dense(inputs=concated, units=config.n_h2,
                         activation=tf.nn.relu, use_bias=True, name='reward_h2')
    reward_out = tf.layers.dense(inputs=h2, units=n_recipients,
                                 activation=output_nonlinearity,
                                 use_bias=False, name='reward')
    return reward_out


def reward_mlp(obs, a_others, config, n_recipients=1,
               output_nonlinearity=tf.nn.sigmoid):
    """Computes reward that this agent gives to (all) agents.

    Args:
        obs: TF placeholder
        a_others: TF placeholder for observation of other agents' actions
        config: configDict
        n_recipients: number of output nodes
        output_nonlinearity: None or a TF function

    Returns: TF tensor
    """
    concated = tf.concat([obs, a_others], axis=1)
    h1 = tf.layers.dense(inputs=concated, units=config.n_hr1,
                         activation=tf.nn.relu,
                         use_bias=True, name='reward_h1')
    h2 = tf.layers.dense(inputs=h1, units=config.n_hr2,
                         activation=tf.nn.relu,
                         use_bias=True, name='reward_h2')
    reward_out = tf.layers.dense(inputs=h2, units=n_recipients,
                                 activation=output_nonlinearity,
                                 use_bias=False, name='reward')
    return reward_out


def vnet(obs, config):
    conv_out = convnet(obs, config.n_filters, config.kernel,
                       config.stride)
    h1 = tf.layers.dense(inputs=conv_out, units=config.n_h1,
                         activation=tf.nn.relu, use_bias=True, name='v_h1')
    h2 = tf.layers.dense(inputs=h1, units=config.n_h2,
                         activation=tf.nn.relu, use_bias=True, name='v_h2')
    out = tf.layers.dense(inputs=h2, units=1, activation=None,
                          use_bias=True, name='v_out')

    return out

def vnet_svo(obs, a_all, config):
    conv_out = convnet(obs, config.n_filters, config.kernel,
                       config.stride)
    h1 = tf.layers.dense(inputs=conv_out, units=config.n_h1,
                         activation=tf.nn.relu, use_bias=True, name='v_h1')
    h2 = tf.layers.dense(inputs=h1, units=config.n_h2,
                         activation=tf.nn.relu, use_bias=True, name='v_h2')
    concated = tf.concat([h2, a_all], axis=1)
    h3 = tf.layers.dense(inputs=concated, units=config.n_h2,
                         activation=tf.nn.relu, use_bias=True, name='v_h3')
    out = tf.layers.dense(inputs=h3, units=1, activation=None,
                          use_bias=True, name='v_out')
    return out

def vnet_mlp(obs, config):
    h1 = tf.layers.dense(inputs=obs, units=config.n_h1,
                         activation=tf.nn.relu,
                         use_bias=True, name='v_h1')
    h2 = tf.layers.dense(inputs=h1, units=config.n_h2,
                         activation=tf.nn.relu,
                         use_bias=True, name='v_h2')
    out = tf.layers.dense(inputs=h2, units=1, activation=None,
                          use_bias=True, name='v_out')
    return out


def vnet_image_vec(obs_image, obs_vec, config):
    """Value function critic network for inequity aversion agents.

    Args:
        obs_image: image part of observation
        obs_vec: agents observe all agents' smoothed rewards
        config: ConfigDict object
    """
    conv_out = convnet(obs_image, config.n_filters, config.kernel,
                       config.stride)
    h1 = tf.layers.dense(inputs=conv_out, units=config.n_h1,
                         activation=tf.nn.relu, use_bias=True, name='v_h1')
    h1 = tf.concat([h1, obs_vec], axis=1)
    h2 = tf.layers.dense(inputs=h1, units=config.n_h2,
                         activation=tf.nn.relu, use_bias=True, name='v_h2')
    out = tf.layers.dense(inputs=h2, units=1, activation=None,
                          use_bias=True, name='v_out')

    return out

class PolicyNewCNN(object):

    def __init__(self, params, dim_obs, l_action, agent_name):
        self.obs = tf.placeholder(tf.float32,
                                  [None, dim_obs[0], dim_obs[1], dim_obs[2]],
                                  'obs_new')
        self.action_taken = tf.placeholder(tf.float32, [None, l_action],
                                           'action_taken')
        prefix = agent_name + '/policy_main/policy/'
        with tf.variable_scope('policy_new'):
            h = tf.nn.relu(
                tf.nn.conv2d(self.obs, params[prefix + 'c1/w:0'],
                             strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC')
                + params[prefix + 'c1/b:0'])
            size = np.prod(h.get_shape().as_list()[1:])
            conv_flat = tf.reshape(h, [-1, size])
            h1 = tf.nn.relu(
                tf.nn.xw_plus_b(conv_flat, params[prefix + 'actor_h1/kernel:0'],
                                params[prefix + 'actor_h1/bias:0']))
            h2 = tf.nn.relu(
                tf.nn.xw_plus_b(h1, params[prefix + 'actor_h2/kernel:0'],
                                params[prefix + 'actor_h2/bias:0']))
            out = tf.nn.xw_plus_b(h2, params[prefix + 'actor_out/kernel:0'],
                                  params[prefix + 'actor_out/bias:0'])
        self.probs = tf.nn.softmax(out)   
