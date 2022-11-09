from resvo.utils.configdict import ConfigDict


def get_config():

    config = ConfigDict()

    config.alg = ConfigDict()
    config.alg.n_episodes = 60000
    config.alg.n_eval = 10
    config.alg.n_test = 2
    config.alg.name = 'resvo'
    config.alg.period = 1000

    config.env = ConfigDict()
    config.env.name = 'ipd'
    config.env.max_steps = 5
    config.env.n_agents = 2
    config.env.r_multiplier = 3.0  # scale up sigmoid output

    config.resvo = ConfigDict()
    config.resvo.asymmetric = False
    config.resvo.decentralized = False
    config.resvo.entropy_coeff = 0.1
    config.resvo.epsilon_div = 5000
    config.resvo.epsilon_end = 0.01
    config.resvo.epsilon_start = 1.0
    config.resvo.gamma = 0.99
    config.resvo.include_cost_in_chain_rule = False
    config.resvo.idx_recipient = 1  # only used if asymmetric=True
    config.resvo.lr_actor = 1e-3
    config.resvo.lr_reward = 1e-3
    config.resvo.lr_cost = 1e-4
    config.resvo.optimizer = 'adam'
    config.resvo.separate_cost_optimizer = True
    config.resvo.reg = 'l1'
    config.resvo.reg_coeff = 0.0
    config.resvo.use_actor_critic = False
    config.resvo.svo = True
    config.resvo.svo_ego = True

    config.main = ConfigDict()
    config.main.dir_name = 'ipd_resvo'
    config.main.exp_name = 'ipd'
    config.main.max_to_keep = 100
    config.main.model_name = 'model.ckpt'
    config.main.save_period = 100000
    config.main.seed = 12341
    config.main.summarize = False
    config.main.use_gpu = False

    config.nn = ConfigDict()
    config.nn.n_h1 = 16
    config.nn.n_h2 = 8
    config.nn.n_hr1 = 16
    config.nn.n_hr2 = 8

    return config
