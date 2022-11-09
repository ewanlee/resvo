from resvo.utils import configdict


def get_config():

    config = configdict.ConfigDict()

    config.alg = configdict.ConfigDict()
    config.alg.n_episodes = 50000
    config.alg.n_eval = 10
    config.alg.n_test = 100
    config.alg.name = 'resvo'
    config.alg.period = 100

    config.env = configdict.ConfigDict()
    config.env.max_steps = 5
    config.env.min_at_lever = 2
    config.env.n_agents = 3
    config.env.name = 'er'
    config.env.r_multiplier = 2.0
    config.env.randomize = False
    config.env.reward_sanity_check = False

    config.resvo = configdict.ConfigDict()
    config.resvo.asymmetric = False
    config.resvo.decentralized = False
    config.resvo.entropy_coeff = 0.01
    config.resvo.epsilon_div = 1000
    config.resvo.epsilon_end = 0.1
    config.resvo.epsilon_start = 0.5
    config.resvo.gamma = 0.99
    config.resvo.include_cost_in_chain_rule = False
    config.resvo.lr_actor = 1e-4
    config.resvo.lr_cost = 1e-4
    config.resvo.lr_svo = 1e-4
    config.resvo.lr_opp = 1e-3
    config.resvo.lr_reward = 1e-3
    config.resvo.lr_v = 1e-2
    config.resvo.optimizer = 'adam'
    config.resvo.reg = 'l1'
    config.resvo.reg_coeff = 1.0
    config.resvo.separate_cost_optimizer = True
    config.resvo.tau = 0.01
    config.resvo.use_actor_critic = False
    config.resvo.svo = True
    config.resvo.svo_ego = True
    config.resvo.low_rank = 1

    config.main = configdict.ConfigDict()
    config.main.dir_name = 'er_n3_resvo'
    config.main.exp_name = 'er'
    config.main.max_to_keep = 100
    config.main.model_name = 'model.ckpt'
    config.main.save_period = 100000
    config.main.seed = 12340
    config.main.summarize = False
    config.main.use_gpu = False

    config.nn = configdict.ConfigDict()
    config.nn.n_h1 = 64
    config.nn.n_h2 = 32
    config.nn.n_hr1 = 64
    config.nn.n_hr2 = 16

    return config
                            
                
