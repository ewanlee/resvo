"""For doing multi-seed runs or hyperparameter sweep

Currently limited to sweeping over a single variable.
"""
import argparse
import sys

from multiprocessing import Process
from copy import deepcopy

import resvo.alg.train_resvo as train_resvo
import resvo.alg.train_ssd as train_ssd
import resvo.alg.config_ipd_resvo as config_ipd_resvo
import resvo.alg.config_room_resvo_n3m2 as config_room_resvo_n3m2
import resvo.alg.config_ssd_resvo as config_ssd_resvo

parser = argparse.ArgumentParser()
parser.add_argument('alg', type=str, default='resvo')
parser.add_argument('exp', type=str, choices=['er', 'ipd', 'ssd'],
                    default='er')
parser.add_argument('--n_agents', type=int, default=2)
parser.add_argument('--seed_min', type=int, default=12340)
parser.add_argument('--seed_base', type=int, default=12340)
parser.add_argument('--n_seeds', type=int, default=5)
args = parser.parse_args()

processes = []

if args.alg == 'resvo':
    if args.exp == 'ssd':
        config = config_ssd_resvo.get_config()
        train_function = train_ssd.train_function
    else:
        if args.exp == 'er':
            if args.n_agents > 2:
                if args.n_agents == 3:
                    config = config_room_resvo_n3m2.get_config()
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        elif args.exp == 'ipd':
            config = config_ipd_resvo.get_config()
        train_function = train_resvo.train

n_seeds = args.n_seeds
seed_min = args.seed_min
seed_base = args.seed_base
dir_name_base = config.main.dir_name

# Specify the group that contains the variable
group = 'main'

# Specify the name of the variable to sweep
variable = 'seed'

# Specify the range of values to sweep through
values = range(n_seeds)

for idx_run in range(len(values)):
    config_copy = deepcopy(config)
    if variable == 'seed':
        config_copy[group][variable] = seed_base + idx_run
        config_copy.main.dir_name = (
            dir_name_base + '_{:1d}'.format(seed_base+idx_run - seed_min))
    else:
        val = values[idx_run]
        if group == 'cleanup_params':
            config_copy['env'][group][variable] = val
        else:
            config_copy[group][variable] = val
        config_copy.main.dir_name = (dir_name_base + '_{:s}'.format(variable) + 
                                     '_{:s}'.format(str(val).replace('.', 'p')))

    p = Process(target=train_function, args=(config_copy,))
    p.start()
    processes.append(p)

for p in processes:
    p.join()
