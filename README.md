# Learning Roles with Emergent Social Value Orientations

This is the code for experiments in the under-reviewed paper "Learning Roles with Emergent Social Value Orientations", which is based on the [LIO](https://github.com/011235813/lio).

## Setup

- Python 3.6
- Tensorflow >= 1.12
- OpenAI Gym == 0.10.9
- Clone and `pip install` [Sequential Social Dilemma](https://github.com/011235813/sequential_social_dilemma_games), which is a fork from the [original](https://github.com/eugenevinitsky/sequential_social_dilemma_games) open-source implementation.
- Clone this repository and run `$ pip install -e .` from the root.


## Navigation

* `alg/` - Implementation of RESVO.
* `env/` - Implementation of the Escape Room game and wrappers around the SSD environment.
* `results/` - Results of training will be stored in subfolders here. Each independent training run will create a subfolder that contains the final Tensorflow model, and reward log files. For example, 5 parallel independent training runs would create `results/cleanup/10x10_resvo_0`,...,`results/cleanup/10x10_resvo_4` (depending on configurable strings in config files).
* `utils/` - Utility methods.


## Examples

### Train RESVO on Escape Room

* Set config values in `alg/config_room_resvo.py`
* `cd` into the `alg` folder
* Execute training script `$ python train_multiprocess.py resvo er`. Default settings conduct 5 parallel runs with different seeds.
* For a single run, execute `$ python train_resvo.py er`.

### Train RESVO on Cleanup

* Set config values in `alg/config_ssd_resvo.py`
* `cd` into the `alg` folder
* Execute training script `$ python train_multiprocess.py resvo ssd`.
* For a single run, execute `$ python train_ssd.py`.


## License

See [LICENSE](LICENSE).

SPDX-License-Identifier: MIT
