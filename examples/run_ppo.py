import argparse

import gym
import gym.spaces as spaces
import numpy as np
from gym import ObservationWrapper
import tensorflow as tf

from simple_rl.PolicyGradient.ppo import ProximalPolicyOptimization
from simple_rl.utils.wrappers import make_vec_env
import simple_rl.utils.loggers as log


def build_network(X):
    h = tf.keras.layers.Dense(256, activation='relu')(X)
    return h


def run(args):
    env = make_vec_env(env_id=args.env_id,
                       num_envs=args.nenvs,
                       sequential=False)

    agent = ProximalPolicyOptimization(env=env,
                                       steps_per_update=args.nsteps,
                                       p_network=build_network,
                                       vf_network=build_network,
                                       learning_rate=args.learning_rate,
                                       discount_factor=args.discount_factor,
                                       gae=args.gae,
                                       refresh_rate=100)

    logger = log.LoggersWrapper(loggers=[log.ConsoleLogger()])

    agent.train(timesteps=args.ntimesteps,
                batch_size=args.nsteps * args.nenvs,
                clip_vl=0.2,
                entropy_coef=0.01,
                vf_coef=0.5,
                epochs=6,
                logger=logger)

    agent.save("weights.h5")

    agent.load("weights.h5")

    print("Score : {}".format(agent.play(args.render)))

    env.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="CartPole-v0", type=str)
    parser.add_argument("--learning_rate", default=1e-3, type=float)
    parser.add_argument("--discount_factor", default=1., type=float)
    parser.add_argument("--gae", default=.95, type=float)
    parser.add_argument("--nsteps", default=20, type=int)
    parser.add_argument("--nenvs", default=12, type=int)
    parser.add_argument("--ntimesteps", default=50000, type=int)
    parser.add_argument("--render", default=True, action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
