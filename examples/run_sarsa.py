import argparse

import gym
from simple_rl.TDLearning.sarsa import SARSAAgent
from simple_rl.utils import schedules
from simple_rl.utils import loggers


def run(args):
    env = gym.make(args.env_id)

    agent = SARSAAgent(env, args.learning_rate, args.discount_factor, 200)

    epsilon = schedules.LinearSchedule(args.ntimesteps, 0.05, 1.0)

    logger = loggers.ConsoleLogger()
    agent.train(epsilon, args.ntimesteps, logger=logger)

    agent.save("qlearning")

    agent.load("qlearning")

    print("Score : {}".format(agent.play(args.render)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="CliffWalking-v0", type=str)
    parser.add_argument("--learning_rate", default=0.1, type=float)
    parser.add_argument("--discount_factor", default=1., type=float)
    parser.add_argument("--ntimesteps", default=50000, type=int)
    parser.add_argument("--render", default=False, action="store_true")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(args)
