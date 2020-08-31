import numpy as np


class DiscountedNStepReturnEstimator():
    def __init__(self, gamma):
        """ Discounted N-Step Return Estimator
        Parameters
        ----------
        gamma: float [0, 1]
            The discount factor
        """
        if not 0. <= gamma <= 1.:
            raise ValueError("Incorrect value for gamma [0, 1]")

        self.gamma = gamma

    def __call__(self, rewards, dones, next_state_vl):
        discounted, target = np.zeros_like(rewards), next_state_vl
        for i in reversed(range(len(rewards))):
            non_terminal = (1.0 - dones[i])
            target = rewards[i] + self.gamma * target * non_terminal
            discounted[i] = target
        return discounted


class DiscountedReturnEstimator():
    def __init__(self, gamma):
        """ Discounted Return Estimator
        Parameters
        ----------
        gamma: float [0, 1]
            The discount factor
        """
        if not 0. <= gamma <= 1.:
            raise ValueError("Incorrect value for gamma [0, 1]")
        self.gamma = gamma

    def __call__(self, rewards):
        discounted, r = np.zeros_like(rewards), 0
        for i in reversed(range(len(rewards))):
            r = rewards[i] + self.gamma * r
            discounted[i] = r
        return discounted


class GeneralizedAdvantageEstimator(object):
    def __init__(self, gamma, lam):
        """ GAE
        Parameters
        ----------
        gamma: float [0, 1]
            The discount factor
        lam: float [0, 1]
            The GAE parameter lambda
        """
        if not 0. <= gamma <= 1.:
            raise ValueError("Incorrect value for gamma [0, 1]")
        if not 0. <= lam <= 1.:
            raise ValueError("Incorrect value for lam [0, 1]")

        self.gamma = gamma
        self.lam = lam

    def __call__(self, rewards, dones, states_vl, next_state_vl):
        advantages = np.zeros_like(rewards)
        last_gae_lam, next_vl = 0, next_state_vl
        for i in reversed(range(len(rewards))):
            non_terminal = (1.0 - dones[i])

            td_error = (rewards[i] + self.gamma * next_vl * non_terminal) - states_vl[i]
            last_gae_lam = td_error + self.gamma * self.lam * non_terminal * last_gae_lam

            advantages[i] = last_gae_lam
            next_vl = states_vl[i]

        return advantages
