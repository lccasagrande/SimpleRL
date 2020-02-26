from collections import defaultdict

import numpy as np
import cloudpickle as pickle

from ..utils.agents import TabularAgent


class SARSAAgent(TabularAgent):
    def update(self, state, action, reward, next_state, next_action, done):
        next_q = self.model[next_state][next_action] if not done else 0
        td_target = reward + self.discount_factor * next_q
        td_error = td_target - self.model[state][action]
        self.model[state][action] += self.learning_rate * td_error

    def train(self, epsilon, timesteps, logger=None):
        get_epsilon = epsilon if callable(epsilon) else lambda _: epsilon

        history = []
        state, done, score, nsteps = self.env.reset(), False, 0, 0
        eps = get_epsilon(0)
        action = self.act(state, eps)
        for step in range(1, timesteps+1):
            next_state, reward, done, info = self.env.step(action)
            next_action = self.act(next_state, eps)

            self.update(state, action, reward, next_state, next_action, done)

            eps = get_epsilon(step)
            score += reward
            nsteps += 1
            if done:
                history.append({"score": score, "nsteps": nsteps})
                state, done, score, nsteps = self.env.reset(), False, 0, 0
                action = self.act(state, eps)
            else:
                state, action = next_state, next_action

            if logger and (step == 1 or step % self.refresh_rate == 0):
                stats = history[-self.refresh_rate:]
                nep = len(history)
                eprew = np.nan if nep == 0 else [h['score'] for h in stats]
                eplen = np.nan if nep == 0 else [h['nsteps'] for h in stats]

                logger.log('progress', round(step / timesteps, 4))
                logger.log('nepisodes', nep)
                logger.log('eplen_avg', round(np.mean(eplen), 2))
                logger.log('eprew_avg', round(np.mean(eprew), 2))
                logger.log('eprew_max', round(np.max(eprew), 2))
                logger.log('eprew_min', round(np.min(eprew), 2))
                logger.dump()

        return history
