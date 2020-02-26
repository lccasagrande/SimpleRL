from collections import defaultdict, deque

import cloudpickle as pickle
import numpy as np

from ..utils.agents import TabularAgent


class QLearningAgent(TabularAgent):
    def update(self, state, action, reward, next_state, done):
        if not done:
            next_q = self.model[next_state][np.argmax(self.model[next_state])]
        else:
            next_q = 0

        td_target = reward + self.discount_factor * next_q

        td_error = td_target - self.model[state][action]

        self.model[state][action] += self.learning_rate * td_error

    def train(self, epsilon, timesteps, logger=None):
        get_epsilon = epsilon if callable(epsilon) else lambda _: epsilon
        history = []
        state, done, score, nsteps = self.env.reset(), False, 0, 0
        for step in range(1, timesteps+1):
            eps = get_epsilon(step)
            action = self.act(state, eps)
            next_state, reward, done, info = self.env.step(action)
            self.update(state, action, reward, next_state, done)

            score += reward
            nsteps += 1
            if done:
                history.append({"score": score, "nsteps": nsteps})
                state, done, score, nsteps = self.env.reset(), False, 0, 0
            else:
                state = next_state

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
