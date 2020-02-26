from abc import ABC, abstractmethod
from collections import defaultdict

import cloudpickle as pickle
import numpy as np
import tensorflow as tf

from .wrappers import VecEnv
from .common import random_choices


class Agent(ABC):
    def __init__(self, env):
        if env is None:
            raise ValueError("Incorrect value for env (not None)")
        self.env = env

    @abstractmethod
    def act(self, state):
        raise NotImplementedError()

    @abstractmethod
    def train(self):
        raise NotImplementedError()

    @abstractmethod
    def save(self, fn):
        raise NotImplementedError()

    @abstractmethod
    def load(self, fn):
        raise NotImplementedError()

    @abstractmethod
    def update(self):
        raise NotImplementedError()

    def play(self, render=False):
        state, done, score = self.env.reset(), False, 0
        while not done:
            if render:
                self.env.render()
            state, reward, done, info = self.env.step(self.act(state))
            score += reward
        return score


class TabularAgent(Agent):
    def __init__(self, env, learning_rate, discount_factor, refresh_rate=50):
        """ Tabular Agent base class
        Parameters
        ----------
        env: environment
            The environment
        learning_rate: float [0, 1]
            The learning rate.
        discount_factor: float [0, 1]
            The discount factor.
        refresh_rate: int [0, inf]
            The number of episodes rewards to average and plot.
        """
        if not 0. <= learning_rate <= 1.:
            raise ValueError("Incorrect value for learning_rate [0, 1]")
        if not 0. <= discount_factor <= 1.:
            raise ValueError("Incorrect value for discount_factor [0, 1]")
        if refresh_rate < 0:
            raise ValueError("Incorrect value for refresh_rate [0, inf]")

        super(TabularAgent, self).__init__(env)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.refresh_rate = refresh_rate
        self.model = defaultdict(lambda: np.zeros(self.env.action_space.n))

    def act(self, state, epsilon=0):
        """Select the action for state.
        Parameters
        ----------
        state: object
            The environment state
        epsilon: float [0, 1]
            The probability of selecting a random action.
        """
        if not 0. <= epsilon <= 1.:
            raise ValueError("Incorrect value for epsilon [0, 1]")

        if np.random.random() >= epsilon:
            return np.argmax(self.model[state])
        else:
            return np.random.choice(self.env.action_space.n)

    def save(self, fn):
        with open(fn, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, fn):
        with open(fn, 'rb') as f:
            self.model = pickle.load(f)


class VecAgent(Agent):
    def __init__(self, env):
        """ Vec Agent base class """
        if not isinstance(env, VecEnv):
            raise ValueError(
                "Incorrect environment type (It must be a VecEnv)")

        super(VecAgent, self).__init__(env)

    def play(self, render=False):
        states, scores = self.env.reset(), np.zeros(self.env.num_envs)
        while True:
            if render:
                self.env.render()
            states, rewards, dones, infos = self.env.step(self.act(states))
            scores += rewards
            if dones.any():
                for i in np.nonzero(dones)[0]:
                    print("[Agent {}]\tScore: {}".format(i+1, scores[i]))
                    scores[i] = 0


class ActorCriticVecAgent(VecAgent):
    def __init__(self, env, p_network, learning_rate, vf_network=None):
        if p_network is None or not callable(p_network):
            raise ValueError("Incorrect policy network (It must be callable)")
        if not 0. <= learning_rate <= 1.:
            raise ValueError("Incorrect value for learning_rate [0, 1]")
        if vf_network is not None and not callable(vf_network):
            raise ValueError(
                "Incorrect value function network (It must be callable)")

        super(ActorCriticVecAgent, self).__init__(env)
        self.model = self.build_model(p_network, vf_network)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def build_model(self, network, vf_network):
        inputs = tf.keras.Input(shape=self.env.observation_space.shape,
                                name='state')

        p_output = tf.keras.layers.Dense(units=self.env.action_space.n,
                                         name='policy',
                                         activation='softmax')

        v_output = tf.keras.layers.Dense(units=1, name='vf')

        # Hidden Layers
        p_network = network(inputs)
        vf_network = p_network if vf_network is None else vf_network(inputs)

        # Output Layer
        p_output = p_output(p_network)
        v_output = v_output(vf_network)[:, 0]

        # Keras model
        model = tf.keras.Model(inputs=inputs, outputs=[p_output, v_output])

        return model

    def predict(self, state):
        return self.model.predict(state)

    def save(self, fn):
        self.model.save_weights(fn)

    def load(self, fn):
        self.model.load_weights(fn)

    def act(self, state, argmax=False):
        p_logits, _ = self.predict(state)
        if not argmax:
            return random_choices(p_logits)
        else:
            return np.argmax(p_logits, axis=-1)
