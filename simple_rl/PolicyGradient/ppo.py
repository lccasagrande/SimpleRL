import time
from collections import defaultdict, deque


import gym
import gym.spaces as spaces
import numpy as np
import tensorflow as tf

from ..utils.common import random_choices
from ..utils.agents import ActorCriticVecAgent
from ..utils.estimators import GeneralizedAdvantageEstimator


class ProximalPolicyOptimization(ActorCriticVecAgent):
    def __init__(self, env, p_network, learning_rate, discount_factor, gae, steps_per_update, vf_network=None, refresh_rate=50):
        """ The Proximal Policy Optimization (PPO) Algorithm
        Parameters
        ----------
        env: gym env
            The environment
        p_network: object
            The policy network
        learning_rate: float [0, 1]
            The learning rate.
        discount_factor: float [0, 1]
            The discount factor.
        gae: float [0, 1]
            The GAE lambda parameter.
        steps_per_update: int
            Number of steps before updating network weights.
        vf_network: object
            The value function network
            If None, it will share the policy network
        refresh_rate: int [0, inf]
            The number of episodes to average and log.
        """
        if refresh_rate < 0:
            raise ValueError("Incorrect value for refresh_rate [0, inf]")
        if steps_per_update < 0:
            raise ValueError("Incorrect value for steps_per_update [0, inf]")

        super(ProximalPolicyOptimization, self).__init__(env=env,
                                                         p_network=p_network,
                                                         learning_rate=learning_rate,
                                                         vf_network=vf_network)
        self.steps_per_update = steps_per_update
        self.refresh_rate = refresh_rate
        self.advantage_fn = GeneralizedAdvantageEstimator(discount_factor, gae)

    def train(self, timesteps, batch_size, clip_vl, entropy_coef, vf_coef, epochs, logger=()):
        get_clip_vl = clip_vl if callable(clip_vl) else lambda _: clip_vl
        states, nupdates, transitions, history = self.env.reset(), 0, [], []
        optimizer_stats = {}
        for global_step in range(1, timesteps + 1):
            tstart = time.time()

            # Get Action
            probs, states_vl = self.model.predict(states)
            actions = random_choices(probs)

            # Act
            next_states, rewards, dones, infos = self.env.step(actions)

            # Store transition
            transitions.append((
                states, actions, next_states, rewards, dones, states_vl, probs
            ))

            # Update local vars
            states = next_states
            for info in infos:
                if 'episode' in info:
                    history.append(info['episode'])

            # Update network's parameters
            if global_step % self.steps_per_update == 0:
                clip = get_clip_vl(nupdates)
                optimizer_stats = self.update(
                    transitions, batch_size, clip, entropy_coef, vf_coef, epochs)
                transitions = []
                nupdates += 1

            # Logging
            if global_step == 1 or global_step % self.refresh_rate == 0:
                stats = history[-self.refresh_rate:]
                nep = len(history)
                eprew = np.nan if nep == 0 else [h['score'] for h in stats]
                eplen = np.nan if nep == 0 else [h['nsteps'] for h in stats]

                elapsed_time = time.time() - tstart
                fps = (self.steps_per_update *
                       self.env.num_envs) / elapsed_time
                logger.log('elapsed_time', round(elapsed_time, 5))
                logger.log('progress', round(global_step / timesteps, 2))
                logger.log('nupdates', nupdates)
                logger.log('fps', int(fps))
                logger.log('nepisodes', nep)
                logger.log('eplen_avg', float(np.mean(eplen)))
                logger.log('eprew_avg', float(np.mean(eprew)))
                logger.log('eprew_max', float(np.max(eprew)))
                logger.log('eprew_min', float(np.min(eprew)))
                for k, v in optimizer_stats.items():
                    logger.log(k, float(v))
                logger.dump()

        return history

    def update(self, transitions, batch_size, clip_vl, entropy_coef, vf_coef, epochs):
        total_steps = self.steps_per_update * self.env.num_envs
        s, a, ns, r, d, states_vl, old_probs = map(
            np.asarray, list(zip(*transitions)))

        # Estimate advantages and total return
        _, next_state_vl = self.predict(ns[-1])
        advantages = self.advantage_fn(r, d, states_vl, next_state_vl)
        total_return = advantages + states_vl

        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((
            np.asarray(s, np.float32).swapaxes(
                0, 1).reshape(total_steps, *s.shape[2:]),
            np.asarray(a, np.float32).swapaxes(0, 1).flatten(),
            np.asarray(total_return, np.float32).swapaxes(0, 1).flatten(),
            np.asarray(advantages, np.float32).swapaxes(0, 1).flatten(),
            np.asarray(states_vl, np.float32).swapaxes(0, 1).flatten(),
            np.asarray(old_probs, np.float32).swapaxes(
                0, 1).reshape(total_steps, *old_probs.shape[2:]),
        ))
        dataset = dataset.shuffle(buffer_size=total_steps)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat(epochs)

        # Update network's parameters
        stats = []
        for (state, action, target, adv, old_vf, old_prob) in dataset:
            with tf.GradientTape() as tape:
                prob, vf = self.model(state, training=True)

                # VF Loss
                vf_clipped = tf.clip_by_value(vf - old_vf, -clip_vl, clip_vl)
                vf_clipped += old_vf

                vf_loss1 = tf.keras.losses.MSE(vf, target)
                vf_loss2 = tf.keras.losses.MSE(vf_clipped, target)
                vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_loss1, vf_loss2))

                # Policy Loss
                nlogp = tf.keras.losses.sparse_categorical_crossentropy(
                    y_true=action,
                    y_pred=prob,
                    from_logits=False
                )

                old_nlogp = tf.keras.losses.sparse_categorical_crossentropy(
                    y_true=action,
                    y_pred=old_prob,
                    from_logits=False
                )

                r = tf.exp(old_nlogp - nlogp)
                r_clipped = tf.clip_by_value(r, 1.0 - clip_vl, 1.0 + clip_vl)

                p_loss1 = -adv * r
                p_loss2 = -adv * r_clipped
                p_loss = tf.reduce_mean(tf.maximum(p_loss1, p_loss2))

                # Shannon entropy.
                entropy = tf.reduce_mean(-tf.math.reduce_sum(prob *
                                                             tf.math.log(prob), axis=-1))

                # Joint Training
                loss_vl = p_loss - entropy * entropy_coef + vf_loss * vf_coef

                # Stats
                explained_var = 1 - \
                    tf.math.reduce_variance(target - old_vf) / \
                    tf.math.reduce_variance(target)
                approxkl = .5 * tf.reduce_mean(tf.square(nlogp - old_nlogp))
                clipfrac = tf.reduce_mean(
                    tf.cast(tf.greater(tf.abs(r - 1.0), clip_vl), dtype=tf.float32))

            grads = tape.gradient(loss_vl, self.model.trainable_weights)
            grads = zip(grads, self.model.trainable_weights)
            self.optimizer.apply_gradients(grads)

            stats.append((p_loss, vf_loss, entropy,
                          approxkl, clipfrac, clip_vl, explained_var))

        stats = np.mean(stats, axis=0)
        stats = {
            "policy_loss":  round(float(stats[0]), 4),
            "vf_loss":      round(float(stats[1]), 4),
            "entropy":      round(float(stats[2]), 4),
            "approxkl":     round(float(stats[3]), 4),
            "clip_frac":    round(float(stats[4]), 4),
            "clip_value":   round(float(stats[5]), 4),
            "explained_variance":   round(float(stats[6]), 4),
        }
        return stats
