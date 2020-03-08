
import time
import json
import csv
import os.path as osp
from abc import ABC, abstractmethod
from collections import deque, defaultdict
from multiprocessing import Process, Pipe

import numpy as np
import gym
from gym import Wrapper, spaces

from .common import tile_images


class AgentWorker():
    def __call__(self, remote, env_fn_wrapper):
        env = env_fn_wrapper.x()
        try:
            while True:
                cmd, data = remote.recv()
                if cmd == 'step':
                    ob, reward, done, info = env.step(data)
                    if done:
                        ob = env.reset()
                    remote.send((ob, reward, done, info))
                elif cmd == 'reset':
                    ob = env.reset()
                    remote.send(ob)
                elif cmd == 'close':
                    remote.close()
                    break
                elif cmd == 'get_spaces':
                    remote.send((env.action_space, env.observation_space))
                elif cmd == 'render':
                    remote.send(env.render(mode='rgb_array'))
                else:
                    raise NotImplementedError
        except KeyboardInterrupt:
            print('Agent Worker: got KeyboardInterrupt')
        finally:
            env.close()


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """
    closed = False

    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space
        self._viewer = None

    @property
    def unwrapped(self):
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self

    @property
    def viewer(self):
        if self._viewer is None:
            from gym.envs.classic_control import rendering
            self._viewer = rendering.SimpleImageViewer()
        return self._viewer

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, actions):
        pass

    @abstractmethod
    def get_images(self):
        raise NotImplementedError

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.closed = True

    def render(self, mode='human'):
        imgs = self.get_images()

        bigimg = tile_images(imgs)

        if mode == 'human':
            self.viewer.imshow(bigimg)
            return self.viewer.isopen
        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError


class DummyVecEnv(VecEnv):
    """
    VecEnv that does runs multiple environments sequentially, that is,
    the step and reset commands are send to one environment at a time.
    Useful when debugging and when num_env == 1 (in the latter case,
    avoids communication overhead)
    """

    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        super().__init__(len(self.envs), env.observation_space, env.action_space)

    def step(self, actions):
        acts = [actions] if isinstance(actions, int) else actions

        results = [self.envs[e].step(acts[e]) for e in range(self.num_envs)]

        obs, rews, dones, infos = map(list, zip(*results))

        obs = [self.envs[e].reset() if done else obs[e]
               for e, done in enumerate(dones)]

        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        obs = [self.envs[e].reset() for e in range(self.num_envs)]
        return np.stack(obs)

    def get_images(self):
        return [env.render(mode='rgb_array') for env in self.envs]

    def close(self):
        if self.closed:
            return

        for env in self.envs:
            env.close()

        super().close()


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns):
        """
        env_fns: list of environments to run in sub-processes
        """
        self.remotes, self.work_remotes = zip(
            *[Pipe() for _ in range(len(env_fns))]
        )

        self.ps = []
        for (work_remote, env_fn) in zip(self.work_remotes, env_fns):
            p = Process(target=AgentWorker(),
                        args=(work_remote, CloudpickleWrapper(env_fn)),
                        daemon=True)
            self.ps.append(p)

        for p in self.ps:
            p.start()

        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        action_space, observation_space = self.remotes[0].recv()

        super().__init__(len(self.ps), observation_space, action_space)

    def step(self, actions):
        self._assert_not_closed()

        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))

        results = [remote.recv() for remote in self.remotes]

        obs, rews, dones, infos = zip(*results)

        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        self._assert_not_closed()

        for remote in self.remotes:
            remote.send(('reset', None))

        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return

        for remote in self.remotes:
            remote.send(('close', None))

        for p in self.ps:
            p.join()

        super().close()

    def get_images(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('render', None))

        imgs = [pipe.recv() for pipe in self.remotes]

        return imgs

    def _assert_not_closed(self):
        assert not self.closed, "Trying to operate on a SubprocVecEnv after calling close()"


class VecEnvWrapper(VecEnv):
    """
    An environment wrapper that applies to an entire batch
    of environments at once.
    """

    def __init__(self, venv, observation_space=None, action_space=None):
        self.venv = venv
        super().__init__(
            num_envs=venv.num_envs,
            observation_space=observation_space or venv.observation_space,
            action_space=action_space or venv.action_space)

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, actions):
        pass

    def close(self):
        return self.venv.close()

    def render(self, mode='human'):
        return self.venv.render(mode=mode)

    def get_images(self):
        return self.venv.get_images()


class VecFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack):
        self.venv = venv
        self.nstack = nstack
        low = np.repeat(venv.observation_space.low, self.nstack, axis=-1)
        high = np.repeat(venv.observation_space.high, self.nstack, axis=-1)
        self.stackedobs = np.zeros((venv.num_envs,) + low.shape, low.dtype)
        super().__init__(venv, observation_space=spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype))

    def step(self, action):
        obs, rews, dones, infos = self.venv.step(action)
        for (i, done) in enumerate(dones):
            if done:
                self.stackedobs[i] = 0

        self._stack(obs)
        return self.stackedobs, rews, dones, infos

    def _stack(self, obs):
        self.stackedobs = np.roll(
            self.stackedobs, shift=-obs.shape[-1], axis=-1)
        self.stackedobs[..., -obs.shape[-1]:] = obs

    def reset(self):
        self.stackedobs[...] = 0
        self._stack(self.venv.reset())
        return self.stackedobs


class Monitor(Wrapper):
    def __init__(self, env, info_kws=()):
        super().__init__(env)
        self.tstart = time.time()
        self.info_kws = info_kws
        self.actions = None
        self.rewards = None
        self.needs_reset = True
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.total_steps = 0
        self.fieldnames = tuple(
            ['score', 'nsteps', 'time'] + [str(i) for i in range(env.action_space.n)]) + tuple(info_kws)

    def step(self, action):
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        ob, rew, done, info = self.env.step(action)
        self._update(action, rew, done, info)
        return ob, rew, done, info

    def reset(self, **kwargs):
        self.rewards = []
        self.actions = [0] * self.env.action_space.n
        self.needs_reset = False
        return self.env.reset(**kwargs)

    def _update(self, action, rew, done, info):
        self.rewards.append(rew)
        self.actions[action] += 1
        if done:
            self.needs_reset = True
            eprew = sum(self.rewards)
            eplen = len(self.rewards)
            epinfo = {"score": round(eprew, 6), "nsteps": eplen, "time": round(
                time.time() - self.tstart, 6)}
            for k, v in enumerate(self.actions):
                epinfo[str(k)] = v / eplen
            for k in self.info_kws:
                epinfo[k] = info[k]
            self.episode_rewards.append(eprew)
            self.episode_lengths.append(eplen)
            self.episode_times.append(time.time() - self.tstart)

            if isinstance(info, dict):
                info['episode'] = epinfo

        self.total_steps += 1

    def get_total_steps(self):
        return self.total_steps

    def get_episode_rewards(self):
        return self.episode_rewards

    def get_episode_lengths(self):
        return self.episode_lengths

    def get_episode_times(self):
        return self.episode_times


class CSVMonitor(Monitor):
    EXT = "monitor.csv"
    f = None

    def __init__(self, env, filename, info_kws=()):
        super().__init__(env, info_kws)
        if not filename.endswith(CSVMonitor.EXT):
            if osp.isdir(filename):
                filename = osp.join(filename, CSVMonitor.EXT)
            else:
                filename = filename + "." + CSVMonitor.EXT
        self.f = open(filename, "wt")
        self.logger = csv.DictWriter(self.f, fieldnames=self.fieldnames)
        self.logger.writeheader()
        self.f.flush()

    def _write_row(self, epinfo):
        if self.logger:
            self.logger.writerow(epinfo)
            self.f.flush()

    def step(self, action):
        ob, rew, done, info = super().step(action)
        if 'episode' in info:
            self._write_row(info['episode'])
        return ob, rew, done, info

    def close(self):
        super().close()
        if self.f is not None:
            self.f.close()


def make_vec_env(env_id, num_envs, sequential=False, seed=None, monitor_dir=None, info_kws=(), env_wrappers=(), **env_args):
    def make_env(rank):
        def _thunk():
            env = gym.make(env_id, **env_args)

            env.seed(None if seed is None else seed + rank)

            for wrapper in env_wrappers:
                env = wrapper(env)

            if monitor_dir is None:
                env = Monitor(env, info_kws)
            else:
                env = CSVMonitor(env, monitor_dir + str(rank), info_kws)
            return env

        return _thunk

    if num_envs <= 0:
        raise ValueError("Incorrect number of environments (num_envs > 0)")
    for wrapper in env_wrappers:
        if not isinstance(wrapper, Wrapper):
            raise ValueError(
                "Incorrect env wrapper (Expected a gym.Wrapper instance)")

    if num_envs == 1 or sequential:
        return DummyVecEnv([make_env(i) for i in range(num_envs)])
    else:
        return SubprocVecEnv([make_env(i) for i in range(num_envs)])
