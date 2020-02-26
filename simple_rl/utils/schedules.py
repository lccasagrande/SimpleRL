import numpy as np


class LinearSchedule(object):
    def __init__(self, timesteps, final_vl, initial_vl=1.0):
        """Linear interpolation between final_vl and initial_vl over timesteps.
        Parameters
        ----------
        timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_vl: float
            initial value
        final_vl: float
            final value
        """
        self.timesteps = timesteps
        self.final_vl = final_vl
        self.initial_vl = initial_vl

    def __call__(self, t):
        frac = min(float(t) / self.timesteps, 1.0)
        return self.initial_vl + frac * (self.final_vl - self.initial_vl)
