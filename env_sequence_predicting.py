"""
Sequence Predicting Tasks
"""
import numpy
import sys
from numpy import random

class SequencePredicting(object):
    @classmethod
    def gen_pattern(self, dim):
        PI = 3.1415926
        patterns = []
        for _ in range(dim):
            amplitude = random.random() * 2.0 + 1.0
            phase = 2.0 * random.random() * PI
            omega = 2.0 * PI  / (random.random() * 90 + 10)
            patterns.append([amplitude, phase, omega])
        return patterns

    def __init__(self, dim, K, N):
        self._shots = K
        self._horizons = N
        self._dim = dim

    def reset(self, pattern, eps_type):
        self._eps_type = eps_type
        self._step = 0
        self._score = 0.0
        patterns = numpy.asarray(pattern, dtype="float32")
        assert patterns.shape[0] == self._dim, "assume patterns length %s==dimension %s"%(patterns.shape[0], self._dim)
        self._A = patterns[:, 0]
        self._phase = patterns[:, 1]
        self._omega = patterns[:, 2]
        return []

    def step(self, action):
        assert (isinstance(action, list)), "action must be a list, receive %s"%type(action)
        self._step += 1
        if(self._eps_type == "TRAIN"):
            cur_y = self._A * numpy.sin(self._omega * self._step + self._phase)
            if(self._step >= self._shots):
                done = True
            else:
                done = False
            return done, [],  {"step": self._step, "observation": list(cur_y), "label": list(cur_y), "is_observation_valid": True}
        elif(self._eps_type == "TEST"):
            t_step = self._step + self._shots
            cur_y = self._A * numpy.sin(self._omega * t_step + self._phase)
            if(self._step >= self._horizons):
                done = True
            else:
                done = False
            self._score += - numpy.mean((cur_y - numpy.array(action, dtype="float32")) ** 2) / self._horizons
            return done, [],  {"step": t_step, "observation": [0.0] * self._dim, "label": list(cur_y), "is_observation_valid": False}
        else:
            raise Exception("No such episode type: %s"%self._eps_type)

    @property
    def score(self):
        return self._score

    def default_action(self):
        return [0.0] * self._dim

    def sample_action(self):
        return [random.random() * 6.0 - 3.0] * self._dim

    def default_info(self):
        if(self._eps_type == "TRAIN"):
            cur_y = list(self._A * numpy.sin(self._phase))
            return {"step": 0, "observation": cur_y, "label": cur_y, "is_observation_valid": 1}
        elif(self._eps_type == "TEST"):
            cur_y = list(self._A * numpy.sin(self._omega * self._shots + self._phase))
            return {"step": self._shots, "observation": [0.0] * self._dim, "label": cur_y, "is_observation_valid": 0}

    def train_stage(self):
        return self._step < self._shots

if __name__=="__main__":
    dim = 3
    env = SequencePredicting(dim, 25, 50)
    env.reset(SequencePredicting.gen_pattern(dim), "TRAIN")
    done = False
    while not done:
        done, _, info = env.step(env.sample_action())
        print(info)
