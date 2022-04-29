"""
Sequence Predicting Tasks
"""
import numpy
import sys
from numpy import random

class SequenceMemorizing(object):
    @classmethod
    def gen_pattern(self, dim, l):
        return random.randint(2, size=(l, dim * 2), dtype="int32")

    def __init__(self, k):
        self._k = k

    def reset(self, pattern, eps_type):
        self._eps_type = eps_type
        self._score = 0.0
        self._l, dim = pattern.shape
        self._dim = dim // 2
        self._pattern = numpy.copy(pattern)
        self._step = 0
        if(self._eps_type=="TRAIN"):
            pat_idx = numpy.arange(self._l, dtype="int32")
            random.shuffle(pat_idx)
            pat_idx = numpy.concatenate([pat_idx, [pat_idx[0]]])
            self._pattern = self._pattern[pat_idx]
        else:
            pat_idx = random.randint(self._l, size=(self._k + 1), dtype="int32")
            self._pattern = self._pattern[pat_idx]
        self.default = numpy.full((self._dim, ), -1, dtype="int32")

        return self._pattern[self._step][:self._dim]

    def step(self, action):
        assert numpy.shape(action) == (self._dim, ), "dimension of action do not match"
        self._step += 1
        if(self._eps_type == "TRAIN"):
            obs = self._pattern[self._step][:self._dim]
            lab = self._pattern[self._step][self._dim:]
            if(self._step < self._l):
                done = False
            else:
                done = True
            return done, obs,  {"step": self._step, "obs": obs, "label": lab, "is_train": True}
        elif(self._eps_type == "TEST"):
            obs = self._pattern[self._step][:self._dim]
            lab = self._pattern[self._step - 1][self._dim:]
            if(self._step < self._k):
                done = False
            else:
                done = True
            self._score += - numpy.mean((lab - numpy.array(action, dtype="float32")) ** 2) / self._k
            return done, obs,  {"step": self._step, "obs": obs, "label": self.default, "is_train": False}
        else:
            raise Exception("No such episode type: %s"%self._eps_type)

    @property
    def score(self):
        return self._score

    def default_action(self):
        return self.default

    def sample_action(self):
        return random.randint(2, size=(self._dim), dtype="int32") * self._dim

    def default_info(self):
        if(self._eps_type == "TRAIN"):
            return {"step": 0, "obs": self._pattern[0][:self._dim], "label": self._pattern[0][self._dim:], "is_train": True}
        elif(self._eps_type == "TEST"):
            return {"step": 0, "obs": self._pattern[0][:self._dim], "label": self.default, "is_train": False}

    def train_stage(self):
        return self._step < self._shots

if __name__=="__main__":
    dim = 100
    l = 25
    env = SequenceMemorizing(10)
    env.reset(SequenceMemorizing.gen_pattern(dim, l), "TRAIN")
    done = False
    while not done:
        done, obs, info = env.step(env.sample_action())
        print(obs, info)
