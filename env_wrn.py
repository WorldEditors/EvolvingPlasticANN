"""
3D Maze Navigation Tasks
"""
import sys
import numpy
from numpy import random
import gym
import rlschool.navigator2d

env = gym.make("navigator-wr-2D-v0", enable_render=False)

def gen_pattern():
    return env.sample_task()

T_Pi = 6.2831852

class NavigatorTask(object):
    def __init__(self, horizons, signal_noise):  # Can set goal to test adaptation.
        self._env = gym.make("navigator-wr-2D-v0", max_steps=horizons, signal_noise=signal_noise, enable_render=False)

    def reset(self, task, eps_type):
        self._env.set_task(task)
        self._score = 0.0
        return self._env.reset()

    def step(self, action):
        obs, r, done, info = self._env.step(action)
        self._score += r
        return done, obs, {"reward": r}

    @property
    def score(self):
        return self._score

    def default_action(self):
        return [0.0, 0.0]

    def random_action(self):
        return self._env.action_space.sample()

    def default_info(self):
        return {"reward": 0.0}

    def need_learning(self):
        return True

    def train_stage(self):
        return True

if __name__=="__main__":
    # running random policies
    game = NavigatorTask(100, 0.05)
    tasks = gen_pattern()
    done = False
    print(tasks)
    game.reset(tasks, "TRAIN")
    while not done:
        action = game.random_action()
        done, obs, info = game.step(action)
        print(obs, action, info)
