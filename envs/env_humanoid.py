"""
3D Maze Navigation Tasks
"""
import sys
import numpy
from numpy import random
from copy import deepcopy
import gym
import metagym.metalocomotion
from epann.utils import categorical

ant_env = gym.make("meta-humanoid-v0")

def gen_task(task_type):
    return ant_env.sample_task(task_type=task_type)


class HumanoidTask(object):
    def __init__(self):  # Can set goal to test adaptation.
        self._ant_env = gym.make("meta-humanoid-v0")
        self._frame_skip = 2
        self._max_step = 500

    def reset(self, task, eps_type):
        self._ant_env.set_task(task)
        self._score = 0.0
        self._step = 0
        obs = self._ant_env.reset()
        self.eps_type = eps_type
        return obs

    def step(self, action):
        self._step += 1
        r = 0
        reward = 0
        done = False
        for _ in range(self._frame_skip):
            if not done:
                obs, r, done, info = self._ant_env.step(action)
                reward += r
        done = (done or self._step >= self._max_step)
        self._score += reward
        return done, obs, {"reward": reward, "done": done, "eps_type": self.eps_type}

    @property
    def score(self):
        return self._score

    def default_action(self):
        return self._ant_env.action_space.sample()

    def random_action(self):
        return self._ant_env.action_space.sample()

    def default_info(self):
        return {"reward": 0.0, "done": False, "eps_type": self.eps_type}

    def need_learning(self):
        return True

    def train_stage(self):
        return True

#Transform from raw output to actions, use is_tra if it is different in meta-training-train and meta-training-test
def output_to_action(output_list, info):
    act_info = dict()
    d_act = list(output_list)
    return d_act, act_info

# Transform the observation, previous action, and info into observation, pattern is not allowed to use in meta-learning
def obs_to_input(obs, action, info):
    ext_info = [info["reward"], info["done"]]
    return list(obs) + list(action) + ext_info, None

if __name__=="__main__":
    # running random policies
    game = AntTask()
    tasks = gen_task("TRAIN")
    print(tasks)
    done = False
    game.reset(tasks, None)
    step = 0
    while not done:
        action = game.random_action()
        done, obs, info = game.step(action)
        step += 1
    print(game.score)
