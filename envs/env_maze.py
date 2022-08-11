"""
3D Maze Navigation Tasks
"""
import sys
import numpy
from numpy import random
from copy import deepcopy
import gym
import metagym
from metagym.metamaze.envs.maze_gen import TaskConfig
from epann.utils import categorical

maze_env = metagym.metamaze.MetaMaze2D(enable_render=False, view_grid=1, max_steps=200)

def gen_task(cell_scale=11, crowd_ratio=0.35):
    return maze_env.sample_task(cell_scale=cell_scale, allow_loops=True, crowd_ratio=crowd_ratio, step_reward=-0.01, goal_reward=1.0)._asdict()

T_Pi = 6.2831852

class MazeTask(object):
    def __init__(self):  # Can set goal to test adaptation.
        self._maze_env = metagym.metamaze.MetaMaze2D(enable_render=False, view_grid=1, max_steps=200)

    def reset(self, pattern, eps_type):
        task = TaskConfig._make(pattern.values())
        self._maze_env.set_task(task)
        self._score = 0.0
        obs = self._maze_env.reset()
        self.eps_type = eps_type
        return numpy.ravel(obs)

    def step(self, action):
        obs, r, done, info = self._maze_env.step(action)
        obs = numpy.ravel(obs)
        goal = False
        if(r > 0 and done):
            goal = True
        self._score += r
        return done, obs, {"reward": r, "done": done, "goal":goal, "eps_type": self.eps_type}

    @property
    def score(self):
        return self._score

    def default_action(self):
        return self._maze_env.action_space.sample()

    def random_action(self):
        return self._maze_env.action_space.sample()

    def default_info(self):
        return {"reward": 0.0, "done": False, "eps_type": self.eps_type}

    def need_learning(self):
        return True

    def train_stage(self):
        return True

    def optimal_steps(self):
        cells = self._maze_env.maze_core._cell_walls
        n_x, n_y = cells.shape

        val_mat = numpy.full_like(cells, fill_value=-1, dtype="float32")
        (g_i, g_j) = numpy.unravel_index(cells.argmin(), cells.shape)
        val_mat[g_i, g_j] = 0.0

        def check_validity(i,j, v):
            if(i >= n_x or j >= n_y or i < 0 or j < 0 or cells[i, j] > 0.5):
                return False
            if(val_mat[i, j] > v or val_mat[i, j] < 0):
                val_mat[i, j] = v
                return True
            return False

        path_list = [(g_i, g_j)]
        while len(path_list) > 0:
            n_i, n_j = path_list.pop()
            n_val = val_mat[n_i, n_j] + 1
            if(check_validity(n_i + 1, n_j, n_val)):
                path_list.insert(0, (n_i + 1, n_j))
            if(check_validity(n_i - 1, n_j, n_val)):
                path_list.insert(0, (n_i - 1, n_j))
            if(check_validity(n_i, n_j + 1, n_val)):
                path_list.insert(0, (n_i, n_j + 1))
            if(check_validity(n_i, n_j - 1, n_val)):
                path_list.insert(0, (n_i, n_j - 1))
        #print(val_mat)
        return val_mat[self._maze_env.maze_core._start]

#Transform from raw output to actions, use is_tra if it is different in meta-training-train and meta-training-test
def output_to_action(output_list, info):
    act_info = dict()
    if(info["rollout"] > 1):
        p = 0.5 * numpy.tanh(output_list[-1]) + 0.5
    else:
        p = 0.50
    if(random.random() < p):
        d_act = numpy.argmax(output_list[:4])
        act_info["entropy"] = 0.0
        act_info["argmax"] = True
    else:
        action_prob = numpy.exp(output_list[:4])
        action_prob *= 1.0 / numpy.sum(action_prob)
        d_act = categorical(action_prob)
        act_info["entropy"] = numpy.sum(-action_prob * numpy.log(action_prob))
        act_info["argmax"] = False
    return d_act, act_info

#Transform from raw output to actions, use is_tra if it is different in meta-training-train and meta-training-test
def output_to_action_2(output_list, info):
    act_info = dict()
    if(random.random() < 0.5 * numpy.tanh(output_list[-1]) + 0.5):
        d_act = numpy.argmax(output_list[:4])
        act_info["entropy"] = 0.0
        act_info["argmax"] = True
    else:
        action_prob = numpy.exp(output_list[:4])
        action_prob *= 1.0 / numpy.sum(action_prob)
        d_act = categorical(action_prob)
        act_info["entropy"] = numpy.sum(-action_prob * numpy.log(action_prob))
        act_info["argmax"] = False
    if(random.random() < 0.1):
        d_act = random.randint(0,3)
    return d_act, act_info

# Transform the observation, previous action, and info into observation, pattern is not allowed to use in meta-learning
def obs_to_input(obs, action, info):
    ext_info = [0.0] * 6
    ext_info[action] = 1.0
    ext_info[4] = info["reward"]
    ext_info[5] = info["done"]
    return list(obs) + ext_info, None

# For DNN to use only
def obs_to_input_2(obs, action, info):
    return list(obs), None

if __name__=="__main__":
    # running random policies
    game = MazeTask()
    tasks = gen_pattern()
    done = False
    game.reset(tasks, "TRAIN")
    print(game.optimal_steps())
    step = 0
    while not done:
        action = game.random_action()
        done, obs, info = game.step(action)
        step += 1
        print(step, obs, action, info, done)
    print(game.score)
