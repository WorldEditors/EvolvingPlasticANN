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
    def __init__(self, need_guide=False, guide_eps=0.20):  # Can set goal to test adaptation.
        self._maze_env = metagym.metamaze.MetaMaze2D(enable_render=False, view_grid=1, max_steps=200)
        self._need_guide = need_guide
        self._guide_eps = guide_eps

    def task_reset(self):
        self.coverage = dict()

    def reset(self, pattern, eps_type):
        task = TaskConfig._make(pattern.values())
        self._maze_env.set_task(task)
        self._score = 0.0
        obs = self._maze_env.reset()
        self.eps_type = eps_type
        if(self._need_guide):
            self.planning()
        return numpy.ravel(obs)

    def step(self, action):
        obs, r, done, info = self._maze_env.step(action)
        obs = numpy.ravel(obs)
        i, j = self._maze_env.maze_core._agent_pos
        self.coverage[(i,j)] = 0
        goal = False
        if(r > 0 and done):
            goal = True
        self._score += r
        info = {"goal": goal, "reward": 0.0, "done": False, "eps_type": self.eps_type}
        if(self._need_guide):
            info["guide"] = self.guide_policy(self._guide_eps)
        return done, obs, info

    def coverage_rate(self):
        all_area = numpy.sum(1.0 - self._maze_env.maze_core._cell_walls) - 1
        covered_area = len(self.coverage)
        return covered_area / all_area

    @property
    def score(self):
        return self._score

    def default_action(self):
        return self._maze_env.action_space.sample()

    def random_action(self):
        return self._maze_env.action_space.sample()

    def default_info(self):
        info = {"goal": False, "reward": 0.0, "done": False, "eps_type": self.eps_type}
        if(self._need_guide):
            info["guide"] = self.guide_policy(self._guide_eps)
        return info

    def need_learning(self):
        return True

    def train_stage(self):
        return True

    def guide_policy(self, eps):
        i, j = self._maze_env.maze_core._agent_pos
        cell_walls = self._maze_env.maze_core._cell_walls[i-1:i+2, j-1:j+2]
        avail_direction = list()
        avail_id = dict()
        explore_id = list()
        min_val = 1e+10
        sel_dir = -1

        if(cell_walls[0,1] < 0.5):
            avail_direction.append(0)
            if(min_val > self.val_mat[i-1, j]):
                min_val = self.val_mat[i-1, j]
                sel_dir = 0
            if((i-1, j) not in self.coverage):
                explore_id.append(0)
        if(cell_walls[2,1] < 0.5):
            avail_direction.append(1)
            if(min_val > self.val_mat[i+1, j]):
                min_val = self.val_mat[i+1, j]
                sel_dir = 1
            if((i+1, j) not in self.coverage):
                explore_id.append(1)
        if(cell_walls[1,0] < 0.5):
            avail_direction.append(2)
            if(min_val > self.val_mat[i, j-1]):
                min_val = self.val_mat[i, j-1]
                sel_dir = 2
            if((i, j-1) not in self.coverage):
                explore_id.append(2)
        if(cell_walls[1,2] < 0.5):
            avail_direction.append(3)
            if(min_val > self.val_mat[i, j+1]):
                min_val = self.val_mat[i, j+1]
                sel_dir = 3
            if((i, j+1) not in self.coverage):
                explore_id.append(3)

        #print(self.val_mat, sel_dir)
        if(eps > random.random()):
            if(len(explore_id) < 1):
            	return random.choice(avail_direction)
            else:
                return random.choice(explore_id)
        else:
            return sel_dir

    def optimal_steps(self):
        if("val_mat" not in self.__dict__):
             raise Exception("Must call planning before getting optimal steps")
        return self.val_mat[self._maze_env.maze_core._start]
         

    def planning(self):
        cells = self._maze_env.maze_core._cell_walls
        n_x, n_y = cells.shape

        self.val_mat = numpy.full_like(cells, fill_value=-1, dtype="float32")
        (g_i, g_j) = numpy.unravel_index(cells.argmin(), cells.shape)
        self.val_mat[g_i, g_j] = 0.0

        def check_validity(i,j, v):
            if(i >= n_x or j >= n_y or i < 0 or j < 0 or cells[i, j] > 0.5):
                return False
            if(self.val_mat[i, j] > v or self.val_mat[i, j] < 0):
                self.val_mat[i, j] = v
                return True
            return False

        path_list = [(g_i, g_j)]
        while len(path_list) > 0:
            n_i, n_j = path_list.pop()
            n_val = self.val_mat[n_i, n_j] + 1
            if(check_validity(n_i + 1, n_j, n_val)):
                path_list.insert(0, (n_i + 1, n_j))
            if(check_validity(n_i - 1, n_j, n_val)):
                path_list.insert(0, (n_i - 1, n_j))
            if(check_validity(n_i, n_j + 1, n_val)):
                path_list.insert(0, (n_i, n_j + 1))
            if(check_validity(n_i, n_j - 1, n_val)):
                path_list.insert(0, (n_i, n_j - 1))
        #print(val_mat)

#Transform from raw output to actions, use is_tra if it is different in meta-training-train and meta-training-test
def output_to_action(output_list, info):
    act_info = dict()
    p = (0.5 * numpy.tanh(output_list[-1]) + 0.5)
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
def output_to_action_offpolicy(output_list, info):
    act_info = dict()
    noise = [0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0]
    if(random.random() < noise[info["rollout"] - 1]):
        d_act = info["guide"]
        act_info["entropy"] = 0.0
        act_info["argmax"] = True
    else:
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
