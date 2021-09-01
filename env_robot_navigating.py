"""
2D WheeledRobot Navigating Tasks
"""
import numpy
import sys
from numpy import random
from numpy import cos, sin
from math import acos, asin


def signals_transmission(d, P_o, d_o, n, sigma):
    signal = max(0.0, P_o - n * numpy.log(d / d_o) + sigma * random.normal(0.0,1.0)) 
    return signal

T_Pi = 6.2831852

class WRNav(object):
    @classmethod
    def gen_pattern(self):
        return [random.uniform(-0.5, 0.5), 
            random.uniform(-0.5, 0.5), 
            random.uniform(0.5, 2.0), 
            random.uniform(0.1, 0.5)] 

    def __init__(self, max_step, noise):  # Can set goal to test adaptation.
        self._max_step = max_step
        #distance between two wheels
        self._wheel_width = 0.04
        self._dt = 0.10
        self._wheel_dia = 0.02
        self._max_wheel_rotation = 25.0
        self._noise = noise

    def reset(self, pattern, eps_type):
        self._step = 0
        self._score = 0.0
        self._eps_type = eps_type
        self._pos_x = 0.0
        self._pos_y = 0.0
        self._direction = 0
        self._goal = pattern[:2]
        self._signal = pattern[2:]
        return []

    def step(self, action):
        self._step += 1
        assert(isinstance(action, list) and len(action) == 2), \
                "input action does not match the requirement, received %s"%action

        #valid actions lie in between -1 and 1
        eff_action = numpy.clip(action, -1, 1)
        l_dist = action[0] * self._dt * self._wheel_dia * self._max_wheel_rotation
        r_dist = action[1] * self._dt * self._wheel_dia * self._max_wheel_rotation
        deta_r = r_dist - l_dist
        avg_dist = 0.5 * (l_dist + r_dist)
        c_cur = cos(self._direction)
        s_cur = sin(self._direction)
        if(abs(deta_r) < 1.0e-6):
            self._pos_x += avg_dist * c_cur
            self._pos_y += avg_dist * s_cur
        else:
            d_theta = deta_r / self._wheel_width
            rot_r = self._wheel_width * avg_dist / deta_r
            c_dtheta_2 = cos(0.5 * d_theta)
            s_dtheta_2 = sin(0.5 * d_theta)
            c_dtheta = c_dtheta_2 ** 2 - s_dtheta_2 ** 2
            s_dtheta = 2.0 * c_dtheta_2 * s_dtheta_2
            if(abs(c_dtheta_2) > 1.0e-6):
                d_dist = rot_r * s_dtheta / c_dtheta_2
            else:
                d_dist = 0.0
            c_mid = c_dtheta_2 * c_cur - s_dtheta_2 * s_cur
            s_mid = c_cur * s_dtheta_2 + s_cur * c_dtheta_2
            self._pos_x += d_dist * c_mid
            self._pos_y += d_dist * s_mid
            self._direction += d_theta
            while self._direction > T_Pi:
                self._direction -= T_Pi
            while self._direction < 0:
                self._direction += T_Pi

        dx = self._goal[0] - self._pos_x
        dy = self._goal[1] - self._pos_y
        dist = (dx ** 2 + dy ** 2) ** 0.5
        goal_theta = asin(dy / dist)
        if(dy > 0):
            if(goal_theta < 0):
                goal_theta = T_Pi - goal_theta
        else:
            goal_theta = 0.5 * T_Pi - goal_theta
        rel_goal_theta = goal_theta - self._direction
        if(rel_goal_theta > T_Pi):
            rel_goal_theta -= T_Pi
        elif(rel_goal_theta < 0):
            rel_goal_theta += T_Pi
        done = (dist < 0.02) or self._step > self._max_step
        reward = - dist 
        self._score += reward
        obs_sig = signals_transmission(dist, self._signal[0], 0.02, self._signal[1], self._noise)
        # Notice that the reward can not be used as instant observations
        return done, [], ({"observation":obs_sig, "reward":reward, "steps":self._step, 
                "position":[self._pos_x, self._pos_y], "direction": self._direction})

    @property
    def score(self):
        return self._score

    def default_action(self):
        return [0, 0]

    def random_action(self):
        return list(random.uniform(-1.0, 1.0, size=(2, )))

    def default_info(self):
        dx = self._goal[0] - self._pos_x
        dy = self._goal[1] - self._pos_y
        dist = (dx ** 2 + dy ** 2) ** 0.5
        obs_sig = signals_transmission(dist, self._signal[0], 0.02, self._signal[1], self._noise)
        return {"observation":obs_sig, "reward":0.0, "steps": 0, "position":[self._pos_x, self._pos_y], "direction": self._direction}

    def need_learning(self):
        return True

    def train_stage(self):
        return True

if __name__=="__main__":
    # running random policies
    game = WRNav(100, 0.01)
    game.reset(WRNav.gen_pattern(), "TRAIN")
    done = False
    while not done:
        done, _, info = game.step(game.random_action())
        print(info)
