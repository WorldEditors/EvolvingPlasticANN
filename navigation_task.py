import numpy
import sys
from numpy import random
from numpy import cos, sin
from math import acos, asin

def gen_pattern():
    signal = []
    return [random.uniform(-0.5, 0.5), 
        random.uniform(-0.5, 0.5), 
        random.uniform(0.5, 2.0), 
        random.uniform(0.1, 0.5)] 

def signals(d, P_o, d_o, n, sigma):
    signal = max(0.0, P_o - n * numpy.log(d / d_o) + sigma * random.normal(0.0,1.0)) 
    return signal

T_Pi = 6.2831852

class WheeledRobot(object):
    def __init__(self, max_step, noise):  # Can set goal to test adaptation.
        self._max_step = max_step
        self._wheel_width = 0.04
        self._max_wheel_rotation = 25.0
        # fraction force
        self._dt = 0.10
        self._force = 0.10
        self._time_cost = 0.05
        self._wheel_dia = 0.02
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

        dx = self._goal[0] - self._pos_x
        dy = self._goal[1] - self._pos_y
        dist_o = (dx ** 2 + dy ** 2) ** 0.5

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
        #energy_consumption = self._force * abs(l_dist) + self._force * abs(r_dist)
        reward = -dist #10.0 * (dist_o - dist) - energy_consumption - self._time_cost
        #if(dist < 0.02 and done):
        #    reward += 1.20
        self._score += reward
        obs_sig = signals(dist, self._signal[0], 0.02, self._signal[1], self._noise)
        return done, [], {"dist":obs_sig, "goal_direction":rel_goal_theta, "true_dist":dist, "reward":reward, "steps":self._step, "done":done, "position":[self._pos_x, self._pos_y], "direction": self._direction}

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
        obs_dist = dist * (random.uniform(-1.0 * self._noise, self._noise) + 1.0)
        return {"dist":obs_dist, "true_dist":dist, "is_dist_valid":1, "steps":0, "done":False, "position":[self._pos_x, self._pos_y], "direction": self._direction}

    def need_learning(self):
        return True

    def train_stage(self):
        return True

if __name__=="__main__":
    game = WheeledRobot(100)
    score_list = []
    for i in range(15):
        score_all = 0.0
        for _ in range(1600):
            pattern = gen_pattern()
            game.reset(pattern, "TRAIN")
            done = False
            info = game.default_info()
            steps = 0.0
            while not done:
                sin_dr = (pattern[1] - info["position"][1]) / info["true_dist"]
                cos_dr = (pattern[0] - info["position"][0]) / info["true_dist"]
                sin_cur_dr = sin(info["direction"])
                cos_cur_dr = cos(info["direction"])
                sin_deta = sin_dr * cos_cur_dr - cos_dr * sin_cur_dr
                cos_deta = cos_dr * cos_cur_dr + sin_dr * sin_cur_dr
                d_theta = asin(sin_deta)
                x = abs(d_theta) * game._wheel_width
                max_dist = game._dt * game._wheel_dia * game._max_wheel_rotation
                ub = min(1.0, info["true_dist"] / max_dist)
                if(d_theta > 0):
                    r = ub
                    l = min(1.0, max(-1.0, ub - x / (max_dist)))
                else:
                    l = ub
                    r = min(1.0, max(-1.0, ub - x / (max_dist)))

                done, next_obs, info = game.step([l,r])
                steps += 1.0
            score_all += game.score
        score_list.append(score_all / 1600)
    print(score_list)
    score_list = numpy.mean(numpy.reshape(score_list, [3,5]), axis = -1)
    print(numpy.mean(score_list))
    print(numpy.std(score_list))
