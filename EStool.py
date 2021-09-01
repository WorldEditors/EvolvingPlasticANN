"""
Tookits for Evolution Strategies
"""

import pickle
import sys
import os
import time
from copy import deepcopy
import numpy
import numpy.random as random

def compute_ranks(x):
    """
    Returns rank as a vector of len(x) with integers from 0 to len(x)
    """
    assert x.ndim == 1
    ranks = numpy.empty(len(x), dtype=int)
    ranks[x.argsort()] = numpy.arange(len(x))
    return ranks

def compute_centered_ranks(x):
    """
    Maps x to [-0.5, 0.5] and returns the rank
    """
    eff_x = numpy.array(x, dtype="float32")
    y = compute_ranks(eff_x.ravel()).reshape(eff_x.shape).astype(numpy.float32)
    y /= (eff_x.size - 1)
    y -= .5
    return y

def smooth_toward(s, s_t, s_b, lr=0.05):
    ret = dict()
    for key in s:
        ret[key] = s[key] + 1.0e-6  # increase 1.0e-6 every time
        for s_t_i, s_b_i in zip(s_t, s_b):
            if(key in s_t_i and key in s_b_i):
                ret[key] += lr * (s_t_i[key] - s_b_i[key])
            else:
                print("warning: key %s is not in the noise factor, neglect it"%key)
    return ret

def noisy_sigma(sigmas):
    ret_dict = dict()
    for key in sigmas:
        ret_dict[key] = sigmas[key] * random.choice([0.90, 1.10])
    return ret_dict

def categorical(p):
    res = numpy.asarray(p)
    return (res.cumsum(-1) >= numpy.random.uniform(size=res.shape[:-1])[..., None]).argmax(-1)

def sort_idx(score_list):
    #mean - variance normalization
    scores = list(zip(score_list, range(len(score_list))))
    scores.sort(reverse=True)
    return  [scores[i][1] for i in range(len(scores))]

def partition(data_list,begin,end):
    partition_key = data_list[end]
    index = begin 
    for i in range(begin,end):
        if data_list[i] < partition_key: 
            data_list[i],data_list[index] = data_list[index],data_list[i] 
            index+=1
    data_list[index],data_list[end] = data_list[end],data_list[index] 
    return index

def find_top_k(data_list,K):
    length = len(data_list)
    if(K > length):
        return numpy.min(data_list)
    begin = 0
    end = length-1
    index = partition(data_list,begin,end)
    while index != length - K:
        if index > length - K:
            end = index-1
            index = partition(data_list,begin,index-1)
        else:
            begin = index+1
            index = partition(data_list,index+1,end)
    return data_list[index]

def add_params(p_1, p_2):
    res = dict()
    for key in p_1.keys():
        if(key in p_2):
            res[key] = p_1[key] + p_2[key]
        else:
            res[key] = numpy.copy(p_1[key])
    return res

def diff_params(p_1, p_2):
    res = dict()
    for key in p_1.keys():
        if(key in p_2):
            res[key] = p_1[key] - p_2[key]
        else:
            res[key] = numpy.copy(p_1[key])
    return res

def multiply_params(p, factor):
    res = dict()
    for key in p:
        res[key] = factor * p[key]
    return res

def noise_like(parameters, noise_factor):
    noise = dict()
    for key in parameters:
        if(key in noise_factor):
            noise[key] = noise_factor[key] * numpy.random.normal(size=numpy.shape(parameters[key]), loc=0.0, scale=1.0) 
    return noise

def rescale(params, scale):
    noise = dict()
    for key in params:
        if(key in scale):
            size = numpy.product(params[key].shape)
            ratio = scale[key] / numpy.sqrt(numpy.mean(params[key] * params[key]) / size + 1.0e-8)
            noise[key] = params[key] * ratio
    return noise

def check_validity(parameters):
    for key in parameters:
        if(numpy.sum(numpy.isnan(parameters[key])) > 0 or numpy.sum(numpy.isinf(parameters[key])) > 0):
            return False
    return True

class ESTool(object):
    def __init__(self, population_size, noise_factor, learning_rate):
        self._sample_size = (population_size - 1) // 2
        self._init_noise_factor = deepcopy(noise_factor)
        self._noise_factor = noise_factor
        self._top_k = population_size // 25
        self._lr = learning_rate
        self._step_size_min = 1.0e-5
        self._cur_step = 0

    def init_popultation(self, seed_para):
        self._evolution_pool = []
        self._cur_param = multiply_params(seed_para, 1.0)
        for _ in range(self._sample_size):
            step_size = noisy_sigma(self._noise_factor)
            epsilon = noise_like(self._cur_param, step_size)
            self._evolution_pool.append([add_params(self._cur_param, epsilon), epsilon, -1e+10, deepcopy(step_size)])
            self._evolution_pool.append([diff_params(self._cur_param, epsilon), multiply_params(epsilon, -1.0), -1e+10, deepcopy(step_size)])

    def evolve(self, verbose=False):
        #remove nan
        start_time = time.time()
        self._cur_step += 1
        for i in range(len(self._evolution_pool)-1, -1, -1):
            if(numpy.isnan(self._evolution_pool[i][2]) or numpy.isinf(self._evolution_pool[i][2])):
                del(self._evolution_pool[i])

        score_pool = [self._evolution_pool[i][2] for i in range(len(self._evolution_pool))]
        fitnesses = compute_centered_ranks(score_pool)
        if(callable(self._lr)):
            fitnesses = self._lr(self._cur_step) * (fitnesses - fitnesses.mean()) / fitnesses.std()
        else:
            fitnesses = self._lr * (fitnesses - fitnesses.mean()) / fitnesses.std()

        for i in range(len(self._evolution_pool)):
            self._cur_param = add_params(self._cur_param, multiply_params(self._evolution_pool[i][1], fitnesses[i]))
        score, top_n_idxes, sorted_idxes = self.stat_top_k(self._top_k)
        bot_n_idxes = sorted_idxes[-self._top_k:]

        top_steps = [self._evolution_pool[i][3] for i in top_n_idxes]
        bot_steps = [self._evolution_pool[i][3] for i in bot_n_idxes]

        self._noise_factor = smooth_toward(self._noise_factor, top_steps, bot_steps, lr=0.50/self._top_k) 

        for key in self._noise_factor:
            self._noise_factor[key] = max(self._step_size_min, self._noise_factor[key])

        self._evolution_pool.clear()
        for _ in range(self._sample_size):
            step_size = noisy_sigma(self._noise_factor)
            epsilon = noise_like(self._cur_param, step_size)
            self._evolution_pool.append([add_params(self._cur_param, epsilon), epsilon, -1e+10, deepcopy(step_size)])
            self._evolution_pool.append([diff_params(self._cur_param, epsilon), multiply_params(epsilon, -1.0), -1e+10, deepcopy(step_size)])

        finish_time = time.time()
        if(verbose):
            numpy.set_printoptions(precision=3, suppress=True)
            print("%s, step choice: %s, calculate time consumption: %.1f, top scores: %s" % 
                    (time.asctime(time.localtime(time.time())), self._noise_factor, 
                        finish_time - start_time, score))
            sys.stdout.flush()
        #self._lr *= 0.999

        return False

    @property
    def pool_size(self):
        return (len(self._evolution_pool))

    def get_weights(self, i):
        return self._evolution_pool[i][0]

    def get_seed_weight(self, i):
        return diff_params(self._evolution_pool[i][0], self._evolution_pool[i][1])

    def set_score(self, i, score):
        self._evolution_pool[i][2] = score

    def load(self, file_name, neural_structure):
        file_op = open(file_name, "rb")
        self._evolution_pool = pickle.load(file_op)
        try:
            self._noise_factor = pickle.load(file_op)
        except Exception:
            print("noise factor not in model file, skip")
        file_op.close()
        self._cur_param = diff_params(self._evolution_pool[0][0], self._evolution_pool[0][1])
        for i in range(len(self._evolution_pool)-1, -1, -1):
            if(not check_validity(self._evolution_pool[i][0])):
                del self._evolution_pool[i]

    def save(self, file_name):
        file_op = open(file_name, "wb")
        pickle.dump(self._evolution_pool, file_op)
        pickle.dump(self._noise_factor, file_op)
        file_op.close()

    def stat_top_k(self, k):
        score_pool = [self._evolution_pool[i][2] for i in range(len(self._evolution_pool))]
        sorted_idx = sort_idx(score_pool)
        return numpy.mean(numpy.asarray(score_pool, dtype="float32")[sorted_idx[:k]]), sorted_idx[:k], sorted_idx

    def stat_avg(self):
        score_pool = [self._evolution_pool[i][2] for i in range(len(self._evolution_pool))]
        return numpy.mean(numpy.asarray(score_pool, dtype="float32"))
