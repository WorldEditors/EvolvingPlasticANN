"""
Tookits for Evolution Strategies
"""

import pickle
import sys
import os
import time
from copy import deepcopy
import numpy
import math
import numpy.random as random

FLOAT_MAX = 1.0e+8

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

def check_validity(parameters):
    if(numpy.sum(numpy.isnan(parameters)) > 0 or numpy.sum(numpy.isinf(parameters)) > 0):
        return False
    return True

class ESTool(object):
    def __init__(self, 
            pool_size, 
            top_k_size, 
            initial_sigma,
            default_cov_lr=None,
            max_step_size=1.0,
            min_cov=1.0e-12,
            segments=None
            ):
        self._pool_size = pool_size
        self._init_sigma = deepcopy(initial_sigma)
        self._sigma_t = deepcopy(initial_sigma)
        self._top_k_size = top_k_size
        self._step = 0
        self._default_cov_lr = default_cov_lr
        self.max_step_size = max_step_size
        self.min_step_size = 1.0e-6
        self.min_cov = min_cov
        self.segments = segments
        self.extra_shaping_val = [0.708, 1.0, 1.225]
        self.extra_shaping_prob = [0.05, 0.90, 0.05]

    def sampling_extra_factor(self, segments):
        extra_factor = []
        for _ in range(self.segment_max_idx + 1):
            extra_factor.append(self.extra_shaping_val[categorical(self.extra_shaping_prob)])
        return numpy.array(extra_factor, dtype="float32")[segments]

    def pre_calculate_parameters(self):
        self.w_r_i = numpy.zeros((self._top_k_size,), dtype="float32")
        for i in range(self._top_k_size):
            self.w_r_i[i] = math.log(self._top_k_size + 1) - math.log(i + 1)
            #self.w_r_i[i] = 1 / (i + 1)
        self.w_r_i *= 1.0 / numpy.sum(self.w_r_i)
        self.l_e = 1.0 / numpy.sum(self.w_r_i * self.w_r_i)
        self.p_std = math.sqrt(self.dim) * (1 - 0.25 / self.dim + 1.0 / (21 * self.dim ** 2))
        self.c_sigma = (self.l_e + 2) / (self.dim + self.l_e + 3)
        self.ps_f = math.sqrt(self.c_sigma * (2 - self.c_sigma) * self.l_e)

        if(self.l_e > self.dim + 2):
            self.d_sigma = 1 + 2.0 * math.sqrt((self.l_e - self.dim - 2) / (self.dim + 1)) + self.c_sigma
        else:
            self.d_sigma = 1 + self.c_sigma

        self.c_c = 4 / (self.dim + 4)
        self.l_m = math.sqrt(self.c_c * (2 - self.c_c) * self.l_e)
        self.s_m = math.sqrt(self.c_sigma * (2 - self.c_sigma) * self.l_e)
        self.p_m = math.sqrt(2 * self.c_sigma) * (1.4 + 2.0 / (self.dim + 1)) * self.p_std
        if(self._default_cov_lr is None):
            self.c_cov = 2 / (self.l_e * (self.dim + 1.414) ** 2) + (1 - 1/self.l_e) * min(
                    1, (2 * self.l_e - 1) / ((self.dim + 1.414) ** 2 + self.l_e))
        else:
            self.c_cov = self._default_cov_lr
        self._sqrt_cov = numpy.sqrt(self._cov)

        if(self.segments is None):
            self.segments = numpy.arange(self.dim, dtype="int32")
            self.segment_max_idx = numpy.max(self.segments)
        else:
            self.segment_max_idx = numpy.max(self.segments)

    def generate_new_offspring(self):
        if(self.segments is not None):
            extra_shaping_factor = self.sampling_extra_factor(self.segments)
            deta_weights = self._sigma_t * self._sqrt_cov * extra_shaping_factor * numpy.random.normal(size=(self.dim))
        else:
            deta_weights = self._sigma_t * self._sqrt_cov * numpy.random.normal(size=(self.dim))
        self._evolution_pool.append([self._base_weights + deta_weights, -FLOAT_MAX])

    def init_popultation(self, weights, static_weights=None):
        assert len(weights.shape) == 1, "Can only support vectorized parameters"
        self.dim = weights.shape[0]

        self._evolution_pool = []
        self._pc = numpy.zeros((self.dim, ), dtype="float32")
        self._ps = numpy.zeros((self.dim, ), dtype="float32")
        self._sigma_t = self._init_sigma
        self._cov = numpy.ones((self.dim, ), dtype="float32")
        self._base_weights = numpy.copy(weights)
        if(static_weights is not None):
            self._static_weights = deepcopy(static_weights)
        else:
            self._static_weights = dict()

        self.pre_calculate_parameters()

        for _ in range((self._pool_size - 1)):
            self.generate_new_offspring()
        self._evolution_pool.append([self._base_weights, -FLOAT_MAX])

    def evolve(self, verbose=False):
        #remove nan
        start_time = time.time()
        self._step += 1
        for i in range(len(self._evolution_pool)-1, -1, -1):
            if(numpy.isnan(self._evolution_pool[i][1]) or numpy.isinf(self._evolution_pool[i][1])):
                if(verbose):
                    print("encounter %s in score in index %d, delete from the pool" % (self._evolution_pool[i][1], i))
                del(self._evolution_pool[i])
        if(len(self._evolution_pool) < 1):
            raise Exception("Evolution Pool is empty, something nasty happened (probably too much nans)")

        score_pool = [self._evolution_pool[i][1] for i in range(len(self._evolution_pool))]
        fitnesses = compute_centered_ranks(score_pool)

        score, top_k_idxes = self.stat_top_k(self._top_k_size)
        if(len(top_k_idxes) < self._top_k_size):
            w_r_i = self.w_r_i[:len(top_k_idxes)]
            w_r_i *= 1.0 / numpy.sum(w_r_i)
        else:
            w_r_i = self.w_r_i

        new_base = numpy.zeros_like(self._base_weights)
        deta_base_sq = numpy.zeros_like(self._base_weights)
        for i, idx in enumerate(top_k_idxes):
            new_base += w_r_i[i] * self._evolution_pool[idx][0]
            deta_para = self._evolution_pool[idx][0] - self._base_weights
            deta_base_sq += w_r_i[i] * (deta_para * deta_para)

        # update p_sigma
        base_deta = new_base - self._base_weights
        n_ps = (1 - self.c_sigma)*self._ps + self.s_m / self._sigma_t * numpy.reciprocal(self._sqrt_cov) * base_deta

        #update step size
        ps_norm = numpy.sqrt(numpy.sum(n_ps * n_ps))
        n_sigma_t = self._sigma_t * numpy.exp(self.c_sigma / self.d_sigma * (ps_norm / self.p_std - 1))

        #update p_c
        h_t = float(ps_norm < self.p_m * math.sqrt(self._step + 1))
        n_pc = (1 - self.c_c) * self._pc + h_t * self.l_m / self._sigma_t * base_deta
        
        #update cov
        n_cov = (1 - self.c_cov) * self._cov + self.c_cov / self.l_e * self._pc * self._pc \
                + self.c_cov * (1 - 1/self.l_e) / self._sigma_t / self._sigma_t * deta_base_sq

        self._base_weights = new_base
        self._evolution_pool.clear()

        self._cov = numpy.clip(n_cov, self.min_cov, 1.0)
        self._sigma_t = numpy.clip(n_sigma_t, self.min_step_size, self.max_step_size)
        self._ps = n_ps
        self._pc = n_pc
        self._sqrt_cov = numpy.sqrt(self._cov)

        self._evolution_pool.append([self._base_weights, -FLOAT_MAX])
        for _ in range((self._pool_size - 1)):
            self.generate_new_offspring()

        finish_time = time.time()
        if(verbose):
            numpy.set_printoptions(precision=3, suppress=True)
            print("%s, sqrt_covariances: %.3f, step_size: %.3f, c_cov: %.3f, calculate time consumption: %.1f, top %.0f average scores: %.4f" % 
                    (time.asctime(time.localtime(time.time())), numpy.mean(self._sqrt_cov), self._sigma_t, self.c_cov,
                        finish_time - start_time, self._top_k_size, score))
            sys.stdout.flush()

        return False

    @property
    def pool_size(self):
        return (len(self._evolution_pool))

    def get_weights(self, i):
        return self._evolution_pool[i][0]

    @property
    def get_base_weight(self):
        return self._base_weights

    @property
    def get_static_weights(self):
        if(isinstance(self._static_weights, dict)):
            return self._static_weights
        else:
            return self._static_weights.tolist()

    def set_score(self, i, score):
        self._evolution_pool[i][1] = score

    def load(self, file_name):
        file_op = open(file_name, "rb")
        self._evolution_pool = pickle.load(file_op)
        base_weights = pickle.load(file_op)
        if(isinstance(base_weights, tuple)):
            self._base_weights, self._static_weights = base_weights
        else:
            self._base_weights = base_weights
            self._static_weights = dict()
        self._cov = pickle.load(file_op)
        self._pc = pickle.load(file_op)
        self._ps = pickle.load(file_op)
        self._sigma_t = pickle.load(file_op)
        self.dim = pickle.load(file_op)
        self._step = pickle.load(file_op)
        self._cov = numpy.clip(self._cov, 0.64, 1.0)
        self._sigma_t = deepcopy(self._init_sigma)
        file_op.close()
        for i in range(len(self._evolution_pool)-1, -1, -1):
            if(not check_validity(self._evolution_pool[i][0])):
                del self._evolution_pool[i]
        self.pre_calculate_parameters()

    def save(self, file_name):
        file_op = open(file_name, "wb")
        pickle.dump(self._evolution_pool, file_op)
        pickle.dump((self._base_weights, self._static_weights), file_op)
        pickle.dump(self._cov, file_op)
        pickle.dump(self._pc, file_op)
        pickle.dump(self._ps, file_op)
        pickle.dump(self._sigma_t, file_op)
        pickle.dump(self.dim, file_op)
        pickle.dump(self._step, file_op)
        file_op.close()

    def stat_top_k(self, k):
        score_pool = [self._evolution_pool[i][1] for i in range(len(self._evolution_pool))]
        sorted_idx = sort_idx(score_pool)
        return numpy.mean(numpy.asarray(score_pool, dtype="float32")[sorted_idx[:k]]), sorted_idx[:k]

    def stat_avg(self):
        score_pool = [self._evolution_pool[i][1] for i in range(len(self._evolution_pool))]
        return numpy.mean(numpy.asarray(score_pool, dtype="float32"))
