import pickle
import sys
import os
import time
from copy import deepcopy
import numpy
import numpy.random as random

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


class AdaptGA(object):
    def __init__(self, eliminate_ratio, pool_size, step_size):
        self._max_size = pool_size
        self._kept_size = int((1 - eliminate_ratio) * self._max_size)
        self._grad_size = self._kept_size // 2

        self._step_size_min = 5.0e-5
        self._step_size = step_size
        self._backup_step_size = None
        self._Z_factor = 0
        for i in range(self._grad_size):
            self._Z_factor += 1.0 / (i + 1)

    def init_popultation(self, seed_para):
        self._evolution_pool = []
        self._evolution_pool.append([seed_para, 0, 0.0, deepcopy(self._step_size), True])
        for _ in range(self._max_size - 1):
            params, src = self.mutation(seed_para, None, p_n=1.0, p_g=0.0)
            self._evolution_pool.append([params, 0, 0.0, src, True])

    def over_sampler(self, i, total_number):
        ratio = total_number / self._Z_factor / (i + 1)
        count = 0
        while count < ratio:
            if(random.random() < ratio - count):
                yield True
            count += 1

    def update_step_size(self):
        print("top_k", self._top_k_increase, "backup", self._backup_step_size, "cont top 1", self._cont_new_top_1)
        if(numpy.sum(self._top_k_increase) > 2 and self._backup_step_size is None and self._cont_new_top_1 > 1):
            self.release_step_size()
        elif(self._backup_step_size is not None and self._cont_new_top_1 > 0):
            self.accept_step_size()
        elif(self._backup_step_size is not None and self._cont_new_top_1 <= 0):
            self.restore_step_size()
        elif(numpy.sum(self._top_k_increase) < 2 and self._cont_new_top_1 <= 0):
            self.shrink_step_size()

    def mutation(self, weights, grad, p_n=0.8, p_g=0.1):
        p = random.random()
        step_size = dict()
        for key in self._step_size:
            step_size[key] = self._step_size[key] * numpy.clip(numpy.random.exponential(1.0), 1.0, 1.6)
        if(p < p_n):
            fin_noises = noise_like(weights, step_size)
            ret_w = add_params(weights, fin_noises)
        elif(p < p_g + p_n):
            fin_grads = rescale(grad, step_size)
            ret_w = add_params(weights, fin_grads)
        else:
            alpha = random.random()
            fin_noises = noise_like(weights, step_size)
            fin_grads = rescale(grad, step_size)
            ret_w = add_params(weights, multiply_params(fin_noises, alpha))
            ret_w = add_params(ret_w, multiply_params(fin_grads, 1.0 - alpha))

        return ret_w, step_size

    def evolve(self, verbose=False):
        start_time = time.time()
        #remove nan
        for i in range(len(self._evolution_pool)-1, -1, -1):
            if(numpy.isnan(self._evolution_pool[i][2]) or numpy.isinf(self._evolution_pool[i][2])):
                del(self._evolution_pool[i])

        score_pool = [self._evolution_pool[i][2] for i in range(len(self._evolution_pool))]
        sorted_idx = sort_idx(score_pool)

        priority_lives = []
        priority_scores = []

        committee_top = sorted_idx[:self._grad_size]
        committee_bot = sorted_idx[self._grad_size:(2 * self._grad_size)]

        newly_top_comm = 0
        for top_idx in committee_top:
            if(len(self._evolution_pool[top_idx]) > 3 and self._evolution_pool[top_idx][4]):
                newly_top_comm += 1
        newly_top_comm /= len(committee_top)

        grad = None
        deta_rs = [1.0e-6]
        rank = 0
        top_idx = sorted_idx[0]

        if(len(self._evolution_pool[top_idx]) > 3 and self._evolution_pool[top_idx][4]):
            #Learn from newly appeared best step size
            for key in self._step_size:
                self._step_size[key] = 0.80 * self._step_size[key] + 0.20 * self._evolution_pool[top_idx][3][key]
                self._step_size[key] = max(self._step_size_min, self._step_size[key])
        if(newly_top_comm < 0.4):
            for key in self._step_size:
                self._step_size[key] = max(self._step_size_min, 0.95 * self._step_size[key])

        for top_idx, bot_idx in zip(committee_top, committee_bot):
            #stat src
            rank += 1
            self._evolution_pool[top_idx][1] += 1
            self._evolution_pool[top_idx][1] = max(4, self._evolution_pool[top_idx][1])
            priority_lives.append(self._evolution_pool[top_idx][1])
            priority_scores.append(self._evolution_pool[top_idx][2])
            #stat gradients
            epsilon = diff_params(self._evolution_pool[top_idx][0], self._evolution_pool[bot_idx][0])
            deta_r = self._evolution_pool[top_idx][2] - self._evolution_pool[bot_idx][2]
            if(grad is None):
                grad = multiply_params(epsilon, deta_r)
            else:
                grad = add_params(grad, multiply_params(epsilon, deta_r))
            deta_rs.append(deta_r)
        grad = multiply_params(grad, 1.0 / numpy.sum(deta_rs))

        dead = 0
        # Selection
        kept_idxes = set(sorted_idx[:self._kept_size])
        for idx in range(len(self._evolution_pool)):
            if(len(self._evolution_pool[idx]) > 3):
                self._evolution_pool[idx][4] = False # Mark those old generations
            if(idx not in kept_idxes):
                self._evolution_pool[idx][1] -= 5
                self._evolution_pool[idx][1] = min(9, self._evolution_pool[idx][1])
            if(self._evolution_pool[idx][1] < 0):
                dead += 1
        exp_child = self._max_size + dead - len(self._evolution_pool)
        # Mutation
        #print("exp_child:", exp_child, "dead:", dead, "evolution_pool:", len(self._evolution_pool))
        for i in range(len(committee_top)):
            idx = committee_top[i]
            for _ in self.over_sampler(i, exp_child):
                params, src = self.mutation(self._evolution_pool[idx][0], grad)
                if(check_validity(params)):
                    self._evolution_pool.append([params, 0, 0.0, src, True])
                
        # elimination
        for idx in range(len(self._evolution_pool)-1, -1, -1):
            if(self._evolution_pool[idx][1] < 0):
                del self._evolution_pool[idx]

        finish_time = time.time()

        # output some information
        if(verbose):
            numpy.set_printoptions(precision=3, suppress=True)
            print("%s,  newly_top_comm: %.2f, step choice: %s, priority_lives %s, calculate time consumption: %.1f, top scores: %s" % 
                    (time.asctime(time.localtime(time.time())), newly_top_comm, self._step_size, priority_lives, 
                        finish_time - start_time, ",".join(map(lambda x:"%.3f"%x, priority_scores)))
                    )
            sys.stdout.flush()

        return

    @property
    def pool_size(self):
        return (len(self._evolution_pool))

    def get_weights(self, i):
        return self._evolution_pool[i][0]

    def set_score(self, i, score):
        self._evolution_pool[i][2] = score

    def load(self, file_name, neural_structure):
        file_op = open(file_name, "rb")
        self._evolution_pool = pickle.load(file_op)
        file_op.close()
        for i in range(len(self._evolution_pool)-1, -1, -1):
            if(not check_validity(self._evolution_pool[i][0])):
                del self._evolution_pool[i]
            else:
                # Give an initial weight of 3
                self._evolution_pool[i][1] = 3

    def save(self, file_name):
        file_op = open(file_name, "wb")
        pickle.dump(self._evolution_pool, file_op)
        file_op.close()

    def stat_top_k(self, k):
        score_pool = [self._evolution_pool[i][2] for i in range(len(self._evolution_pool))]
        sorted_idx = sort_idx(score_pool)
        return numpy.mean(numpy.asarray(score_pool, dtype="float32")[sorted_idx[:k]])
