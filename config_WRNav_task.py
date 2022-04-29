#!/usr/bin/env python
# coding=utf8
# File: config.py
import os
import math
import numpy

#from env_wrn import NavigatorTask
#import env_wrn
from env_robot_navigating import WRNav
from models_zoo import ModelPFCBase1, ModelRNNBase1, ModelLSTMBase1, ModelPRNNBase1

root = "./results/"
directory = root + "/workspace_WRNav/"
ent_factor = 1.0e-6
horizons = 100
noise = 0.0

# Model Structure for EPRNN
model = ModelPRNNBase1(input_shape=(3,), output_shape=(2,), hidden_size=64, extra_hidden_size=64, output_activation="tanh", initialize_settings='P', plasticity_type="SABCD")
#model = ModelPFCBase1(input_shape=(3,), output_shape=(2,), hidden_size=64, extra_hidden_size=64, output_activation="tanh", initialize_settings='P', plasticity_type="SABCD")
#model = ModelRNNBase1(input_shape=(3,), output_shape=(2,), hidden_size=64, extra_hidden_size=64, output_activation="tanh")
#model = ModelLSTMBase1(input_shape=(3,), output_shape=(2,), hidden_size=64, extra_hidden_size=64, output_activation="tanh")

# Refer to config_SeqPred_task.py for other configurations

#If no load_model is specified, the model is random initialized
#load_model = model.dat

#Address for xparl servers, do "xparl start " in your server
#server = "localhost:8010"
server = "10.216.186.18:8010"

#True Batch size = Actor_number * batch_size
actor_number = 410
batch_size = 1
task_sub_iterations = 1
#Each Element includes weight of the episode in calculating score, "TEST/TRAIN" information that is given to the environments, and Train=True/False information given to the inner_loop
inner_rollouts = [(1.0, "TEST", True)]

#The task pattern are kept still for that much steps
pattern_renew = 4
pattern_retain_iterations = 1

#Select the inner-loop type, for PRNN / RNN / LSTM / EPMLP select "forward", for ES-MAML select "policy_gradient_continuous"
adapt_type = "forward"
#adapt_type = "policy_gradient_continuous"

evolution_pool_size = 400
learning_rate = 0.02

save_iter = 100
max_iter = 15000
#Intervals for meta-testing
test_iter = 100

#Transform from raw output to actions, use is_tra if it is different in meta-training-train and meta-training-test
def output_to_action(output_list, is_tra):
    return output_list.tolist()

#Transform the observation, previous action, and info into observation, pattern is not allowed to use in meta-learning
def obs_to_input(obs, action, info, pattern):
    return list(action) + list(obs), None

#Sampled Tasks for meta-training
def gen_pattern(pattern_number=4):
    return [WRNav.gen_pattern() for _ in range(pattern_number)]

#Sampled Tasks for meta-testing
def test_patterns(pattern_number=1600):
    return [WRNav.gen_pattern() for _ in range(pattern_number)]

def game():
    return WRNav(horizons, noise)

# This reward normalization is specially for ES-MAML
GAMMA_ARRAY_ST = numpy.logspace(0, 9, 10, base=0.90, endpoint=0)
def rewards2adv(reward_list):
    length = len(reward_list)
    add_len = GAMMA_ARRAY_ST.shape[0]
    reward_list.extend([reward_list[-1] for _ in range(add_len)])
    adv = [numpy.sum(numpy.asarray(reward_list[i:i+add_len]) * GAMMA_ARRAY_ST) for i in range(length)]
    return list(adv)
