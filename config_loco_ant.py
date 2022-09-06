#!/usr/bin/env python
# coding=utf8
# File: config.py
import os
import math
import random
import numpy
from envs.env_ant import AntTask
from envs.env_ant import gen_task as gen_single
from envs.env_ant import output_to_action
from envs.env_ant import obs_to_input
from epann.utils import categorical
from models_zoo import ModelRNNBase1, ModelLSTMBase1, ModelFCBase2
from models_zoo import ModelPRNNPreMod, ModelPRNNAfterMod, ModelPRNNNoMod
from models_zoo import ModelPFCPreMod, ModelPFCAfterMod, ModelPFCNoMod
from gen_train_test_patterns import import_ants

root = "./results"
directory = root + "/workspace_loco_ant/"

# Model Structure for EPRNN
#model = ModelFCBase2(input_shape=(38,), output_shape=(8,), hidden_size=64, output_activation="none", initialize_settings='R', init_scale=0.05)
#model = ModelLSTMBase1(input_shape=(38,), output_shape=(8,), hidden_size=64, extra_hidden_size=64, output_activation="none", init_scale=0.05)
#model = ModelRNNBase1(input_shape=(38,), output_shape=(8,), hidden_size=64, output_activation="none", initialize_settings='R', init_scale=0.05)

#model = ModelPFCNoMod(input_shape=(38,), output_shape=(8,), hidden_size=64, output_activation="none", initialize_settings='R', hebbian_type=3, init_scale=0.05)
#model = ModelPFCPreMod(input_shape=(38,), output_shape=(8,), hidden_size=64, output_activation="none", initialize_settings='R', hebbian_type=3, init_scale=0.05)
#model = ModelPFCAfterMod(input_shape=(38,), output_shape=(8,), hidden_size=64, output_activation="none", initialize_settings='R', hebbian_type=3, init_scale=0.05)

#model = ModelPRNNNoMod(input_shape=(38,), output_shape=(8,), hidden_size=64, output_activation="none", initialize_settings='R', hebbian_type=3, init_scale=0.05)
#model = ModelPRNNPreMod(input_shape=(38,), output_shape=(8,), hidden_size=64, output_activation="none", initialize_settings='R', hebbian_type=3, init_scale=0.05)
# Hebbian-Type: 1. alpha ABCD; 2. Eligibility Traces; 3. Decomposed Plasticity; 4. Evolving & Merging
model = ModelPRNNAfterMod(input_shape=(38,), output_shape=(8,), hidden_size=64, output_activation="none", initialize_settings='R', hebbian_type=3, init_scale=0.05)

# Refer to config_SeqPred_task.py for other configurations

#If no load_model is specified, the model is random initialized
#load_model = root + "demo/models/..."

#Address for xparl servers, do "xparl start " in your server
#server = "10.216.186.20:8010"
server = "localhost:8010"

#True Batch size = Actor_number * batch_size
actor_number = 380
batch_size = 1
task_sub_iterations = 1
inner_rollouts = [(0.0, "TRAIN", True), (0.0, "TRAIN", True), (0.36, "TEST", True), (0.6, "TEST", True), (1.0, "TEST", True)]

#Select the inner-loop type, for PRNN / RNN / LSTM / EPMLP select "forward", for ES-MAML select "policy_gradient_continuous"
adapt_type = "forward"
ent_factor = 1.0e-6

evolution_pool_size = 360
evolution_topk_size = 180
# CMA-ES initial noise variance
evolution_step_size = 0.01
# CMA-ES hyper-parameter
evolution_lr = 0.02

# Model Saving Intervals
save_iter = 500
# Maximum Iterations
max_iter = 15000
# Intervals for meta-testing
test_iter = 50

#Sampled Tasks for meta-training
def train_patterns(n_step=0):
    return import_ants(task_type="TRAIN", num=8)

#Sampled Tasks for meta-testing
def valid_patterns(pattern_number=64):
    return import_ants(task_type="TRAIN", num=64)

def game():
    return AntTask()
