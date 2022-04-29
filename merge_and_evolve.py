"""
Runing Meta-Training and Meta-Testing Loops
"""
import sys
import time
import numpy
import parl
import importlib
from time import sleep
from copy import copy, deepcopy 
from utils import make_dir
from utils import add_params, diff_params, multiply_params, mean_params, sum_params, param_max
from inner_loop_agents import *
from EStool import ESTool

def copy_parameter(nn_1, nn_2):
    rep_keys = dict()
    for key in nn_1._parameters:
        if(key in nn_2._parameters):
            nn_2._parameters[key] = nn_1._parameters[key]
    nn_2.l1.heb_h.inherit_parameter(nn_1._parameters["PRNN_1/hebbian_h/A"],
            nn_1._parameters["PRNN_1/hebbian_h/B"],
            nn_1._parameters["PRNN_1/hebbian_h/C"],
            nn_1._parameters["PRNN_1/hebbian_h/D"],
            )
    nn_2.l1.heb_x.inherit_parameter(nn_1._parameters["PRNN_1/hebbian_x/A"],
            nn_1._parameters["PRNN_1/hebbian_x/B"],
            nn_1._parameters["PRNN_1/hebbian_x/C"],
            nn_1._parameters["PRNN_1/hebbian_x/D"],
            )

if __name__=='__main__':
    if(len(sys.argv) < 2):
        print("Usage: %s configuration_file" % sys.argv[0])
        sys.exit(1)

    config_module_name = sys.argv[1].replace(".py", "")
    config = importlib.import_module(config_module_name)

    nn_1 = config.model_from
    nn_2 = config.model_to

    eh_1 = ESTool(
            config.evolution_pool_size, 
            config.evolution_topk_size,
            config.evolution_step_size,
            default_cov_lr=config.evolution_lr,
            segments=nn_1.para_segments
            )

    eh_2 = ESTool(
            config.evolution_pool_size, 
            config.evolution_topk_size,
            config.evolution_step_size,
            default_cov_lr=config.evolution_lr,
            segments=nn_2.para_segments
            )

    _, nn_1_shape = nn_1.to_vector
    eh_1.load(config.load_model_from)
    nn_1.from_vector(eh_1._base_weights, nn_1_shape)
    copy_parameter(nn_1, nn_2)
    nn_2_whts, nn_2_shape = nn_2.to_vector
    print(len(nn_2_whts), nn_2_shape)
    eh_2.init_popultation(nn_2_whts, nn_2.static_parameters)
    eh_2.save(config.save_model_to)
