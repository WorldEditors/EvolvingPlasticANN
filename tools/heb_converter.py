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
        if(key.find("/W_Ax")>=0):
            nkey = key.replace("/W_Ax", "A")
            rep_keys[nkey] = numpy.outer(nn_1._parameters[key.replace("/W_Ax", "/W_Ay")], nn_1.parameters[key])
        elif(key.find("/W_Bx")>=0):
            nkey = key.replace("/W_Bx", "B")
            rep_keys[nkey] = numpy.outer(nn_1._parameters[key.replace("/W_Bx", "/W_By")], nn_1.parameters[key])
        elif(key.find("/W_Cx")>=0):
            nkey = key.replace("/W_Cx", "C")
            rep_keys[nkey] = numpy.outer(nn_1._parameters[key.replace("/W_Cx", "/W_Cy")], nn_1.parameters[key])
        elif(key.find("/W_Dx")>=0):
            nkey = key.replace("/W_Dx", "D")
            rep_keys[nkey] = numpy.outer(nn_1._parameters[key.replace("/W_Dx", "/W_Dy")], nn_1.parameters[key])
        elif(key.find("/W_Ay")>=0 or key.find("/W_By")>=0 or key.find("/W_Cy")>=0 or key.find("/W_Dy")>=0):
            pass
        else:
            nn_2._parameters[key] = nn_1._parameters[key]

    for key in rep_keys:
        nn_2._parameters[key] = rep_keys[key]


def convert(nn_1, nn_2, nn_1_shape, nn_2_shape, nn_1_weights):
    nn_1.from_vector(nn_1_weights, nn_1_shape)
    copy_parameter(nn_1, nn_2) 
    return nn_2.to_vector[0]

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
    nn_2_init_w, nn_2_shape = nn_2.to_vector
    eh_1.load(config.load_model_from)
    eh_2.init_popultation(nn_2_init_w)

    nn_1_weights = eh_1._base_weights
    eh_2._base_weights = convert(nn_1, nn_2, nn_1_shape, nn_2_shape, nn_1_weights)
    eh_2._evolution_pool = []
    for nn_1_weights, value in eh_1._evolution_pool:
        eh_2._evolution_pool.append([ convert(nn_1, nn_2, nn_1_shape, nn_2_shape, nn_1_weights), value ])

    eh_2.save(config.save_model_to)
