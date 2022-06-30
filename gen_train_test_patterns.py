from numpy import random
import numpy
import pickle
import sys

#from envs.env_maze import MazeTask
#from envs.env_maze import gen_task as gen_maze
from envs.env_ant import AntTask
from envs.env_ant import gen_task as gen_ant
from envs.env_humanoid import HumanoidTask
from envs.env_humanoid import gen_task as gen_humanoid

def dump_mazes(pattern_number, cell_scale, file_name):
    handle = open(file_name, "wb")
    value = [gen_single(cell_scale=cell_scale, crowd_ratio=0.35) for _ in range(pattern_number)]
    pickle.dump(value, handle)
    handle.close()

def load_mazes(file_name):
    handle = open(file_name, "rb")
    value = pickle.load(handle)
    handle.close()
    return value

def import_mazes(n=16, cell_scale=11, file_name=None, crowd_ratio=None):
    if(file_name is None):
        if(crowd_ratio is not None):
            return [gen_single(cell_scale=cell_scale, crowd_ratio=crowd_ratio) for _ in range(n)]
        else:
            return [gen_single(cell_scale=cell_scale) for _ in range(n)]
    else:
        patterns = load_pattern(file_name)
        size = len(patterns)
        if(size < n):
            return patterns
        else:
            idxes = numpy.arange(size, dtype="int32")
            random.shuffle(idxes)
            ret = []
            for i in idxes[:n]:
                ret.append(patterns[idxes[i]])
            return ret

def resample_maze21(n):
    return gen_maze(n=n, cell_scale=21, crowd_ratio=0.35)

def resample_maze15(n):
    return gen_maze(n=n, cell_scale=15, crowd_ratio=0.35)

def resample_maze9(n):
    return gen_maze(n=n, cell_scale=9, crowd_ratio=0.30)

def get_ant(task_type="TRAIN", id=0):
    if(task_type == "TRAIN"):
    	return "ant_var_tra_%03d.xml"%id
    elif(task_type == "TEST"):
    	return "ant_var_tst_%03d.xml"%id
    elif(task_type == "OOD"):
    	return "ant_var_ood_%03d.xml"%id

def import_ants(task_type="TRAIN", num=1):
    if(task_type == "TRAIN"):
        idx = numpy.arange(256, dtype="int32")
        random.shuffle(idx)
    elif(task_type == "TEST"):
        idx = numpy.arange(64, dtype="int32")
        random.shuffle(idx)
    elif(task_type == "OOD"):
        idx = numpy.arange(64, dtype="int32")
        random.shuffle(idx)

    return [get_ant(task_type=task_type, id=i) for i in idx[:num]]

def get_humanoid(task_type="TRAIN", id=0):
    if(task_type == "TRAIN"):
    	return "humanoid_var_tra_%03d.xml"%id
    elif(task_type == "TEST"):
    	return "humanoid_var_tst_%03d.xml"%id
    elif(task_type == "OOD"):
    	return "humanoid_var_ood_%03d.xml"%id

def import_humanoids(task_type="TRAIN", num=1):
    if(task_type == "TRAIN"):
        idx = numpy.arange(256, dtype="int32")
        random.shuffle(idx)
    elif(task_type == "TEST"):
        idx = numpy.arange(64, dtype="int32")
        random.shuffle(idx)
    elif(task_type == "OOD"):
        idx = numpy.arange(64, dtype="int32")
        random.shuffle(idx)

    return [get_humanoid(task_type=task_type, id=i) for i in idx[:num]]
