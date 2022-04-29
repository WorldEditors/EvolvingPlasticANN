from numpy import random
import numpy
import pickle
import sys

from envs.env_maze import MazeTask
from envs.env_maze import gen_pattern as gen_single

def dump_pattern(pattern_number, cell_scale, file_name):
    handle = open(file_name, "wb")
    value = [gen_single(cell_scale=cell_scale, crowd_ratio=0.10) for _ in range(pattern_number)]
    pickle.dump(value, handle)
    handle.close()

def load_pattern(file_name):
    handle = open(file_name, "rb")
    value = pickle.load(handle)
    handle.close()
    return value

def gen_patterns(n=16, cell_scale=11, file_name=None, crowd_ratio=None):
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

def resample_maze15(n):
    return gen_patterns(n=n, cell_scale=15, crowd_ratio=0.35)

def resample_maze9(n):
    return gen_patterns(n=n, cell_scale=9, crowd_ratio=0.30)

if __name__=="__main__":
    if(len(sys.argv) < 4):
        print("Usage: %s number_pattern cell_scale file_name"%(sys.argv[0]))
        sys.exit(0)
    dump_pattern(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
