import numpy.random as random
import numpy
from adaptGA import AdaptGA
from copy import deepcopy

class TestProblem(object):
    """
    A Test Problem for Black Box Optimization
    """
    def __init__(self):
        #parameter name
        self._param_keys = ["p1", "p2", "p3", "p4"]
        #parameter dimension
        self._param_dims = [30, 20, 10, 4]
        #parameter scales
        self._param_scales = [1.0, 1.0e-1, 1.0e-2, 1.0e-3]
        #parameter importance
        self._param_fitness_whts = {"p1":1, "p2":10, "p3":100, "p4":1000}
        self._fitness_noise = 1.0e-6
        self.reset()

    def reset(self):
        #target parameters
        self._target_params = dict()
        #initial parameters
        self._init_params = dict()
        for key,dim,scale in zip(self._param_keys, self._param_dims, self._param_scales):
            self._target_params[key] = random.normal(size=(dim,dim), loc=0.0, scale=scale)
            self._init_params[key] = numpy.zeros(shape=(dim,dim))
    
    @property
    def init_params(self):
        return deepcopy(self._init_params)

    def fitness_score(self, params):
        # fitness = distance between params and target params
        score = 1.0
        for key in self._target_params:
            assert (key in params) and (numpy.shape(params[key]) == numpy.shape(self._target_params[key])), "input params for fitness do not match the target"
            error = self._target_params[key] - params[key]
            score +=  - self._param_fitness_whts[key] * numpy.mean(error ** 2) + random.normal(scale=self._fitness_noise)
        return score

if __name__=='__main__':
    #The problem to be optimized
    TaskHandle = TestProblem()
    #init parameter
    init_parameter = TaskHandle.init_params
    #set the initial noise for each parameters, suggest 0.001~0.1
    init_noise_factor = dict()
    for key in init_parameter:
        init_noise_factor[key] = 0.01
    #init Evolution Tool
    EvoHandle = AdaptGA(
        0.5, # The ratio to be kept to next generation in GA, typically 0.1 ~ 0.6
        100, # Pool Size for Genetic Algorithms
        init_noise_factor)

    #must call before the evolution iteration
    #The rules of init parameter:
    #dicts of {"key1":val1, "key2":val2, "key3":val3}, val1~val3 must be numpy arrays 
    EvoHandle.init_popultation(init_parameter)

    #the main optimization loop
    iteration = 0
    top_k_fitness = 0
    while iteration < 10000 and top_k_fitness < 0.99:
        iteration += 1
        avg_score = []
        #iterate over the pool and set the fitness score
        for i in range(EvoHandle.pool_size):
            params = EvoHandle.get_weights(i)
            EvoHandle.set_score(i, TaskHandle.fitness_score(params))
        #stat the top 3
        top_k_fitness = EvoHandle.stat_top_k(3)
        print("iteration:", iteration, "Top_3_fitness:", top_k_fitness)
        #do one step evolution
        EvoHandle.evolve()
    print("Costed iteartions:", iteration)
    # get the top params
    top_params = EvoHandle.get_weights(0)
