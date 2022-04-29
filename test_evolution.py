import numpy.random as random
import numpy
from epann.EStool import ESTool


class ToyDemo(object):
    def __init__(self):
        self.dims = [200, 200, 200, 200]
        self.scales = [1.0, 1.0e-1, 1.0e-2, 1.0e-3]
        self.biases = [1, 10, 100, 1000]
        self.targets = []
        self.init_para = []
        self.segments = []
        idx = 0
        for dim, scale in zip(self.dims, self.scales):
            self.targets.append(random.normal(size=(dim,), loc=0.0, scale=scale))
            self.init_para.append(numpy.zeros(shape=(dim,)))
            self.segments.append(numpy.ones(shape=(dim,), dtype="int32") * idx)
            idx += 1
        self.init_para = numpy.concatenate(self.init_para, axis=0)
        self.segments = numpy.concatenate(self.segments, axis=0)

    def get_score(self, w):
        #print(w.shape)
        assert len(w.shape) == 1 and w.shape[0] == numpy.sum(self.dims)
        scores = []
        s = 0
        for dim, scale, bias, target in zip(self.dims, self.scales, self.biases, self.targets):
            e = s + dim
            error = target - w[s:e]
            scores.append(- bias * numpy.mean(error ** 2))
            s = e
        return numpy.sum(scores)


if __name__=="__main__":
    demo = ToyDemo()
    EvoHandle = ESTool(100, 50, 0.01, default_cov_lr=0.10, segments=demo.segments)
    EvoHandle.init_popultation(demo.init_para)
    iteration = 0
    while iteration < 10000:
        iteration += 1
        avg_score = []
        for i in range(EvoHandle.pool_size):
            weights = EvoHandle.get_weights(i)
            score = demo.get_score(weights)
            EvoHandle.set_score(i, score + random.normal(scale=0.0))
            avg_score.append(score)
        scores, _ = EvoHandle.stat_top_k(3)
        EvoHandle.evolve(verbose=True)
        if(scores > -0.02):
            break
    print("iteartions:", iteration)
