import pyswarms as ps
from test_evolution import ToyDemo
from pyswarms.utils.functions import single_obj as fx
# Set-up hyperparameters
options = {'c1': 0.01, 'c2': 0.01, 'w':0.5}
# Call instance of PSO
optimizer = ps.single.GlobalBestPSO(n_particles=400, dimensions=800, options=options)

demo = ToyDemo()

# Fit Function
def fit_func(w):
    scores = []
    for wht in w:
        scores.append(-demo.get_score(wht))
    return scores

# Perform optimization
best_cost, best_pos = optimizer.optimize(fit_func, iters=1000)
