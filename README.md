# Introduction
A crucial difference between artificial neural networks (ANNs) and biological neural networks (BNNs) is that BNNs can acquire new skills across variant tasks on their own. Motivated by BNNs, we try to implement the "Learning By Interaction" principle in the meta-learning framework. We aim to unify supervised learning, reinforcement learning, and unsupervised learning in a model-based / plasticity-based manner. The learning no longer relies on human-designed target function and optimization but through the black-box mechanism of the neural networks and plasticity rules. We build this evolving plasticity repo to facilitate the research on this topic.

# Requirement
python >= 3.7.4

parl == 1.4.1

numpy >= 1.8.1

metagym >= 0.1.0

# Run Meta-Training in Random Maze-2D environments
```bash
python run_ga.py config_maze_train
```

# Run Meta-Testing in Random Maze-2D environments
```bash
python run_test.py config_maze_test
```

If you are to use parallelization mode, start xparl master on your remote server by using: 
```bash
xparl start --cpu_num $cpu_num --port $port_id
```
and change the "server" configuration to "$IP_ADDRESS:$port_id".
Also be sure that "$cpu_num" surpass the "actor_number" in the configuration file
