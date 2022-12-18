# Introduction
Inspired by biological neural networks (BNNs) is that BNNs that acquire new skills across variant tasks on their own, we try to implement "Model-Based Meta learning" or "In-Context Learning" in the nested learning loops. 
The learning no longer relies on human-designed target function and optimization but through the black-box mechanism of the neural networks and plasticity rules. 
Our investigation show that Lifelong In-Context Learning is possible through modeling Hebbian plasticity.
We build this evolving plasticity repo to facilitate the research on this topic.

# Requirement
python >= 3.7.4

parl == 1.4.1

numpy >= 1.8.1

metagym >= 0.2.0.dev2

# Run Meta-Training in Random Maze-2D environments
```bash
python run_train.py config_maze_train
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

Cite this work with
```
@article{
wang2022evolving,
title={Evolving Decomposed Plasticity Rules for Information-Bottlenecked Meta-Learning},
author={Fan Wang and Hao Tian and Haoyi Xiong and Hua Wu and Jie Fu and Yang Cao and Kang Yu and Haifeng Wang},
journal={Transactions of Machine Learning Research},
year={2022},
url={https://openreview.net/forum?id=6qMKztPn0n},
note={}
}
```
