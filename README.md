# Introduction
This is the meta-training code for paper "Do What Nature Did To Us: Evolving Plastic Recurrent Neural Networks For Task Generalization"

# Requirement
python >= 3.7.4
parl == 1.4.1

# Run Meta-Training
```bash
#Running Sequence Predicting Tasks
python run_ga.py config_SeqPred_task
#Running Wheeled Robot Navigating Tasks
python run_ga.py config_WRNav_task
```

If you are to use parallelization mode, start xparl master on your remote server by using: 
```bash
xparl start --cpu_num $cpu_num --port $port_id
```
and change the "server" configuration to "$IP_ADDRESS:$port_id"
Be sure that "$cpu_num" surpass the "actor_number" in the configuration file
