
## Task-Agnostic Online Reinforcement Learning with an Infinite Mixture of Gaussian Processes

Paper available at: https://arxiv.org/pdf/2006.11441.pdf

### Python3 requirements

```
torch==1.4.0
gpytorch==1.0.1
gym==0.17.1
mujoco_py==2.0.2.9
matplotlib
numpy
scipy
loguru
```

### Environment requirements
The cartpole-swingup environment is provided. Parameters are stored in in ./config/config_swingup.yml. We modify the environment based on Gym.
```angularjs
cd ./envs/cartpole-envs
pip install -e .
```
Besides the python3 requirements, you should also install mujoco 2.0.0 to run the experiments of HalfCheetah. Parameters are stored in in ./config/config_halfcheetah.yml. We modified the HalfCheetah-v1 environment of gym.mujoco.
```angularjs
cd ./envs/halfcheetah-env
pip install -e .
```
The highway-env is provided and can be installed as follows. Parameters are stored in in ./config/config_highway.yml.
```angularjs
cd DPGP-MBRL/envs/highway-env
pip install -e .
```

### Usage
To run our method, GP, NN and NP baselines:
```angularjs
python3 train_on_swingup.py --model GPMM          # 'GPMM', 'NN', 'NP', 'GP'
python3 train_on_halfcheetah.py --model GPMM      # 'GPMM', 'NN', 'NP', 'GP'
python3 train_on_highway.py --model GPMM          # 'GPMM', 'NN', 'NP', 'GP'
```

To run MAML and DPNN baselines:
```angularjs
python3 train_on_all_MAML.py    # change ENVS name in Line 18: 'Cartpole', 'Intersection', 'Halfcheetah'
python3 train_on_all_DPNN.py    # change ENVS name in Line 18: 'Cartpole', 'Intersection', 'Halfcheetah'

```



### Note:

* Results (data and plots) will be stored in ./misc.

* Configuration files contain all the parameter required and are located in ./config.
