import time
import copy
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
from loguru import logger
import yaml
from utils import dumb_reward_plot, dumb_predict_error_plot
import datetime

import gym
from baselines.MAML import MAML

ENVS = 'Cartpole'

if ENVS == 'Cartpole':
    sys.path.append('./envs/cartpole-envs')
    import cartpole_envs
    from mpc.mpc_cp import MPC
    CONFIG_NAME = './config/config_swingup.yml'
    meta_rollout_list = [0, 1, 2, 3]  # meta task id may not be the same as real task id
elif ENVS == 'Intersection':
    # intersection envs is difficult to merge into this file, it has a lot different functions
    sys.path.append('./envs/highway-env')
    import highway_env
    from mpc.mpc_is import MPC
    from train_on_highway import state_preprocess_full as state_preprocess
    from train_on_highway import gen_model_input_target_full as gen_model_pair
    CONFIG_NAME = './config/config_intersection.yml'
    meta_rollout_list = [0, 1, 2]
elif ENVS == 'Halfcheetah':
    sys.path.append('./envs/halfcheetah-env')
    import halfcheetah_envs
    from mpc.mpc_hc import MPC
    CONFIG_NAME = './config/config_halfcheetah.yml'
    meta_rollout_list = [0, 1]


def prepare_dynamics(gym_config):
    dynamics_name = gym_config['dynamics_name']
    dynamics_set = []
    for i in range(len(dynamics_name)):
        dynamics_set.append(gym.make(dynamics_name[i]))
    # use pre-defined env sequence
    adapt_task = [dynamics_set[i] for i in gym_config['task_dynamics_list']]
    meta_task = [dynamics_set[i] for i in meta_rollout_list]
    return meta_task, adapt_task


def load_config(config_path="config.yml"):
    if os.path.isfile(config_path):
        f = open(config_path)
        return yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise Exception("Configuration file is not found in the path: "+config_path)


if __name__ == '__main__':
    # dynamic model configuration
    config = load_config(CONFIG_NAME)    
    maml_config = config['maml_config']
    mpc_config = config['mpc_config']
    gym_config = config['gym_config']
    render = gym_config['render']

    # initialize the mixture model
    model = MAML(maml_config=maml_config)
    logger.info('Using model: {}', model.name)
    logger.info('Using environment: {}', ENVS)

    # initial MPC controller
    mpc_controller = MPC(mpc_config=mpc_config)

    # prepare meta and adaptiive task
    meta_task, adapt_task = prepare_dynamics(gym_config)

    task_epi = 0
    log = []
    log_name = datetime.datetime.now()

    """ Meta Learning Stage """
    pretrain_episodes = 10
    for task_idx in range(len(meta_task)):
        meta_env = meta_task[task_idx]
        # data collection
        for epi in range(pretrain_episodes):
            state = meta_env.reset()
            if ENVS == 'Intersection':
                state = state_preprocess(state)
            done = False
            while not done:
                action = meta_env.action_space.sample()
                state_next, reward, done, info = meta_env.step(action)
                if ENVS == 'Intersection':
                    state_next = state_preprocess(state_next)
                    model_input, model_output = gen_model_pair(state, state_next)
                    model.data_process([task_idx, model_input, action, model_output], used_for_adaption=False)
                else:
                    model.data_process([task_idx, state, action, state_next-state], used_for_adaption=False)
                state = copy.deepcopy(state_next)
            model.finish_one_episode()
    model.fit()

    """ Adapt Stage (also train meta model) """
    # while task_epi < gym_config['task_episode']*len(gym_config['dynamics_name']):
    while task_epi < gym_config['task_episode']:
        task_epi += 1
        print('task epi: ', task_epi, ' total: ', gym_config['task_episode'])
        time_task_0 = time.time()
        for subtask_index in range(len(adapt_task)):
            for epi in range(gym_config['subtask_episode']): # each subtask contains a fixed number of episode
                print('subtask: ', subtask_index, ', epi: ', epi)
                # train stage
                O, A, R, acc_reward, done, E = [], [], [], 0, False, []
                time_subtask_0 = time.time()
                acc_reward = 0
                state = adapt_task[subtask_index].reset()
                action = adapt_task[subtask_index].action_space.sample()
                state_next, reward, done, info = adapt_task[subtask_index].step(action)
                if ENVS == 'Intersection':
                    state = state_preprocess(state)
                    state_next = state_preprocess(state_next)
                    model_input, model_output = gen_model_pair(state, state_next)
                    # reset the controller at the beginning of each new dynamic
                    x_range = adapt_task[subtask_index].config["observation"]["features_range"]["x"][1]
                    normalized_goal = adapt_task[subtask_index].goal_pos/x_range
                    goal = np.concatenate((normalized_goal, adapt_task[subtask_index].goal_heading), axis=0)
                    mpc_controller.reset(goal=goal, range=x_range)
                else:
                    mpc_controller.reset()

                while not done:
                    if render: 
                        adapt_task[subtask_index].render()

                    # adapt model
                    if ENVS == 'Intersection':
                        model.data_process([subtask_index, model_input, action, model_output], used_for_adaption=True)
                    else:
                        model.data_process([subtask_index, state, action, state_next-state], used_for_adaption=True)
                    model.adapt()

                    if ENVS == 'Intersection':
                        # check single_vehicle
                        cur_state = state[:, 1:].reshape(1,-1).squeeze()
                        # if single vehicle, no collision check
                        single_vehicle = False
                        if ((cur_state[int(cur_state.shape[0]/2):] == 0)*1).mean() == 1:
                            single_vehicle = True
                        action = mpc_controller.act(task=adapt_task[subtask_index], model=model, state=cur_state, single_vehicle=single_vehicle)
                    else:
                        action = np.array([mpc_controller.act(task=adapt_task[subtask_index], model=model, state=state)])

                    # interact with env
                    state_next, reward, done, info = adapt_task[subtask_index].step(action)

                    if ENVS == 'Intersection':
                        state_next = state_preprocess(state_next)
                        model_input, model_output = gen_model_pair(state, state_next)
                        state_delta = model.predict(model_input[None], action[None])
                        mse_error = 0
                    # [TODO] mpc_hc.act should be modified
                    if ENVS == 'Halfcheetah':
                        action = action[0]
                    state_delta, mse_error = model.test(state[None], action[None], state_next)
                    
                    # print('reward: ', reward)
                    # print('prediction: ', state_delta)
                    # print('groundtruth: ', state_next-state)
                    # print('mse_error: ', mse_error)

                    acc_reward += reward
                    A.append(action)
                    O.append(state_next)
                    R.append(reward)
                    E.append(mse_error)

                    state = copy.deepcopy(state_next)

                    if done:
                        model.finish_one_episode()
                        # model.fit() # online train meta-model
                        mse_error_train = 0
                        mse_error_train_std = 0
                        samples = {
                            "obs": np.array(O),
                            "actions": np.array(A),
                            "rewards": np.array(R),
                            "reward_sum": acc_reward,
                            "mse_error": np.mean(E),
                            "mse_error_std": np.std(E),
                            "mse_error_train": mse_error_train,
                            "mse_error_train_std": mse_error_train_std,
                            "runtime": datetime.datetime.now().strftime("%d-%H-%M")
                        }
                        log.append(samples)
                        path = './misc/log/' + ENVS + '-MAML-' + log_name.strftime("%d-%H-%M") + '.npy'
                        np.save(path, log, allow_pickle=True)

                        # draw some plots
                        if ENVS == 'Cartpole':
                            dumb_reward_plot(path, PREFIX=ENVS, xlim=[0, 200], ylim=[0, 200], y_line1=175, y_line2=175)
                            dumb_predict_error_plot(path, PREFIX=ENVS, ylim=[0, 60])
                        elif ENVS == 'Halfcheetah':
                            dumb_reward_plot(path, PREFIX=ENVS, xlim=[0, 200], ylim=[-100, 100], y_line1=40, y_line2=40)
                            dumb_predict_error_plot(path, PREFIX=ENVS, ylim=[0, 200])
                        elif ENVS == 'Intersection':
                            dumb_reward_plot(path, PREFIX=ENVS+'-'+str(pretrain_episodes) + '-MAML-' + log_name.strftime("%d-%H-%M"), xlim=[0, 40], ylim=[0, 80], y_line1=50, y_line2=50)
                        
                        print('-------------------------------------------------')
                        print('Episode finished, time: ', time.time()-time_subtask_0, ' with acc_reward: ', acc_reward, ' with final reward: ', reward)
                        print('-------------------------------------------------')
