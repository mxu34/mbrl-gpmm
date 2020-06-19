
import numpy as np
import gym
import cenvs
#import cost_functions


def get_env_names(env, training_params, test_params):

    def num_to_str(num):
        return "{:.2f}".format(num).replace(".", "")

    def params_to_env(env, params):
        return ["{}_{}-{}-v0".format(
            env, num_to_str(p1), num_to_str(p2)) for (p1, p2) in params]

    training_envs = params_to_env(env, training_params)
    test_envs = params_to_env(env, test_params)

    return training_envs, test_envs


def initialize_envs(env_names, seed):
    envs = []
    for name in env_names:
        genv = gym.make(name)
        genv.seed(seed)
        envs.append(genv)
    return envs


def close_envs(envs):
    return [env.close() for env in envs]


def get_envs(**kwargs):

    if kwargs["env"] == "PendulumEnv":

        training_env_params = [
            (0.7, 0.7),
            (0.9, 0.9),
            (0.7, 0.9),
            (0.9, 0.7)
        ]

        test_env_params = [
            (0.8, 0.8),
            (1.0, 1.0),
            (0.8, 1.0),
            (1.0, 0.8)
        ]

        training_envs, test_envs = get_env_names(
            kwargs["env"], training_env_params, test_env_params)

        #dataset = data.MultiEnvData_Pendulum()

        target_state = np.float32([1., 0.])
        #cost = cost_functions.PendulumSquaredCost(
        #    dim_states=2, target_state=target_state)
        labels = None

    elif kwargs["env"] == "CartpoleSwingup":

        training_envs = [
            "MTDartCartPoleSwingUp_040-060-v1",
            "MTDartCartPoleSwingUp_040-080-v1",
            "MTDartCartPoleSwingUp_060-060-v1",
            "MTDartCartPoleSwingUp_060-080-v1",
            "MTDartCartPoleSwingUp_080-060-v1",
            "MTDartCartPoleSwingUp_080-080-v1"
        ]

        test_envs = [
            "MTDartCartPoleSwingUp_070-050-v1",
            "MTDartCartPoleSwingUp_070-070-v1",
            "MTDartCartPoleSwingUp_090-050-v1",
            "MTDartCartPoleSwingUp_090-070-v1"
        ]

        #dataset = data.MultiEnvData_Cartpole()

        create_label = lambda m, l: "m={:.1f}, l={:.1f}".format(m, l)
        labels = [
            create_label(0.4, 0.6),
            create_label(0.4, 0.8),
            create_label(0.6, 0.6),
            create_label(0.6, 0.8),
            create_label(0.8, 0.6),
            create_label(0.8, 0.8),
            create_label(0.7, 0.5),
            create_label(0.7, 0.7),
            create_label(0.9, 0.5),
            create_label(0.9, 0.7)
        ]

        target_state = np.float32([-1., 0., 0.])
        #cost = cost_functions.CartpoleSquaredCost(
        #    dim_states=3, target_state=target_state)

    return training_envs, test_envs #, dataset, cost, labels
