
import numpy as np
from loguru import logger
#import torch
import time

from .DPMixture import DPMixture
from .NN import NNComponent


class Config_Parser(object):
    def __init__(self, dp_config):
        self.alpha = dp_config['alpha']
        self.ada_alpha = dp_config['ada_alpha']
        self.merge = dp_config['merge']
        self.merge_threshold = dp_config['merge_threshold']
        self.merge_burnin = dp_config['merge_burnin']
        self.window_prob = dp_config["window_prob"]
        self.self_prob = dp_config["self_prob"]
        self.DPprior = dp_config["DPprior"]


class DPNNMM:
    name = 'DPNNMM'
    
    def __init__(self, dp_config, nn_config):
        # init the mixture model
        args = Config_Parser(dp_config=dp_config)
        self.nn_config = nn_config
        self.nn_meta_model = NNComponent(NN_config=nn_config)
        self.nn_meta_weight_path = nn_config['model_config']['save_model_path']
        
        self.DP_mix = DPMixture(NNComponent, args, nn_config)

        self.stm = []  # short-term memory
        self.stm_length = dp_config["stm_length"]
        self.meta_load()

    '''
    def meta_fit(self, dataset):
        # dataset format: list of [task_idx, state, action, next_state-state]
        self.nn_meta_model.fit(dataset)
        self.nn_meta_model.save_model(self.nn_meta_weight_path)
    '''
    
    def meta_load(self):
        self.nn_meta_model.load_model(self.nn_meta_weight_path)

    def add_data_point(self, data):
        # data format: [task_idx, state, action, next_state-state]
        self.stm.append(data)

    def fit(self):
        if len(self.stm) == 0:
            logger.error("No data in the memory yet")
        time_record = []
        start_time = time.time()

        task_idx = self.DP_mix.svi(self.stm, self.nn_meta_model)
        self.stm = []
        time_record.append(time.time() - start_time)
        return time_record, task_idx

    def predict(self, s, a):
        '''
        # todo, reuse the history component
        if (len(self.DP_mix.index_chain) > 1) and (self.DP_mix.comps[-1].n < 50):
            self.index = self.DP_mix.index_chain[-2]
        else:
            self.index = self.DP_mix.assigns[len(self.DP_mix.data)-1]
        '''

        self.index = self.DP_mix.assigns
        x = np.concatenate((s, a), axis=1)
        # predict the next state given the mixture model, current state x, and previous step index
        ds, var = self.DP_mix.comps[self.index].predict(s, a)
        return ds

    @property
    def stm_is_full(self):
        return len(self.stm) >= self.stm_length

    @staticmethod
    def data_process(data, discret=False):
        s = data[1]
        if discret:
            a = np.array([data[2]])
        else:
            a = data[2]
        s_n = data[3]

        data_point = np.concatenate((s, a, s_n), axis=0)[None]
        label = data[0]
        # print('data: ', data_point)
        return data_point, label
