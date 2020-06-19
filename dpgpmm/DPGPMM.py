

import numpy as np
from loguru import logger
import time

from .DPMixture import DPMixture


class Config_Parser(object):
    def __init__(self, dpgp_config):
        self.alpha = dpgp_config['alpha']
        self.ada_alpha = dpgp_config['ada_alpha']

        self.merge = dpgp_config['merge']
        self.merge_threshold = dpgp_config['merge_threshold']
        self.merge_burnin = dpgp_config['merge_burnin']
        self.window_prob = dpgp_config["window_prob"]
        self.self_prob = dpgp_config["self_prob"]

        self.state_dim = dpgp_config['state_dim']
        self.action_dim = dpgp_config['action_dim']
        self.lr = dpgp_config['lr']
        self.gp_iter = dpgp_config['gp_iter']

        self.model_type = dpgp_config['model_type']
        self.max_inducing_point = dpgp_config['max_inducing_point']
        self.trigger_induce = dpgp_config['trigger_induce']
        self.sample_number = dpgp_config['sample_number']
        self.param = dpgp_config['param']


class DPGPMM:
    name = 'DPGPMM'
    
    def __init__(self, dpgp_config):
        # initiate the mixture model
        args = Config_Parser(dpgp_config=dpgp_config)

        # choose GP model type
        if args.model_type == 'exact':
            from .GPComponent_pytorch_cartpole_exact import GPComponent
        elif args.model_type == 'sparse':
            from .GPComponent_pytorch_cartpole_sparse import GPComponent
        elif args.model_type == 'sample':
            from .GPComponent_pytorch_cartpole_sample import GPComponent
        elif args.model_type == 'normalize':
            from .GPComponent_pytorch_cartpole_normalize import GPComponent
        elif args.model_type == 'test':
            from .GPComponent_pytorch_cartpole_test import GPComponent
        else:
            logger.error('No such model type')
        self.DP_mix = DPMixture(GPComponent, args)

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

    def fit(self, data, comp_trainable=None, inference='VI', discret=False):
        # given new data point, train the mixture model
        time_record = []
        # add one data at a time
        data_point, label = self.data_process(data)
        self.DP_mix.add_point(data_point)
        start_time = time.time()
        if inference == 'VI':
            # alpha = self.DP_mix.sequential_vi()
            alpha = self.DP_mix.sequential_vi_w_transition(comp_trainable=comp_trainable)
        elif inference == 'GS':
            alpha = self.DP_mix.gibbs_sample(n_iter=1)
        time_record.append(time.time() - start_time)

        data_per_class = []
        for d_i in self.DP_mix.comps:
            data_per_class.append(d_i.n)
        print("label: {}, predict: {}, data size per cluster: {}".format(label, self.DP_mix.assigns[len(self.DP_mix.data)-1], data_per_class))
        #np.save('./gym_data.npy', self.DP_mix.data)
        return time_record

    def predict_old(self, index, s, a):
        x = np.concatenate((s, a), axis=1)
        # predict the next state given the mixture model, current state x, and previous step index
        sample_func = self.DP_mix.comps[index].predict(x)
        ds = []
        for i in range(self.DP_mix.args.state_dim):
            ds.append(sample_func[i].loc.cpu().numpy())
        ds = np.array(ds).T
        return ds

    def predict(self, s, a):
        '''
        # todo, reuse the history component
        if (len(self.DP_mix.index_chain) > 1) and (self.DP_mix.comps[-1].n < 50):
            self.index = self.DP_mix.index_chain[-2]
        else:
            self.index = self.DP_mix.assigns[len(self.DP_mix.data)-1]
        '''

        self.index = self.DP_mix.assigns[len(self.DP_mix.data)-1]
        x = np.concatenate((s, a), axis=1)
        # predict the next state given the mixture model, current state x, and previous step index
        ds = self.DP_mix.comps[self.index].predict(x)
        return ds
