
import numpy as np
from loguru import logger
import time

from .DPMixture import DPMixture
from baselines.SingleGP import SingleGP


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


class SDPGPMM:
    # this model will separate the consistent states and changable states. 
    # Use a single shared GP to learn the consistent states and infite mixture of GP to learn the changable states.
    name = 'SDPGPMM'

    def __init__(self, dpgp_config):
        # initiate the mixture model, use the same parameters as dpgp
        dpgp_config['state_dim'] = 2
        dpgp_config['action_dim'] = 0
        args = Config_Parser(dpgp_config=dpgp_config)

        # choose GP model type
        if args.model_type == 'sparse':
            from .GPComponent_pytorch_cartpole_sparse import GPComponent
        elif args.model_type == 'sample':
            from .GPComponent_pytorch_cartpole_sample import GPComponent
        elif args.model_type == 'test':
            from .GPComponent_pytorch_cartpole_test import GPComponent
        else:
            logger.error('No such model type')
        self.DP_mix = DPMixture(GPComponent, args)

        # use the same parameters as dpgp, but the action dimension is 0
        dpgp_config['state_dim'] = 2
        dpgp_config['action_dim'] = 1
        self.sharedGP = SingleGP(gp_config=dpgp_config) 

    @staticmethod
    def data_process(data):
        label = data[0]
        s = data[1]
        a = data[2]
        s_n = data[3]

        data_point_shared = [label, s[0:2], a, s_n[0:2]]
        data_point_independent = np.concatenate((s[2:4], s_n[2:4]), axis=0)[None]

        return data_point_shared, data_point_independent, label

    def fit(self, data, comp_trainable=None, inference='VI', discret=False):
        # given new data point, train the mixture model
        time_record = []
        # add one data at a time
        data_point_shared, data_point_independent, label = self.data_process(data)

        # train indepedent states (need to process the data again)
        self.sharedGP.fit(data_point_shared)

        # train shared states
        self.DP_mix.add_point(data_point_independent)
        start_time = time.time()
        if inference == 'VI':
            # alpha = DP_mix.sequential_vi()
            alpha = self.DP_mix.sequential_vi_w_transition(comp_trainable=comp_trainable)
        elif inference == 'GS':
            alpha = self.DP_mix.gibbs_sample(n_iter=1)
        time_record.append(time.time() - start_time)

        data_per_class = []
        for d_i in self.DP_mix.comps:
            data_per_class.append(d_i.n)
        print("label: {}, predict: {}, data size per cluster: {}".format(label, self.DP_mix.assigns[len(self.DP_mix.data)-1], data_per_class))
        return time_record

    def predict(self, s, a):
        '''
        # todo, reuse the history component
        if (len(self.DP_mix.index_chain) > 1) and (self.DP_mix.comps[-1].n < 50):
            self.index = self.DP_mix.index_chain[-2]
        else:
            self.index = self.DP_mix.assigns[len(self.DP_mix.data)-1]
        '''

        # prepare data, the shape is different from fit phase
        x_shared = [s[:, 0:2], a]
        x_independent = s[:, 2:4]

        # shared states prediction (need to process the data again)
        ds_shared = self.sharedGP.predict(*x_shared)

        # independent states prediction
        self.index = self.DP_mix.assigns[len(self.DP_mix.data)-1]
        sample_func = self.DP_mix.comps[self.index].predict(x_independent)
        ds = []
        for i in range(self.DP_mix.args.state_dim):
            ds.append(sample_func[i].loc.cpu().numpy())
        ds = np.array(ds).T

        ds = np.concatenate((ds_shared, ds), axis=1)
        return ds
