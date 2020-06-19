
import numpy as np
from loguru import logger
#import torch
from scipy.special import loggamma
from gpytorch.distributions.multivariate_normal import kl_mvn_mvn as KL_divergence


class DPMixture:
    def __init__(self, model, args, nn_config):
        # each component is a NN
        self.component_model = model
        self.nn_config = nn_config

        self.alpha = args.alpha
        self.ada_alpha = args.ada_alpha

        self.args = args
        self.merge = args.merge
        self.merge_burnin = args.merge_burnin
        self.merge_threshold = args.merge_threshold
        self.window_prob = args.window_prob
        self.self_prob = args.self_prob

        self.alpha_max = 5.01
        self.alpha_min = 0.01

        self.data = []
        self.assigns = 0
        self.comps = []
        self.rho_sum = []
        self.rho = []
        self.rho_transition = [[]]
        self.index_chain = [0]

        self.new_comp = None
        #self.n_comps = 0 # components number
        self.meta_model_path = self.nn_config['model_config']['save_model_path']

    def init_new_comp(self):
        new_comp = self.component_model(self.nn_config)
        # load meta training weight
        new_comp.load_model(self.meta_model_path)
        return new_comp

    @staticmethod
    def Discrete(a):
        r = np.random.uniform()*np.sum(a)
        cum_sum = [np.sum(a[:i+1]) for i in range(len(a))]
        return np.sum([r > e for e in cum_sum])

    def add_point(self, x):
        self.data.append(x)

    def _inner_update_alpha_posterior_dist(self, K, N, alpha_min, alpha_max, n_sample):
        alpha = np.linspace(alpha_min, alpha_max, n_sample)
        logp = (K - 1.5) * np.log(alpha) - 1.0 / (2.0 * alpha) + loggamma(alpha) - loggamma(N + alpha)
        alpha_select = alpha[np.argmax(logp)]
        return alpha_select

    def update_alpha_posterior_dist(self, n_sample=100):
        N = len(self.data)
        K = len(self.comps)
        alpha_select = self._inner_update_alpha_posterior_dist(K, N, self.alpha_min, self.alpha_max, n_sample)
        while np.abs(alpha_select-self.alpha_max) < 1e-10:
            self.alpha_max += 5
            self.alpha_min += 4.9
            alpha_select = self._inner_update_alpha_posterior_dist(K, N, self.alpha_min, self.alpha_max, n_sample)
        return alpha_select

    def svi(self, dataset, nn_meta_model):

        # calculate \rho_{n+1, 1:k} - [k, 1]
        n_comps = len(self.comps)
        if not self.args.DPprior:
            rho_old = [self.comps[k].likelihood(dataset) for k in range(n_comps)]

            # create a new component and use the new data point as training data
            # use the cached component if last data point does not belong to a new component
            new_comp_prior = nn_meta_model

            # calculate \rho_{n+1, k+1} - [1, 1]
            rho_new = [self.alpha * new_comp_prior.likelihood(dataset)]
        else:
            rho_old = [np.log(len(self.comps[k].dataset)) + self.comps[k].likelihood(dataset)
                       for k in range(n_comps)]

            # create a new component and use the new data point as training data
            # use the cached component if last data point does not belong to a new component
            new_comp_prior = nn_meta_model

            # calculate \rho_{n+1, k+1} - [1, 1]
            rho_new = [np.log(self.alpha) + new_comp_prior.likelihood(dataset)]

        print("rho old :", rho_old, "rho_new: ", rho_new)

        # normalize \rho
        rho = rho_old+rho_new
        rho = rho/np.sum(rho)

        # get the max probability index
        cluster_idx = np.argmax(rho, axis=0)
        self.assigns = cluster_idx

        if cluster_idx == n_comps:
            # \rho_{1:k+1} = [\rho_{1:k}, \rho_{k+1}]
            self.rho_sum.extend([0])
            # \rho_{n+1} = \sum_{i=1}^n \rho_{i}
            self.rho_sum = [a + b for a, b in zip(self.rho_sum, rho)]
            new_comp = self.init_new_comp()
            for data in dataset:
                new_comp.add_data_point(data)
            new_comp.fit(logger=True)
            self.comps.append(new_comp)

        elif cluster_idx < n_comps:
            rho_old = rho_old/np.sum(rho_old)
            # \rho_{n+1} = \sum_{i=1}^n \rho_{i}
            self.rho_sum = [a + b for a, b in zip(self.rho_sum, rho_old)]
            for data in dataset:
                self.comps[cluster_idx].add_data_point(data)
            self.comps[cluster_idx].load_model(self.meta_model_path)
            self.comps[cluster_idx].fit(logger=True)
        else:
            logger.error('Index exceeds the length of components!')

        # merge existing components
        # NOTE that self.rho_sum should also be changed when two components are merged
        #if self.merge and n_comps > 1:
        #    self.comps, self.rho_sum = self._merge(self.comps, self.rho_sum)

        # normalize rho_sum
        #normalizer = np.sum(self.rho_sum)
        #self.rho_sum = [a/normalizer for a in self.rho_sum]

        # update alpha
        #if self.ada_alpha:
        #    self.alpha = self.update_alpha_posterior_dist()
        return cluster_idx

    def svit(self, dataset, nn_meta_model):
        # dataset format: list of [task_idx, state, action, next_state-state]
        # nn_meta_model serves as a prior

        if self.n_comps==0:  # init a new comp
            new_comp = self.init_new_comp()
            new_comp.fit(dataset)
            self.comps.append(new_comp)
            self.n_comps = 1
            rho_old = []
            rho_new = [self.alpha * np.exp(self.new_comp.log_likelihood(x))]
            # normalize rho
            rho = rho_old + rho_new
            rho = rho / np.sum(rho)
            return 0

        # get the newest data point x_{n+1}
        i = len(self.data) - 1
        x = self.data[i]

        # calculate \rho_{n+1, 1:k} - [k, 1]
        n_comps = len(self.comps)
        if i == 0: ###################################
            rho_old = []
            #rho_old_likelihood = []
        else:
            rho_old = [self.rho_transition[self.assigns[i - 1]][k] * np.exp(self.comps[k].log_likelihood(x)) for k in range(n_comps)]

        # create a new component and use the new data point as training data
        # use the cached component if last data point does not belong to a new component

        self.new_comp = nn_meta_model

        # calculate \rho_{n+1, k+1} - [1, 1]
        # [NOTE] during the burn-in stage, we dont want to create a new component
        # this could be too defensive but useful
        if len(self.comps) > 0 and self.comps[-1].n < self.merge_burnin:
            rho_new = [0.0]
        else: #####################################
            rho_new = [self.alpha * np.exp(self.new_comp.log_likelihood(x))]

        # normalize rho
        rho = rho_old + rho_new
        rho = rho / np.sum(rho)

        # TODO: check the nan problem, may add rho = [nan,nan,...]
        if np.isnan(rho).any():
            print('Detect Nan before normalization')
            print(rho)
            print('directly normalize rho')
            rho = np.isnan(rho)*1
            print(rho)

        # get the max probability index
        k = np.argmax(rho, axis=0)
        self.assigns[i] = k
        if k != self.index_chain[-1]:
            self.index_chain.append(k)

        # assign the new point to a component
        if k == n_comps:
            self.new_comp.add_point(x, i)
            self.new_comp.train_model()
            self.comps.append(self.new_comp)
            self.new_comp = None

            # \rho_{n+1} = \sum_{i=1}^n \rho_{i}
            if i == 0:
                self.rho_transition[0].extend([0])
                self.rho_transition[0] = [a + b for a, b in zip(self.rho_transition[0], rho)]
            else:
                for j in range(len(self.comps)-1):
                    self.rho_transition[j].extend([0])
                self.rho_transition[self.assigns[i-1]] = [a + b for a, b in zip(self.rho_transition[self.assigns[i-1]], rho)]

                # add an initial line to the transition matrix
                # [NOTE] the new component should have higher enough probability to go back to itself
                # because a new component is easy to be ignored if we set all elements to 0 (checked 0.1,0.01, but not very good)
                # But if the self_prob is too large, the new component later will be hard to create
                init = []
                for j in range(len(self.comps)):
                    init.append(self.window_prob)
                init[k] = self.self_prob
                self.rho_transition.append(init)

        elif k < n_comps:
            rho_old = rho_old / np.sum(rho_old)
            self.rho_transition[self.assigns[i-1]] = [a + b for a, b in zip(self.rho_transition[self.assigns[i-1]], rho_old)]
            self.comps[k].add_point(x, i)
            # only train when the flag is 1
            if comp_trainable[k] == 1:
                self.comps[k].train_model()
        else:
            logger.error('Index exceeds the length of components!')

        # merge existing components
        # NOTE that self.rho_sum should also be changed when two components are merged
        #if self.merge and n_comps > 1:
        #    self._merge_w_transition(self.comps, self.rho_transition)

        # update alpha
        #if self.ada_alpha:
        #    self.alpha = self.update_alpha_posterior_dist()
        return self.alpha

    def _inner_merge(self, p_dist, q_dist):
        '''
        """
            Calculate the KL divergence of all outputs at the sam time.
            This method may change the relative value of KL divergence, thus new threshold should be explored.
        """
        '''
        
        test_x = q_dist.get_point()
        n_data = len(test_x)
        kld = []

        # calculate the posterior of the new component
        new_posterior = []
        for x in test_x:
            new_posterior.append(q_dist.predict(x[None]))

        # collect the posterior distribution of all previous components
        for c_i in self.comps:
            kl_average = 0
            for (x_i, x) in enumerate(test_x):
                old_posterior = c_i.predict(x[None])
                # use KL-divergence to compare the similarity
                # NOTE: the KL(p, q) means the divergence from q to p
                kl_average += np.sum([KL_divergence(p, q).item() for p, q in zip(old_posterior, new_posterior[x_i])])
            kl_average = kl_average/n_data
            kld.append(kl_average)
        min_index = np.argmin(kld)
        return kld, min_index

    def _merge(self, comps, rho_sum):
        for (n_i, c_i) in enumerate(comps):
            # Condition 1: check the components that has less data points than the self.merge_burnin condition
            if n_i != len(comps)-1 and c_i.n < self.merge_burnin:
                rho_sum_ii = rho_sum[n_i]
                comps.remove(c_i)
                rho_sum.remove(rho_sum_ii)
                _, min_index = self._inner_merge(comps, c_i)
                # add all points to the old component
                comps[min_index].merge_point(c_i.data, c_i.index_list)
                rho_sum[min_index] += rho_sum_ii
                logger.warning('Force merging a small component to component {}', min_index)

            # Condition 2: check the new components that reaches the self.merge_burnin condition
            if n_i == len(comps)-1 and c_i.n == self.merge_burnin:
                rho_sum_i = rho_sum[n_i]
                comps.remove(c_i)
                kld, min_index = self._inner_merge(comps, c_i)
                min_kld = kld[min_index]
                if min_kld < self.merge_threshold:
                    # add all points to the old component
                    comps[min_index].merge_point(c_i.data, c_i.index_list)
                    rho_sum.remove(rho_sum_i)
                    rho_sum[min_index] += rho_sum_i
                    logger.warning('Merge the new component to component {}', min_index)
                else:
                    # return to the original comps
                    comps.append(c_i)
                    logger.warning('Finish burnin and create a new component')

        return comps, rho_sum

    def _merge_w_transition(self, comps, rho_transition):
        for (n_i, c_i) in enumerate(comps):
            # Condition 1: check the components that has less data points than the self.merge_burnin condition
            # if n_i != len(comps) - 1 and c_i.n < self.merge_burnin:
            # [BUG] When a new component is still in the brunin stage, if a newer component is created,
            # An IndexError will be triggered
            if n_i != self.index_chain[-1] and c_i.n < self.merge_burnin:
                print('n_i:', n_i, 'self.index_chain[-1]:', self.index_chain[-1])
                logger.warning('reach force merge')
                rho_transition_ii = rho_transition[n_i]
                comps.remove(c_i)
                rho_transition.remove(rho_transition_ii)
                kld, _ = self._inner_merge(comps, c_i) # calculate the KL

                # only consider the likelihood of adjacent index
                pos = self.index_chain.index(n_i)
                print('pos ', pos)
                print('len components ', len(comps))
                print('kld ', kld)
                adjacent = [self.index_chain[pos-1], self.index_chain[pos+1]]
                print('adjacent ', adjacent)
                kld = [kld[adjacent[0]], kld[adjacent[1]]]
                print('kld ', kld)
                min_index = adjacent[np.argmin(kld)]

                # add all points to the old component
                comps[min_index].merge_point(c_i.data, c_i.index_list)
                # update all the assignments
                for i in c_i.index_list:
                    self.assigns[i] = min_index

                # add c_i's weight to the destination cluster
                rho_transition[min_index] = [a + b for a, b in zip(rho_transition[min_index], rho_transition_ii)]
                # pop out the transition to the merged index
                for i in range(len(rho_transition)):
                    rho_transition[i].pop(n_i)

                # add transition to the destination index
                # TODO: note that this only returns the first one
                rho_transition[self.index_chain[pos - 1]][min_index] += c_i.n
                # add transition to the last index in the index chain
                rho_transition[min_index][self.index_chain[pos + 1]] += 1
                # change the index chain
                self.index_chain.pop(pos)

                logger.warning('Force merging a small component to component {}', min_index)

            # Condition 2: check the new components that reaches the self.merge_burnin condition
            if n_i == self.index_chain[-1] and c_i.n == self.merge_burnin:
                rho_transition_i = rho_transition[n_i]
                comps.remove(c_i)
                kld, min_index = self._inner_merge(comps, c_i)
                min_kld = kld[min_index]

                print('---------------------------------------')
                print('kld ', kld)
                print('min_index ', min_index)
                print('---------------------------------------')

                # TODO: check the threshold self.merge_threshold and self.merge_burnin
                # when burnin_num = 15,
                if min_kld < self.merge_threshold:
                    rho_transition.remove(rho_transition_i)
                    # add all points to the old component
                    comps[min_index].merge_point(c_i.data, c_i.index_list)
                    # update all the assignments
                    for i in c_i.index_list:
                        self.assigns[i] = min_index
                    rho_transition[min_index] = [a + b for a, b in zip(rho_transition[min_index], rho_transition_i)]
                    # pop out the transition to the merged index
                    for i in range(len(rho_transition)):
                        rho_transition[i].pop(n_i)

                    # add one more transition to the previous index
                    pos = len(self.index_chain)-1 # the checked component is the last one in the chain
                    rho_transition[self.index_chain[pos - 1]][min_index] += c_i.n
                    self.index_chain.pop()
                    if self.index_chain[-1] != min_index:
                        self.index_chain.append(min_index)
                    logger.warning('Merge the new component to component {}', min_index)
                else:
                    # return to the original comps
                    comps.append(c_i)
                    logger.warning('Finish burnin and create a new component')

        # update parameters and models
        self.comps = comps
        self.rho_transition = rho_transition
