

import numpy as np
from loguru import logger
from scipy.special import loggamma
from gpytorch.distributions.multivariate_normal import kl_mvn_mvn as KL_divergence
import copy

class DPMixture:
    def __init__(self, model, args):
        # each component is a GPmixture
        self.component_model = model
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
        self.assigns = {}
        self.comps = []
        self.rho_sum = []
        self.rho = []
        self.rho_transition = [[]]
        self.index_chain = [0]

        self.new_comp = None

    @staticmethod
    def Discrete(a):
        r = np.random.uniform()*np.sum(a)
        cum_sum = [np.sum(a[:i+1]) for i in range(len(a))]
        return np.sum([r > e for e in cum_sum])

    def add_point(self, x):
        self.data.append(x)

    def _sample_z(self, i):
        x = self.data[i]
        if i in self.assigns:
            k = self.assigns[i]
            # delete the component when this component has only one data point
            if self.comps[k].del_point(x, i) == 0:
                del self.comps[k]
                for j, v in self.assigns.items():
                    self.assigns[j] -= int(v > k)

        # Compute parameters for discrete distribution
        n_comps = len(self.comps)
        log_pp = [np.log(self.comps[k].n) + self.comps[k].log_posterior_pdf(x) for k in range(n_comps)]

        # create a new component
        new_comp = self.component_model(test_data=x, args=self.args)
        log_pp.append(np.log(self.alpha) + new_comp.log_posterior_pdf(x))

        j = self.Discrete(np.exp(log_pp))
        self.assigns[i] = j

        if j == n_comps:
            new_comp.n = 1
            new_comp.index_list.append(i)
            self.comps.append(new_comp)
        elif j < n_comps:
            self.comps[j].add_point(x, i)
        else:
            logger.error('Index exceeds the length of components!')

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

    def gibbs_sample(self, n_iter=1):
        # only use the last 5 samples for gibbs sampling
        max_sample_len = 5
        data_len = len(self.data)
        end_ = data_len-1
        if data_len < max_sample_len:
            start_ = -1
        else:
            start_ = data_len - max_sample_len - 1

        # use all data
        #start_ = -1
        #end_ = len(self.data)-1

        # do gibbs sampling
        for step in range(n_iter):
            for i in range(end_, start_, -1):
                self._sample_z(i)

        # # update alpha based on posterior
        if self.ada_alpha:
            self.alpha = self.update_alpha_posterior_dist()
        return self.alpha

    def sequential_vi(self):
        # get the newest data point x_{n+1}
        i = len(self.data)-1
        x = self.data[i]

        # calculate \rho_{n+1, 1:k} - [k, 1]
        n_comps = len(self.comps)
        rho_old = [self.rho_sum[k] * np.exp(self.comps[k].log_posterior_pdf(x)) for k in range(n_comps)]

        # create a new component and use the new data point as training data
        # use the cached component if last data point does not belong to a new component
        if self.new_comp is None:
            self.new_comp = self.component_model(test_data=x, args=self.args)
        else:
            self.new_comp.reset_parameters()

        # calculate \rho_{n+1, k+1} - [1, 1]
        rho_new = [self.alpha * np.exp(self.new_comp.log_posterior_pdf(x))]

        # normalize \rho
        rho = rho_old+rho_new
        #print('rho: ', rho)
        rho = rho/np.sum(rho)

        # get the max probability index
        k = np.argmax(rho, axis=0)
        self.assigns[i] = k
        if k != self.index_chain[-1]:
            self.index_chain.append(k)

        if k == n_comps:
            # \rho_{1:k+1} = [\rho_{1:k}, \rho_{k+1}]
            self.rho_sum.extend([0])
            # \rho_{n+1} = \sum_{i=1}^n \rho_{i}
            self.rho_sum = [a + b for a, b in zip(self.rho_sum, rho)]
            # self.new_comp.n = 1
            # self.new_comp.index_list.append(i)
            self.new_comp.add_point(x, i)
            self.new_comp.train_model()
            self.comps.append(self.new_comp)
            self.new_comp = None
        elif k < n_comps:
            rho_old = rho_old/np.sum(rho_old)
            # \rho_{n+1} = \sum_{i=1}^n \rho_{i}
            self.rho_sum = [a + b for a, b in zip(self.rho_sum, rho_old)]
            self.comps[k].add_point(x, i)
            self.comps[k].train_model()
        else:
            logger.error('Index exceeds the length of components!')

        # # merge existing components
        # # NOTE that self.rho_sum should also be changed when two components are merged
        # if self.merge and n_comps > 1:
        #     self.comps, self.rho_sum = self._merge(self.comps, self.rho_sum)

        # print('assignments')
        # print(self.assigns)

        # normalize rho_sum
        #normalizer = np.sum(self.rho_sum)
        #self.rho_sum = [a/normalizer for a in self.rho_sum]

        # update alpha
        if self.ada_alpha:
            self.alpha = self.update_alpha_posterior_dist()
        return self.alpha

    def sequential_vi_w_transition(self, comp_trainable=None):
        if comp_trainable is None:
            comp_trainable = [] # construct a 0-1 list
            for i in range(len(self.comps)):
                comp_trainable.append(1)

        # get the newest data point x_{n+1}
        i = len(self.data) - 1
        x = self.data[i]

        # calculate \rho_{n+1, 1:k} - [k, 1]
        n_comps = len(self.comps)
        if i == 0:
            rho_old = []
            #rho_old_likelihood = []
        else:
            rho_old = [self.rho_transition[self.assigns[i - 1]][k] * np.exp(self.comps[k].log_posterior_pdf(x)) for k in range(n_comps)]
            #rho_old_likelihood = [np.exp(self.comps[k].log_posterior_pdf(x)) for k in range(n_comps)]

        # the RBF kernel will not be positive definite, then the variance will be negative
        # ignore the point that is similar to existing points
        '''
        if np.isnan(rho_old).any():
            self.assigns[i] = 0
            #self.data.pop() # delete this point
            return self.alpha
        '''

        # create a new component and use the new data point as training data
        # use the cached component if last data point does not belong to a new component
        if self.new_comp is None:
            self.new_comp = self.component_model(test_data=x, args=self.args)
        else:
            self.new_comp.reset_parameters()

        # calculate \rho_{n+1, k+1} - [1, 1]
        # [NOTE] during the burn-in stage, we dont want to create a new component
        # this could be too defensive but useful
        if len(self.comps) > 0 and self.comps[-1].n < self.merge_burnin:
            rho_new = [0.0]
        else:
            rho_new = [self.alpha * np.exp(self.new_comp.log_posterior_pdf(x))]

        # normalize rho
        rho = rho_old + rho_new
        print('raw rho', *rho)
        rho = rho / np.sum(rho)

        #print('------rho--------')
        #rho_ll = rho_old_likelihood + [np.exp(self.new_comp.log_posterior_pdf(x))]
        #print(*self.rho_transition, sep='\n')
        #print(*rho_ll)

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

        # # merge existing components
        # # NOTE that self.rho_sum should also be changed when two components are merged
        # if self.merge and n_comps > 1:
        #     self._merge_w_transition(self.comps, self.rho_transition)

        # update alpha
        if self.ada_alpha:
            self.alpha = self.update_alpha_posterior_dist()
        return self.alpha

    def _inner_merge(self, p_dist, q_dist):
        '''
        """
            Calculate the KL divergence of all outputs at the sam time.
            This method may change the relative value of KL divergence, thus new threshold should be explored.
        """

        test_x = q_dist.get_point()
        n_data = len(test_x)

        # calculate the posterior of the new component
        # TODO: this part could be replaced by directly using the training output. 
        # Otherwise, this will cause a RuntimeWarning: A not p.d., added jitter of 1e-06 to the diagonal.
        new_posterior = q_dist.predict_distribution(test_x)

        # collect the posterior distribution of all previous components
        kld = []
        for c_i in self.comps:
            old_posterior = c_i.predict_distribution(test_x)
            kl_average = np.sum([KL_divergence(p, q).item() for p, q in zip(old_posterior, new_posterior)])/n_data
            kld.append(kl_average)
        
        print('kld', kld)
        min_index = np.argmin(kld)
        return kld, min_index
        '''
        
        test_x = q_dist.get_point()
        n_data = len(test_x)
        kld = []

        # calculate the posterior of the new component
        new_posterior = []
        for x in test_x:
            new_posterior.append(q_dist.predict_distribution(x[None]))

        # collect the posterior distribution of all previous components
        for c_i in self.comps:
            kl_average = 0
            for (x_i, x) in enumerate(test_x):
                old_posterior = c_i.predict_distribution(x[None])
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
            # if n_i != len(comps)-1 and c_i.n < self.merge_burnin:
            if n_i != self.index_chain[-1] and c_i.n < self.merge_burnin:
                print('n_i:', n_i, 'self.index_chain[-1]:', self.index_chain[-1])
                logger.warning('reach force merge')
                rho_sum_ii = rho_sum[n_i]
                comps.remove(c_i)
                rho_sum.remove(rho_sum_ii)
                kld, _ = self._inner_merge(comps, c_i)

                # only consider the likelihood of adjacent index
                pos = self.index_chain.index(n_i)
                adjacent = [self.index_chain[pos - 1], self.index_chain[pos+1]]
                if adjacent[1] > n_i:
                    adjacent[1] -= 1

                kld = [kld[adjacent[0]], kld[adjacent[1]]]
                min_index = adjacent[np.argmin(kld)]
                print('n_i: ', n_i, ' adjacent: ', adjacent,
                      ' KLD: ', kld, ' min_kld: ', np.min(kld),
                      ' min_index: ', min_index)

                # add all points to the old component
                comps[min_index].merge_point(c_i.data, c_i.index_list)
                # update all the assignments
                for i in c_i.index_list:
                    self.assigns[i] = min_index

                rho_sum[min_index] += rho_sum_ii
                # change the index chain
                self.index_chain.pop(pos)
                if self.index_chain[-1] != min_index:
                    self.index_chain.append(min_index)

                logger.warning('Force merging a small component to component {}', min_index)

            # Condition 2: check the new components that reaches the self.merge_burnin condition
            # if n_i == len(comps)-1 and c_i.n == self.merge_burnin:
            if n_i == self.index_chain[-1] and c_i.n == self.merge_burnin:
                rho_sum_i = rho_sum[n_i]
                comps.remove(c_i)
                kld, min_index = self._inner_merge(comps, c_i)
                min_kld = kld[min_index]

                print('---------------------------------------')
                print('kld ', kld)
                print('min_index ', min_index)
                print('---------------------------------------')

                if min_kld < self.merge_threshold:
                    # add all points to the old component
                    comps[min_index].merge_point(c_i.data, c_i.index_list)
                    rho_sum.remove(rho_sum_i)
                    rho_sum[min_index] += rho_sum_i

                    # update all the assignments
                    for i in c_i.index_list:
                        self.assigns[i] = min_index


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
            # Intuitively, when a small group of component appears between two large groups, this is perhaps mis-classified
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
                if adjacent[1] > n_i:
                    adjacent[1] -= 1
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
