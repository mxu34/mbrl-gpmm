
import random
import os

import numpy as np
np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)  # disable Scientific Notation
from loguru import logger
from tqdm import tqdm

import torch
import gpytorch
from gpytorch.constraints import GreaterThan, Positive, LessThan


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def CUDA(var):
    # the default dtype is float32, but it will cause some numerical problems.
    var = var.double()
    #return var
    return var.cuda() if torch.cuda.is_available() else var


class SampleGPR(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, input_dim, params):
        super(SampleGPR, self).__init__(train_x, train_y, likelihood)
        self.params = params
        self.input_dim = input_dim
        self.lengthscale_prior = None #gpytorch.priors.GammaPrior(3.0, 6.0)
        self.outputscale_prior = None #gpytorch.priors.GammaPrior(2.0, 0.15)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=input_dim, 
                lengthscale_prior=self.lengthscale_prior
            ),
            outputscale_prior=self.outputscale_prior
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def inducing_ELBO(self, Xs):
        X = self.train_inputs[0]
        Y = self.train_targets[:, None]
        K_MM = self.covar_module(Xs, Xs).evaluate()
        K_NM = self.covar_module(X, Xs).evaluate()
        Sigma = self.likelihood.noise_covar.noise

        Inverve_K_MM = torch.inverse(K_MM)  
        K_MN = K_NM.transpose(1, 0) 
        Y_T = Y.transpose(1, 0) 

        # We define A = K_NM^T * invK_MM * K_MN
        A = torch.mm(K_NM, torch.mm(Inverve_K_MM, K_MN))

        # B is an array containing only diagonal elements of K_NN - A.
        # Note we assume diagonal elements of A are always equal to 1.
        B = 1 - A.diagonal()

        C = A + CUDA(torch.eye(len(X)))*Sigma**2
        Sign, LogDetC = torch.slogdet(C)
        Inverse_C = torch.inverse(C)

        # Calculate the lower bound
        EBLO = -0.5*Sign*LogDetC - 0.5*torch.mm(Y_T, torch.mm(Inverse_C, Y)) - 1/(2*Sigma**2)*torch.sum(B)
        
        return EBLO

    def reset_parameters(self):
        self.initialize(**{
            'likelihood.noise': self.params[0],
            'mean_module.constant': self.params[2],
            'covar_module.outputscale': self.params[3], 
            'covar_module.base_kernel.lengthscale': self.params[4].repeat(1, self.input_dim),
        })


class GPComponent(object):
    def __init__(self, test_data, args):
        # data buffer, only store training data, test_data will only be stored in GP model before the model is trained
        self.n = 0
        self.data = None
        self.index_list = []

        self.previous_loss = CUDA(torch.tensor(np.inf))
        self.trigger_training = CUDA(torch.tensor(1e-4))

        self.lr = args.lr
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.input_dim = self.state_dim + self.action_dim
        self.gp_iter = args.gp_iter

        self.normalize_trigger = 1
        self.eps = CUDA(torch.tensor(1e-10))
        self.mu_x = CUDA(torch.zeros((self.input_dim)))
        self.sigma_x = CUDA(torch.ones((self.input_dim)))
        #self.sigma_x[9:12] = CUDA(torch.tensor(10.0))
        #self.sigma_x[12:18] = CUDA(torch.tensor(10.0))
        self.mu_y = CUDA(torch.zeros((self.state_dim)))
        self.sigma_y = CUDA(torch.ones((self.state_dim)))
        #self.sigma_y[9:12] = CUDA(torch.tensor(10.0))
        #self.sigma_y[12:18] = CUDA(torch.tensor(10.0))

        # parameters for inducing GP
        self.max_inducing_point = args.max_inducing_point
        self.trigger_induce = args.trigger_induce
        self.sample_number = args.sample_number

        # prior of the kernel parameters
        # [NOTE] these prior parameters should be similar to the estimated parameters of real data
        # if lengthscale is too large, it will be too difficult to create new components
        # if lengthscale is too small, it will be too esay to create new components
        # if noise_covar is too large, the prediction will be inaccurate
        # if noise_covar is too small, the covariance will be very small, causing some numerical problems
        self.param = CUDA(torch.tensor(args.param))

        # initialize model and likelihood
        model_list = []
        likelihood_list = []
        for m_i in range(self.state_dim):
            likelihood = CUDA(gpytorch.likelihoods.GaussianLikelihood(noise_constraint=GreaterThan(self.param[1])))
            model = CUDA(SampleGPR(None, None, likelihood, self.input_dim, self.param))
            model.reset_parameters()
            likelihood_list.append(model.likelihood)
            model_list.append(model)

        # initialize model list
        self.model = gpytorch.models.IndependentModelList(*model_list)
        self.likelihood = gpytorch.likelihoods.LikelihoodList(*likelihood_list)

        # initialize optimizer
        self.optimizer = torch.optim.Adam([{'params': self.model.parameters()}], lr=self.lr)
        self.mll = gpytorch.mlls.SumMarginalLogLikelihood(self.likelihood, self.model)

        # change the flag
        self.model.eval()
        self.likelihood.eval()

    def reset_parameters(self):
        for m_i in range(self.state_dim):
            self.model.models[m_i].reset_parameters()

    def train_model(self):
        # prepare data
        train_x = self.data[:, :self.input_dim]
        train_y = self.data[:, self.input_dim:]

        if self.n > self.trigger_induce:
            logger.warning('Running MC sample inducing method')
            # prepare data
            train_x = self.data[:, :self.input_dim]
            train_y = self.data[:, self.input_dim:]
            with torch.no_grad():
                # sample inducing point
                for s_i in range(self.sample_number):
                    elbo_sum = 0
                    indices = random.sample(range(self.n), self.max_inducing_point)
                    X_candidate = train_x[indices, :]
                    for m_i in range(self.state_dim):
                        elbo = self.model.models[m_i].inducing_ELBO(X_candidate)
                        elbo_sum += elbo.item()
                    #print(s_i, elbo_sum)
                    if s_i == 0:            
                        Xs = X_candidate  
                        Ys = train_y[indices]  
                        LB_best = elbo_sum    
                    else:
                        if elbo_sum > LB_best: # Maximize the lower bound
                            Xs = X_candidate 
                            Ys = train_y[indices]  
                            LB_best = elbo_sum   

            # replace data with inducing points
            self.data = torch.cat((Xs, Ys), dim=1)
            self.n = len(self.data)

        # normalize data
        # for online setting, the sigma is unstable, therefore we ignore it
        if self.n > self.normalize_trigger:
            self.mu_x = torch.mean(train_x, dim=0)
            #self.sigma_x = torch.std(train_x, dim=0)
            self.mu_y = torch.mean(train_y, dim=0)
            #self.sigma_y = torch.std(train_y, dim=0)
            train_x = (train_x-self.mu_x) / (self.sigma_x + self.eps)
            train_y = (train_y-self.mu_y) / (self.sigma_y + self.eps)

        # reset training data
        # DO NOT reset parameters, use the parameters of last time
        for m_i in range(self.state_dim):
            self.model.models[m_i].set_train_data(train_x, train_y[:, m_i], strict=False)
            #self.model.models[m_i].prediction_strategy = None

        with gpytorch.settings.max_cholesky_size(3000):
            # training stage
            self.model.train() # set prediction_strategy = None inside
            self.likelihood.train()
            for i in range(self.gp_iter):
                self.optimizer.zero_grad()
                output_func = self.model(*self.model.train_inputs)
                loss = -self.mll(output_func, self.model.train_targets)

                # [NOTE] This early-stop trick will dramatically decrease the running time
                # abort training when the loss is similar to previous one after training
                if torch.abs(self.previous_loss-loss) < self.trigger_training and self.previous_loss-loss > 0:
                    break
                else:
                    self.previous_loss = loss
                
                loss.backward()
                self.optimizer.step()

            # change the flag
            self.model.eval()
            self.likelihood.eval()

    def get_data_number(self):
        return len(self.data)

    def get_point(self):
        return self.data.cpu().numpy()

    def merge_point(self, new_tensor_data, new_list):
        # the data to be merged is expected to be a torch.Tensor
        self.data = torch.cat((self.data, new_tensor_data), dim=0)
        self.index_list += new_list
        self.n += len(new_list)

    def add_point(self, x, i):
        # add some noise to the data
        #x_noise = np.random.randn(1, 4) * np.sqrt(0.0001)
        #x[:, 5:9] = x[:, 5:9] + x_noise

        if self.data is None:
            self.data = CUDA(torch.Tensor(x))
        else:
            self.data = torch.cat((self.data, CUDA(torch.tensor(x).float())), dim=0)
        self.n += 1
        self.index_list.append(i)

    def del_point(self, x, i):
        # for sequential vi method, this function is deprecated
        # TODO: check this may be really slow, modify with index later
        remove_index = self.index_list.index(i)
        self.data = torch.cat([self.data[:remove_index,:], self.data[remove_index+1:,:]], dim=0)
        self.index_list.remove(i)
        self.n -= 1
        return self.n

    def log_posterior_pdf(self, x, train=True):
        # prepare data
        x = CUDA(torch.Tensor(x))
        test_x = x[:, :self.input_dim]
        test_y = x[:, self.input_dim:]
        if self.n > self.normalize_trigger:
            print(test_y)
            # normalize the data
            test_x = (test_x-self.mu_x) / (self.sigma_x + self.eps)
            # normalize the groundtruth
            test_y = (test_y-self.mu_y) / (self.sigma_y + self.eps)

            print(test_y)
            print('=====')

        # get the log likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_all = test_x.repeat(self.state_dim, 1, 1)
            sample_func = self.model(*test_all)
            #sample_func_lik = self.likelihood(*sample_func)

            log_ppf = 0
            for f_i in range(len(sample_func)):
                # [BUG of GPytorch] when numerical problem happens, the covariance_matrix will be non-positive-definite
                # then, the log_porb will return nan. We reset the covariance_matrix to a pre-defined value (constraint of noise_covar)
                if sample_func[f_i].covariance_matrix <= CUDA(torch.tensor([[0.0]])):
                    sample_func[f_i] = gpytorch.distributions.MultivariateNormal(sample_func[f_i].loc, CUDA(self.param[0][None, None]))
                incre = sample_func[f_i].log_prob(test_y[:, f_i]).item()
                log_ppf += incre

        # since we added all likelihood together
        return log_ppf/len(sample_func)

    # this is used for KL divergence calculation
    # this part does not need to normalize the output
    def predict_distribution(self, x):
        # prepare data
        test_x = CUDA(torch.Tensor(x[:, :self.input_dim]))
        if self.n > self.normalize_trigger:
            test_x = (test_x-self.mu_x) / (self.sigma_x + self.eps)

        # get the log likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_all = test_x.repeat(self.state_dim, 1, 1)
            sample_func = self.model(*test_all)
            #sample_func_lik = self.likelihood(*sample_func)
        return sample_func

    def predict(self, x):
        # predict the distribution firstly, and we only need the .loc
        sample_func = self.predict_distribution(x)

        ds = []
        for i in range(self.state_dim):
            test_y = sample_func[i].loc
            # normalize back
            if self.n > self.normalize_trigger:
                test_y = test_y * (self.sigma_y[i] + self.eps) + self.mu_y[i]
            ds.append(test_y.detach().cpu().numpy())
        ds = np.array(ds).T
        return ds
