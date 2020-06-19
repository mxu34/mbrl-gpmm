

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


class Config_Parser(object):
    def __init__(self, sparse_gp_config):
        self.state_dim = sparse_gp_config['state_dim']
        self.action_dim = sparse_gp_config['action_dim']
        self.lr = sparse_gp_config['lr']
        self.gp_iter = sparse_gp_config['gp_iter']
        self.param = sparse_gp_config['param']
        self.max_inducing_point = sparse_gp_config['max_inducing_point']


class ExactGPR(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, input_dim, params):
        super(ExactGPR, self).__init__(train_x, train_y, likelihood)
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

    def reset_parameters(self):
        self.initialize(**{
            'likelihood.noise': self.params[0],
            'mean_module.constant': self.params[2],
            'covar_module.outputscale': self.params[3], 
            'covar_module.base_kernel.lengthscale': self.params[4].repeat(1, self.input_dim),
        })
        

class SparseGPR(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, input_dim, params):
        super(SparseGPR, self).__init__(train_x, train_y, likelihood)
        self.params = params
        self.input_dim = input_dim
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=input_dim))
        
        # use some training data to initialize the inducing_module
        if train_x is None:
            inducing_points = CUDA(torch.zeros((1, input_dim)))
        else:
            inducing_points = train_x
        self.inducing_module = gpytorch.kernels.InducingPointKernel(self.covar_module, inducing_points=inducing_points, likelihood=likelihood)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.inducing_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def reset_parameters(self):
        self.initialize(**{
            'likelihood.noise': self.params[0],
            'mean_module.constant': self.params[2],
            'covar_module.outputscale': self.params[3], 
            'covar_module.base_kernel.lengthscale': self.params[4].repeat(1, self.input_dim),
        })
        

class SingleSparseGP(object):
    name = 'SingleSparseGP'
    
    def __init__(self, sparse_gp_config):
        args = Config_Parser(sparse_gp_config=sparse_gp_config)

        # data buffer, only store training data, test_data will only be stored in GP model before the model is trained
        self.n = 0
        self.data = None
        self.index_list = []
        self.use_sparse = True
        self.train_interval = 1

        self.previous_loss = CUDA(torch.tensor(np.inf))
        self.trigger_training = CUDA(torch.tensor(1e-4))

        self.lr = args.lr
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.input_dim = self.state_dim + self.action_dim
        self.gp_iter = args.gp_iter

        self.normalize_trigger = 1
        self.eps = CUDA(torch.tensor(1e-20))
        self.mu_x = CUDA(torch.zeros((self.input_dim)))
        self.sigma_x = CUDA(torch.ones((self.input_dim)))
        #self.sigma_x[9:12] = CUDA(torch.tensor(10.0))
        self.sigma_x[12:18] = CUDA(torch.tensor(10.0))
        self.mu_y = CUDA(torch.zeros((self.state_dim)))
        self.sigma_y = CUDA(torch.ones((self.state_dim)))
        #self.sigma_y[9:12] = CUDA(torch.tensor(10.0))
        self.sigma_y[12:18] = CUDA(torch.tensor(10.0))
        
        # parameters for inducing GP
        self.max_inducing_point = args.max_inducing_point

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
            model = CUDA(ExactGPR(None, None, likelihood, self.input_dim, self.param))
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

    def data_process(self, data):
        if self.action_dim == 0:
            data_point = data
            label = None
        else:
            s = data[1]
            a = data[2]
            s_n = data[3]
            data_point = np.concatenate((s, a, s_n), axis=0)[None]
            label = data[0]
        return data_point, label
        
    def add_point(self, x):
        data_point, label = self.data_process(x)
        # for the first data, construct the gp models
        if self.data is None:
            self.n = 1
            self.data = CUDA(torch.Tensor(data_point))
        else:
            self.data = torch.cat((self.data, CUDA(torch.from_numpy(data_point))), dim=0)
            self.n += 1
        return label

    def fit(self, data):
        print('data size', self.n)
        # add data point to the data buffer
        label = self.add_point(data)
        # prepare data
        train_x = self.data[:, :self.input_dim]
        train_y = self.data[:, self.input_dim:]

        # normalize data
        if self.n > self.normalize_trigger:
            self.mu_x = torch.mean(train_x, dim=0)
            self.sigma_x = torch.std(train_x, dim=0)
            self.mu_y = torch.mean(train_y, dim=0)
            self.sigma_y = torch.std(train_y, dim=0)
            train_x = (train_x-self.mu_x) / (self.sigma_x + self.eps)
            train_y = (train_y-self.mu_y) / (self.sigma_y + self.eps)

        if self.n >= self.max_inducing_point and self.use_sparse:
            self.use_sparse = False
            logger.warning('Change to Sparse GP model')
            
            # initialize model and likelihood
            model_list = []
            likelihood_list = []
            for m_i in range(self.state_dim):
                likelihood = CUDA(gpytorch.likelihoods.GaussianLikelihood(noise_constraint=GreaterThan(self.param[1])))
                model = CUDA(SparseGPR(train_x, train_y[:, m_i], likelihood, self.input_dim, self.param))
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

            self.trigger_training = CUDA(torch.tensor(1e-3))
            # we wont train the model every time
            self.train_interval = 10

        # reset training data
        # DO NOT reset parameters, use the parameters of last time
        for m_i in range(self.state_dim):
            self.model.models[m_i].set_train_data(train_x, train_y[:, m_i], strict=False)

        if self.n % self.train_interval == 0:
            # training stage
            self.model.train() 
            self.likelihood.train()
            for i in range(self.gp_iter):
                self.optimizer.zero_grad()
                output_func = self.model(*self.model.train_inputs)
                loss = -self.mll(output_func, self.model.train_targets)

                # [NOTE] This early-stop trick will dramatically decrease the running time
                # abort training when the loss is similar to previous one after training
                if torch.abs(self.previous_loss-loss) < self.trigger_training:
                    break
                else:
                    self.previous_loss = loss

                loss.backward()
                self.optimizer.step()

            # change the flag
            self.model.eval()
            self.likelihood.eval()
        
        return None

    def predict(self, s, a):
        # when the action dimension is 0, we dont input action
        if self.action_dim == 0:
            test_x = CUDA(torch.Tensor(s[:, :self.state_dim]))
        else:
            x = np.concatenate((s, a), axis=1)
            # prepare data
            test_x = CUDA(torch.Tensor(x[:, :self.input_dim]))
            if self.n > self.normalize_trigger:
                test_x = (test_x-self.mu_x) / (self.sigma_x + self.eps)

        # get the log likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_all = test_x.repeat(self.state_dim, 1, 1)
            sample_func = self.model(*test_all)

        ds = []
        for i in range(self.state_dim):
            test_y = sample_func[i].loc
            if self.n > self.normalize_trigger:
                test_y = test_y * (self.sigma_y[i] + self.eps) + self.mu_y[i]
            ds.append(test_y.detach().cpu().numpy())
        ds = np.array(ds).T
        return ds

    def test(self, s, a, x_g):
        x = np.concatenate((s, a), axis=1)
        test_x = CUDA(torch.Tensor(x[:, :self.input_dim]))
        if self.n > self.normalize_trigger:
            test_x = (test_x-self.mu_x) / (self.sigma_x + self.eps)

        # get the log likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_all = test_x.repeat(self.state_dim, 1, 1)
            sample_func = self.model(*test_all)

            ds = []
            ds_unnormal = []
            for i in range(self.state_dim):
                test_y = sample_func[i].loc
                test_y_unnormal = test_y * (self.sigma_y[i] + self.eps) + self.mu_y[i]
                ds.append(test_y.detach().cpu().numpy())
                ds_unnormal.append(test_y_unnormal.detach().cpu().numpy())
        
        ds = np.array(ds).T
        ds_unnormal = np.array(ds_unnormal).T

        x_g = CUDA(torch.Tensor(x_g))
        x_g = (x_g-self.mu_y) / (self.sigma_y + self.eps)
        x_g = x_g.cpu().numpy()
        mse_error = np.sum((ds[0]-x_g)**2)

        return ds_unnormal, mse_error

    def test_on_train_data(self):
        # get the log likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            sample_func = self.model(*self.model.train_inputs)

            ds = []
            for i in range(self.state_dim):
                test_y = sample_func[i].loc
                x_g = self.model.train_targets[i]
                mse_error = (test_y-x_g)**2
                ds.append(mse_error.cpu().numpy())

        ds = np.sum(ds, axis=0)
        return np.mean(ds), np.std(ds)
