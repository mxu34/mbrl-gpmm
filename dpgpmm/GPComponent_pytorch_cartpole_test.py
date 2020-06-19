import os

import numpy as np

import torch
import gpytorch
from gpytorch.constraints import GreaterThan, Positive, LessThan


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def CUDA(var):
    # the default dtype is float32, but it will cause some numerical problems.
    var = var.double()
    #return var
    return var.cuda() if torch.cuda.is_available() else var


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
                lengthscale_prior=self.lengthscale_prior,
                lengthscale_constraint=LessThan(self.params[4])
            ),
            outputscale_prior=self.outputscale_prior,
            outputscale_constraint=GreaterThan(self.params[5])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def reset_parameters(self):
        self.initialize(**{
            'likelihood.noise': self.params[0],
            'mean_module.constant': self.params[1],
            'covar_module.outputscale': self.params[2],
            'covar_module.base_kernel.lengthscale': self.params[3].repeat(1, self.input_dim),
        })


class GPComponent(object):
    def __init__(self, test_data, args):
        # data buffer, only store training data, test_data will only be stored in GP model before the model is trained
        self.n = 0
        self.data = None
        self.index_list = []
        self.norm = 1.0

        self.previous_loss = CUDA(torch.tensor(np.inf))
        self.trigger_training = CUDA(torch.tensor(1e-3))

        self.lr = args.lr
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.input_dim = self.state_dim + self.action_dim
        self.gp_iter = args.gp_iter

        # prior of the kernel parameters
        # [NOTE] these prior parameters should be similar to the estimated parameters of real data
        # if lengthscale is too large, it will be too difficult to create new components
        # if lengthscale is too small, it will be too esay to create new components
        # if noise_covar is too large, the prediction will be inaccurate
        # if noise_covar is too small, the conjugate gradient will not converge, and the prediction will be improve if too small
        self.param = [
            1e-5,   # noise_covar initilize and constraint
            0.0,    # constant initilize
            0.7,   # outputscale initilize
            1.0,    # [lengthscale initilize]
            100.0,  # lengthscale_constraint
            0.0001  # outputscale_constraint
        ]
        self.param = CUDA(torch.tensor(self.param))

        # initialize model and likelihood
        model_list = []
        likelihood_list = []
        for m_i in range(self.state_dim):
            likelihood = CUDA(gpytorch.likelihoods.GaussianLikelihood(noise_constraint=GreaterThan(self.param[0])))
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

    def reset_parameters(self):
        for m_i in range(self.state_dim):
            self.model.models[m_i].reset_parameters()

    def find_similar_datapoints(self):
        for d_i in range(len(self.data)):
            for d_j in range(d_i+1, len(self.data)):
                diff = torch.sum(torch.abs(self.data[d_i]-self.data[d_j]))
                if diff < CUDA(torch.tensor(0.01)):
                    print(diff)

    def train_model(self):
        # prepare data
        train_x = self.data[:, :self.input_dim]
        train_y = self.data[:, self.input_dim:]

        # reset training data
        # DO NOT reset parameters, use the parameters of last time
        for m_i in range(self.state_dim):
            self.model.models[m_i].set_train_data(train_x, train_y[:, m_i], strict=False)
            #self.model.models[m_i].prediction_strategy = None

        # gpytorch.settings.fast_computations(solves=False), gpytorch.settings.max_eager_kernel_size(1e20),
        # before we reach max_cholesky_size, we will always use cholesky decomposition, which is very memory comsuming.
        # after the max_cholesky_size, we will use conjugate gradient method, which will save much memory.
        with gpytorch.settings.max_cholesky_size(2000):
            # training stage
            self.model.train() # set prediction_strategy = None inside
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
                
                # accmulate the gradient and train
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

    def normalization(self, x):
        # we only modify the input, the output can learn this scale
        x[:, 4] = x[:, 4]/self.norm
        #if x.shape[1] > self.input_dim:
        #    x[:, 10] = x[:, 10]/self.norm
        return x

    def denormalization(self, x):
        x[:, 4] = x[:, 4]*self.norm
        return x
        
    def add_point(self, x, i):
        # add some noise to the data
        #x_noise = np.random.normal(0, 0.0001, self.state_dim)
        #x[:, self.input_dim:] = x[:, self.input_dim:] + x_noise[None]
        #print('x', x)

        # normalization will make the clustering unstable
        x = self.normalization(x)
        #print('normalized x', x)
        
        if self.data is None:
            self.data = CUDA(torch.Tensor(x))
        else:
            self.data = torch.cat((self.data, CUDA(torch.tensor(x))), dim=0)
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
        x = self.normalization(x)
        x = CUDA(torch.Tensor(x))
        test_x = x[:, :self.input_dim]
        test_y = x[:, self.input_dim:]

        # get the log likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # [TODO] if likelihood is added here, the new component is hard to create
            test_all = []
            for t_i in range(self.state_dim):
                test_all.append(test_x)
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

                '''
                if self.n > 1:
                    print(
                        'prediction', sample_func[f_i].loc.detach().cpu().numpy(), 
                        'variance', sample_func[f_i].covariance_matrix.detach().cpu().numpy(), 
                        'observation', test_y[:, f_i].item()
                    )
                '''
                
                if np.isnan(incre):
                    print('---------------NaN detected---------------')
                    print('f_i: ', f_i)
                    print('x', test_x)
                    print('y', test_y)
                    print('y[f_i] ', test_y[:, f_i])
                    print('loc: ', sample_func[f_i].loc.numpy())
                    print('covariance_matrix: ', sample_func[f_i].covariance_matrix.detach().cpu().numpy())
                    print('likelihood.covariance_matrix: ', sample_func[f_i].covariance_matrix.detach().cpu().numpy())
                    #print('likelihood.noise_covar.raw_noise', self.model.models[f_i].likelihood.noise_covar.raw_noise)
                    #print(self.model.models[f_i].state_dict())
                    print('lengthscale', self.model.models[f_i].covar_module.base_kernel.lengthscale)
                    print('lengthscale', self.model.models[f_i].covar_module.outputscale)
                    print('------------------------------------------')
                #else:
                #    print('lengthscale', self.model.models[f_i].covar_module.outputscale)
                #    print('lengthscale', self.model.models[f_i].covar_module.base_kernel.lengthscale)
                    #print('covariance_matrix: ', sample_func[f_i].covariance_matrix.detach().cpu().numpy())
                    #print('likelihood.covariance_matrix: ', sample_func_lik[f_i].covariance_matrix.detach().cpu().numpy())
                #    #print(self.model.models[f_i].state_dict())
                #    print('------------------------------------------')
        #if self.n > 1:
        #    print('-------------------')
        # since we added all likelihood together
        return log_ppf/len(sample_func)

    def predict(self, x):
        # prepare data
        x = self.normalization(x)
        # when we calculate kld, the x will contain the observation_next
        test_x = CUDA(torch.Tensor(x[:, :self.input_dim]))

        # get the log likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_all = []
            for t_i in range(self.state_dim):
                test_all.append(test_x)
            sample_func = self.model(*test_all)
            #sample_func_lik = self.likelihood(*sample_func)
            
        ds = []
        for i in range(self.state_dim):
            ds.append(sample_func[i].loc.cpu().numpy())
        ds = np.array(ds).T
        return ds
