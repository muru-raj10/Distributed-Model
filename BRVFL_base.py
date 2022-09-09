#Bayesian RVFL base code
from scipy.special import expit #logistic function 1/1+exp(-x)
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_random_state
from sklearn.linear_model import Ridge #, BayesianRidge, ARDRegression, Lasso, LogisticRegression
import numpy as np
import theano.tensor as tt
from theano import shared
from theano.tensor.nnet import sigmoid
from abc import ABC, abstractmethod
import pymc3 as pm


class RVFLRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, neurons=50, C=10 ** -2, random_state=None):
        self.neurons = neurons
        self.C = C  #the gamma
        self.random_state = random_state

    def fit(self, X, y):
        N, d = X.shape

        self.init_hidden_matrices(d)

        H = self.build_H(X)
        self.W_o = np.linalg.inv(H.T.dot(H) + self.C * np.eye(self.neurons)).dot(H.T.dot(y)) #Dimension depends on y. Possible to autoencode!
        self.W_o_sig = np.linalg.inv(H.T.dot(H) + self.C * np.eye(self.neurons)).dot(H.T.dot(y))
        return self

    def fit_hp(self,X,y):
        #fit with hyper prior considerations
        #to override
        return self

    def init_hidden_matrices(self, d):
        r = check_random_state(self.random_state)
        self.W_h = r.rand(d, self.neurons) - 0.5
        self.b_h = r.rand(self.neurons) - 0.5

    def predict(self, X):
        H = self.build_H(X)
        return H.dot(self.W_o) #MAP estimate 56000x63

    def build_H(self, X):
        H = X.dot(self.W_h) + self.b_h
        #for skip connections, just add here!
        return expit(H)  # logistic function 1/1+exp(-x)


class BRVFL_cl_f(RVFLRegressor):
    #closed form solutions
    #Mean is prior mean, Gamma is the prior variance, sigma is likelihood variance, gamma is prior precision value for init

    def __init__(self, neurons=50, random_state=None, sigma = 10**-2, gamma = 10**-2, Mean=np.zeros((50,20)), Gamma = np.eye(50), omega=1.0):
        RVFLRegressor.__init__(self, neurons=neurons, random_state=random_state)
        self.gamma = gamma
        # should be gamma*identity matrix but we are separating for hyperprior and weights prior
        self.Gamma = Gamma
        self.Mean = Mean
        self.sigma = sigma
        self.omega = omega #omega is for power EP. alpha divergence. refer to EP paper (not used here)

    def fit(self, X, y):
        N, d = X.shape
        self.init_hidden_matrices(d)
        H = self.build_H(X) #56000x100
        # Initialization of the parameters
        Hy = H.T.dot(y)  #100x63
        cov = np.linalg.inv(np.linalg.inv(self.Gamma) + (self.omega / self.sigma)* H.T.dot(H)) #shape of H = (56000,100) -> shape of H.T H -> 100x100
        m = cov.dot(np.linalg.inv(self.Gamma).dot(self.Mean)+(self.omega / self.sigma)*Hy) #100x63

        self.W_o = m #MAP estimate
        self.W_o_sig = cov #100x100. full matrix!

    def fit_hp(self,X,y):
        N, d = X.shape
        self.init_hidden_matrices(d)
        H = self.build_H(X) #56000x100

        pred_mean = H.dot(self.W_o)
        self.nbr_data = N  #to send to central
        self.tot_sq_res = np.sum((y - pred_mean) ** 2) #send to central #this is correct since it's like RE
        #self.mean_sq_res = np.mean(np.sum((y - pred_mean) ** 2,axis=0))  # send to central

        eig = np.linalg.eig(H.T.dot(H))[0] #not being used
        lambda_ = (1.0 / self.sigma) * eig #not being used. omega not accounted here for distributed setting
        self.delt = np.sum(lambda_ / (self.gamma + lambda_)) #send to central

    def predict(self, X):
        #goes in one data point by one
        #predictive mean and variance
        H = self.build_H(X)
        pred_mean = H.dot(self.W_o) #MAP estimate 56000x63
        pred_var = self.sigma + (H.dot(self.W_o_sig)).dot(H.T)
        return pred_mean, pred_var

    def predict_mat(self, X):
        #the whole test set goes in once. Won't work if test set is too large.
        #predictive mean and variance
        H = self.build_H(X)
        pred_mean = H.dot(self.W_o) #MAP estimate 56000x63
        pred_var = self.sigma + np.diag((H.dot(self.W_o_sig)).dot(H.T))
        return pred_mean, pred_var

class SklearnBRVFL(RVFLRegressor):

    def __init__(self, neurons=50, random_state=None, ard=False):

        self.neurons = neurons
        self.random_state = random_state
        self.ard = ard

    def fit(self, X, y):

        self.init_hidden_matrices(X.shape[1])
        H = self.build_H(X)
        if not self.ard:
            self.sklearn_model = Ridge(fit_intercept=False).fit(H,y)
        else:
            pass
            # self.sklearn_model = ARDRegression(fit_intercept=False, lambda_1=25.0, lambda_2=1.0/2.5).fit(H, y)
            #self.sklearn_model = ARDRegression(fit_intercept=False, n_iter=50).fit(H, y)
        self.W_o = (self.sklearn_model.coef_).T
        return self


class BRVFL_cl_f_skip(RVFLRegressor):
    #closed form solutions with skip connections. build_h is overridden
    #Mean is prior mean, Gamma is the prior variance, sigma is likelihood variance

    def __init__(self, neurons=50, random_state=None, sigma = 10**-2, Mean=np.zeros((50,20)), Gamma = np.eye(50)):
        RVFLRegressor.__init__(self, neurons=neurons, random_state=random_state)
        self.Gamma = Gamma
        self.Mean = Mean
        self.sigma = sigma

    def fit(self, X, y):
        N, d = X.shape
        self.init_hidden_matrices(d)
        H = self.build_H(X) #56000x100

        # Initialization of the parameters
        Hy = H.T.dot(y)  #100x63
        cov = np.linalg.inv(np.linalg.inv(self.Gamma) + (1.0 / self.sigma)* H.T.dot(H)) #shape of H = (56000,100) -> shape of H.T H -> 100x100
        m = cov.dot(np.linalg.inv(self.Gamma).dot(self.Mean)+(1.0 / self.sigma)*Hy) #100x63

        self.W_o = m #MAP estimate
        self.W_o_sig = cov #100x100. full matrix!

    def predict(self, X):
        #goes in one data point by one
        #predictive mean and variance
        H = self.build_H(X)
        pred_mean = H.dot(self.W_o) #MAP estimate 56000x63
        pred_var = self.sigma + (H.dot(self.W_o_sig)).dot(H.T)
        return pred_mean, pred_var

    def build_H(self, X):
        H = X.dot(self.W_h) + self.b_h
        #for skip connections, just add here!
        exp_H = expit(H)  # logistic function 1/1+exp(-x)
        hx = np.concatenate((exp_H,X),axis=1) #add skip connections
        return hx



class AbstractBayesianRVFL(RVFLRegressor, ABC):
    TRAIN_MAP = "MAP"
    TRAIN_VAR = "VAR" #Full rank advi
    TRAIN_VAR_ADVI = "VAR_ADVI" #advi propagating only variances and not covariances
    TRAIN_MCMC = "MCMC"  #for bimodal posteriors distributions but in EP you are still projecting on to an exponential family
    TRAIN_MCMC_DIAG = "MCMC_DIAG" #diagonal covariance matrix

    #Mean is prior mean, Gamma is prior variance, sigma is likelihood variance

    def __init__(self, neurons=50, random_state=None, method=TRAIN_MAP, sigma = 10**-2,Mean = np.zeros(50),Gamma=100*np.eye(50)):

        self.neurons = neurons
        self.method = method
        self.random_state = random_state
        self.sigma = sigma  #precision parameter
        self.Mean = Mean
        self.Gamma = Gamma

    def build_H(self):
        H = self.X_shared.dot(self.W_h) + self.b_h
        return sigmoid(H).eval()

    def fit(self, X, y):

        self.init_hidden_matrices(X.shape[1])

        self.X_shared = shared(X)
        self.y_shared = shared(y)

        self.model = self.build_pymc3_model()
        self.map_estimate = pm.find_MAP(model=self.model)

        if self.method == self.TRAIN_MAP:
            self.W_o = tt.cast(self.map_estimate['W_o'], 'float64').eval()
        elif self.method == self.TRAIN_VAR:
            with self.model:
                #self.mean_field = pm.ADVI(n=60000, start=self.map_estimate)
                self.mean_field = pm.fit(n=1000, method='fullrank_advi', start=self.map_estimate)
                self.trace = self.mean_field.sample(draws=1000)
            self.W_o = tt.cast(np.mean(self.trace['W_o'],axis=0), 'float64').eval()
            self.W_o_sig= tt.cast(np.cov(self.trace['W_o'],rowvar=False), 'float64').eval()
        elif self.method == self.TRAIN_VAR_ADVI:
            with self.model:
                #self.mean_field = pm.ADVI(n=60000, start=self.map_estimate)
                self.mean_field = pm.fit(n=1000,method='advi',start=self.map_estimate)  #advi assumes gaussian
                self.trace = self.mean_field.sample(draws=1000)
            self.W_o = tt.cast(np.mean(self.trace['W_o'],axis=0), 'float64').eval()
            self.W_o_sig= tt.cast(np.cov(self.trace['W_o'],rowvar=False), 'float64').eval()
        else:
            with self.model:
                self.trace = pm.sample(10000, start=self.map_estimate)
                self.trace = self.trace[-1000]
            self.W_o = tt.cast(self.trace['W_o'].mean(axis=1), 'float64').eval()
            self.W_o_sig = tt.cast(np.cov(self.trace['W_o'],rowvar=False), 'float64').eval()
        return self

    def predict(self, X):
        #goes in one data point by one
        # if self.method == self.TRAIN_MAP:
        H = self.build_H(X)
        pred_mean = H.dot(self.W_o) #MAP estimate 56000x63
        pred_var = self.sigma + (H.dot(self.W_o_sig)).dot(H.T)
        # else:
        #    ppc = sample_ppc(self.trace, model=self.model, samples=500)
        #    return ppc['y_likelihood'].mean(axis=0)
        return pred_mean, pred_var


    @abstractmethod
    def build_pymc3_model(self):
        pass


class VarRVFLRegressor(AbstractBayesianRVFL):

    def build_pymc3_model(self):
        basic_model = pm.Model()
        with basic_model:
            # prior
            W_o = pm.MvNormal('W_o',mu=self.Mean,cov=self.Gamma, shape=self.neurons) #for one dimensional 1 shape of in put (n,)
            #W_o = pm.MvNormal('W_o', mu=np.zeros(self.neurons), cov=1000 * np.eye(self.neurons), shape=(self.neurons,2)) #not working

            # sigma = Gamma('sigma', alpha=10**-6, beta=10**-6)

            mu = tt.dot(self.build_H(), W_o)

            nu = pm.Uniform('nu', lower=1, upper=100) #hyperprior

            # Define Student T likelihood
            y_likelihood = pm.StudentT('y_likelihood', mu=mu, sigma=np.sqrt(self.sigma), nu=nu, observed=self.y_shared)
            #y_likelihood = pm.Normal('y_likelihood', mu=0, sigma=np.sqrt(self.sigma), observed=self.y_shared)

        return basic_model
