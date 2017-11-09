from matplotlib import pyplot as plt
import numpy as np
from Pend2dBallThrowDMP import *
from enum import Enum


class PG():
    """
    Implements gradient descent to update parameters of the Gaussians parametrizing the upper-level policy.
    """

    def __init__(self):
    	"""
        numDim: dimension of state space
		numSamples: number of episodic rollouts per iteration
		maxIter: number of parameter updates
		numTrials: number of independent learning trials
    	"""
        self.env = Pend2dBallThrowDMP()
        self.lambd = 7
        self.numDim = 10
        self.numSamples = 25
        self.maxIter = 100
        self.numTrials = 10
        self.sigm = 10
        self.alpha = 0.1
        self.lower = 0.1
        self.gamma = 0.9
        self.saveFigures = True
        self.fullGrad = False
        self.gradMethod = 'NAG' #alternatively 'GD'


    # Do your learning
    def calculate_R_and_theta(self, Mu_w, Sigma_w):
    # initialize theta vector (2D array: state dimension x number of samples)
    # and R (reward) vector (1D array: reward per episode)
    # and w (weights) vector (1D array: weight of current sample)
        numDim = self.numDim
        numSamples = self.numSamples
        env = self.env
        theta = np.zeros((numDim, numSamples))
        R = np.zeros(numSamples)
        for i in range(0, numSamples):
            # ... then draw a sample and simulate an episode
            sample = np.random.multivariate_normal(Mu_w, Sigma_w)
            reward = env.getReward(sample)
            theta[:,i] = sample
            R[i] = reward
        return R, theta


    # update omega
    def update_gradient(self, theta, R, Mu_w, Sigma_w):
        # calculate Parameter Exploring Policy Gradient (PGPE)
        Mu_gradient = 0
        Sigma_gradient = np.zeros((self.numDim, self.numDim))
        # subtract baseline later to reduce variance
        baseline = np.mean(R)
        Sigma_w_inv = np.linalg.inv(Sigma_w)
        for l in range(0, self.numSamples):

            if self.fullGrad:
                Mu_gradient += (theta[:,l] - Mu_w).dot(Sigma_w_inv) * (R[l] - baseline)
                Sigma_gradient += -0.5*(Sigma_w_inv - Sigma_w_inv.T.dot(np.outer((theta[:,l] - Mu_w) ,(theta[:,l] - Mu_w))) \
                    .dot(Sigma_w_inv.T))*(R[l] - baseline)
            else:
                #diag(sigma^-2):
                sub_0 = np.eye(self.numDim)
                np.fill_diagonal(sub_0, np.diag(Sigma_w)**(-1))
                # diag(sigma^-3):
                sub_1 = np.eye(self.numDim)
                np.fill_diagonal(sub_1, np.diag(Sigma_w)**(-1.5))
                # diag(sigma^-1):
                sub_2 = np.eye(self.numDim)
                np.fill_diagonal(sub_2, np.diag(Sigma_w)**(-0.5))
                #diag(theta-mu)(theta-mu)^T:
                sub_3 = np.eye(self.numDim)
                np.fill_diagonal(sub_3, np.diag((np.outer((theta[:,l] - Mu_w) ,(theta[:,l] - Mu_w)))))

                Mu_gradient += (theta[:,l] - Mu_w).dot(sub_0) * (R[l] - baseline)
                Sigma_gradient += -(sub_2 - (sub_3.dot(sub_1)))*(R[l] - baseline)
        return Mu_gradient/self.numSamples, Sigma_gradient/self.numSamples

    def run_trials(self):
        maxIter = self.maxIter
        numSamples = self.numSamples
        numDim = self.numDim
        numTrials = self.numTrials
        env = self.env
        gamma = self.gamma
        alpha_list = [0.4, 0.1]

        # run trials for different values of alpha
        for alph in alpha_list:
            self.alpha = alph

            R_mean_storage = np.zeros((maxIter, numTrials))
            R_mean = np.zeros(maxIter)
            R_std = np.zeros(maxIter)

            for t in range(0, numTrials):
                R_old = np.zeros(numSamples)
                Mu_w = np.zeros(numDim)
                sigm = self.sigm
                Sigma_w = np.eye(numDim) * sigm**2
                Mu_grad_old = 0
                Sigma_grad_old = np.zeros((self.numDim, self.numDim))

                for k in np.arange(0, maxIter):
                    #alpha = self.alpha / (k/13.0+1)
                    alpha = self.alpha
                    R, theta = self.calculate_R_and_theta(Mu_w, Sigma_w)
                    #if np.linalg.norm(np.mean(R_old) - np.mean(R)) < 1e-3:
                     #   break
                    Mu_gradient, Sigma_gradient = self.update_gradient(theta, R, Mu_w, Sigma_w)

                    if self.gradient_method == 'NAG':
                        # use Nesterov accelerated gradient
                        Mu_gradient = gamma*Mu_grad_old + alpha*(Mu_gradient - gamma*Mu_grad_old)
                        Sigma_gradient = gamma*Sigma_grad_old + alpha*(Sigma_gradient - gamma*Sigma_grad_old)/15

                        Mu_w += Mu_gradient
                        Sigma_w += Sigma_gradient
                    else:
                        Mu_w += alpha*Mu_gradient
                        #Sigma_w += alpha*Sigma_gradient
                    # ensure positive eigenvalues for positive semi definite property
                    w, V = np.linalg.eig(Sigma_w)
                    if np.min(w) < self.lower:
                        for j in range(0, np.shape(w)[0]):
                            if w[j] < self.lower:
                                w[j] = self.lower
                        Sigma_w = V.dot(np.diag(w)).dot(np.linalg.inv(V))
                    mR = np.mean(R)
                    R_mean_storage[k, t] = mR
                    R_old = R
                    Mu_grad_old = Mu_gradient
                    Sigma_grad_old = Sigma_gradient
                    if k == maxIter and t == numTrials:
                        print(np.mean(R))
            R_mean = np.mean(R_mean_storage, axis=1)
            R_std = np.sqrt(np.diag(np.cov(R_mean_storage)))
            print("\n")
            if self.gradient_method == 'NAG':
                plt.errorbar(np.arange(1, maxIter + 1), R_mean, 1.96 * R_std, marker='^', color='red',
                             label='NAG')
                plt.legend(loc='best')

            elif self.gradient_method == 'GD':
                plt.errorbar(np.arange(1, maxIter + 1), R_mean, 1.96 * R_std, marker='^', color='green',
                             label='alpha = 0.1, no baseline')
                plt.legend(loc='best')

            plt.yscale("symlog")
            # Save animation
            if self.saveFigures:
                plt.savefig('MeanVarianceGradalpha01meanwbaseline.pdf')
            #env.animate_fig ( np.random.multivariate_normal(Mu_w,Sigma_w) )
        

if __name__ == '__main__':
    plt.figure()
    plt.hold('on')
    plt.xlabel("Number of Runs")
    plt.ylabel("Average return")
    test = PG()
    test.run_trials()