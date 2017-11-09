from matplotlib import pyplot as plt
import numpy as np
from Pend2dBallThrowDMP import *

class EM():

    def __init__(self):
        self.env = Pend2dBallThrowDMP()
        self.lambd = 7
        self.numDim = 10
        self.numSamples = 25
        self.maxIter = 100
        self.numTrials = 10
        self.saveFigures = True
        # For example, let initialize the distribution...





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

    def calculate_w(self, R, theta):
        # calculate the weights
        lambd = self.lambd
        numSamples = self.numSamples
        w = np.zeros(numSamples)
        beta = lambd / (np.max(R) - np.min(R))
        for i in range(0, numSamples):
            w[i] = np.exp(beta * (R[i] - np.max(R)))
        return w

    # update omega
    def update_omega(self, w, theta):
        # calculate mean
        Mu_w = 0
        for l in range(0, self.numSamples):
            Mu_w += w[l]*theta[:,l]
        Mu_w /= np.sum(w)
        # calculate covariance
        Sigma_w = 0
        for l in range(0, self.numSamples):
            Sigma_w += w[l]*(np.outer((theta[:,l] - Mu_w),(theta[:,l] - Mu_w)))
        Sigma_w /= np.sum(w)
        return Mu_w, Sigma_w

    def run_trials(self):
        maxIter = self.maxIter
        numSamples = self.numSamples
        numDim = self.numDim
        numTrials = self.numTrials
        env = self.env
        cnt = 0
        lambd_list = [7, 3, 25]

        # Run Trials for different temperature values lambda
        for lambd in lambd_list:
            self.lambd = lambd

            R_mean_storage = np.zeros((maxIter,numTrials))
            R_mean = np.zeros(maxIter)
            R_std = np.zeros(maxIter)

            for t in range(0, numTrials):
                R_old = np.zeros(numSamples)
                Mu_w = np.zeros(numDim)
                Sigma_w = np.eye(numDim) * 1e6
                for k in range(0, maxIter):
                    # sample the R and theta vector (25 values each) for the current iteration
                    R, theta = self.calculate_R_and_theta(Mu_w, Sigma_w)
                    if np.linalg.norm(np.mean(R_old) - np.mean(R)) < 1e-3:
                        break
                    # get the weights
                    w = self.calculate_w(R, theta)
                    # update the parameters
                    Mu_w, Sigma_w = self.update_omega(w, theta)
                    # Regularization
                    Sigma_w += np.eye(numDim)
                    mR = np.mean(R)
                    # store the average return of current run
                    R_mean_storage[k,t] = mR
                    R_old = R
                    if k == maxIter and t == numTrials:
                        print(np.mean(R))
            R_mean = np.mean(R_mean_storage, axis=1)
            R_std = np.sqrt(np.diag(np.cov(R_mean_storage)))
            print("Average return of final policy: ")
            print(R_mean[-1])
            print("\n")

            if cnt == 0:
                plt.errorbar(np.arange(1, maxIter + 1), R_mean,  1.96 * R_std, marker='^',color='blue', label='lambda = 7')
            elif cnt == 1:
                plt.errorbar(np.arange(1, maxIter + 1), R_mean,  1.96 * R_std, marker='^',color='green', label='lambda = 3')
            elif cnt == 2:
                plt.errorbar(np.arange(1, maxIter + 1), R_mean,  1.96 * R_std, marker='^', color='red',label='lambda = 25')

            plt.yscale("symlog")
            if self.saveFigures:
                plt.savefig('1lambda3725.pdf')
            # Save animation
            env.animate_fig ( np.random.multivariate_normal(Mu_w,Sigma_w) )
            plt.savefig('EM-Ex3.pdf')
            cnt += 1
        plt.legend(loc='best')

if __name__ == '__main__':

    plt.figure()
    plt.xlabel("Number of Runs")
    plt.ylabel("Average return")
    plt.hold('on')
    test = EM()
    test.run_trials()