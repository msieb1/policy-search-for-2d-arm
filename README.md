# How to Use

There are two versions to run: Either the probabilistic policy search using Expectation Maximization or gradient based policy search (policy gradient). The first version can be run using example_em.py:

    test = EM()
    test.run_trials()
    

For the EM-based policy search, in the current setup, we proceed as follows: With the given parameters, we run 25 episodes (with 25 random samples of lower-level DMP parameters - we are only optimizing the higher level policy here). We then update the parameters of the higher level policy and run another 25 episodes and update our parameters based on these rollouts. This is done for 100 times overall (one can see that convergence is achieved).    

The second version uses example_pg.py:

    test = PG()
    test.run_trials()
    
For the gradient based policy search, in the current setup, we proceed as follows: With the given parameters, we run 25 episodes (with 25 random samples of lower-level DMP parameters - we are only optimizing the higher level policy here). We then update the parameters of the higher level policy and run another 25 episodes and update our parameters based on these rollouts. This is done for 100 times overall (one can see that convergence is achieved). 

While in the EM we solve for MLE solutions to the variance and mean of the upper-level Gaussian policies, we use gradient descent in the second version. The results will be different and also different approaches can be integrated, for example natural gradients in the policy gradient version.

