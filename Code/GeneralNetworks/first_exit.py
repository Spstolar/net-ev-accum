import numpy as np
import matplotlib.pyplot as plt
"""
This is to simply simulate the first exit time distribution for a group of agents.
Simulate n accumulators and let them independently make observations and gather evidence until one hits the boundary.
Then log the time, store it, and restart.
"""

# Parameters for the simulation
time_limit = 200
num_agents = 10
llr = np.zeros(num_agents)
num_sims = int(1e2)
fpt_log = np.zeros(num_sims)

mean1 = 0.1
mean2 = -0.1
var1 = 1
var2 = 1
bdy_plus = 3
bdy_minus = -3

class Dist:
    """We define a class for distributions so that we can easily access the truth distributions rather than writing out
    the formula for the distribution each time we want to use it."""
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def prob(self, x):
        return np.exp(-np.power(x - self.mean, 2) / (2*self.var))/(np.sqrt(2 * np.pi * self.var))


pos = Dist(mean1, var1)  # the positive state distribution
neg = Dist(mean2, var2)


def compute_llr(x_array, dist1, dist2):
    """
    Computes the log-likelihood ratio for a given array of observations.
    :param x_array: an array of observations
    :param dist1: the positive truth distribution
    :param dist2: the negative truth distribution
    :return: an array the size of x_array of LLRs
    """
    return np.log(dist1(x_array)/dist2(x_array))

s = 0
t = 0

while (s < num_sims):
    t += 1
    obs = np.sqrt(var1) * np.random.randn(num_agents) + mean1
    llr += compute_llr(obs, pos.prob, neg.prob)
    for a in np.arange(num_agents):
        if np.abs(llr[a]) > bdy_plus:  # check if any agent has exited
            fpt_log[s] = t  # log the first passage time
            # reset
            llr = np.zeros(num_agents)
            t = 0
            s += 1  # next simulation
#             print s 

#plot escape times and compute mean
print(fpt_log[:10])
print np.mean(fpt_log)

plt.hist(fpt_log)
plt.show()