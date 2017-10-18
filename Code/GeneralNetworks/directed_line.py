import numpy as np
import matplotlib.pyplot as plt
"""
This is to simply simulate the first exit time distribution for a group of agents.
Simulate n accumulators and let them independently make observations and gather evidence until one hits the boundary.
Then log the time, store it, and restart.
"""

# Parameters for the simulation
time_limit = 300
num_sims = int(1e0)

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
num_agents = 3
bump = bdy_plus * np.ones(num_agents)
fire = 0
next_fire = 0
decisions = np.zeros(num_agents)
correct = 0
llr_store = np.zeros((num_agents, time_limit))
final_time = 0
still_accumulating = np.ones(num_agents)

llr = np.zeros(num_agents)
for s in range(num_sims):
    decided = 0
    while (decided == 0):
        obs = np.sqrt(var1) * np.random.randn(num_agents) + mean1  # make observations
        llr += compute_llr(obs, pos.prob, neg.prob) * still_accumulating
        llr_store[:,t] = llr 
        for a in np.arange(num_agents):
            if still_accumulating[a] == 1:  # skip all of this if the agent is done
                fire = next_fire
                if np.abs(llr[a]) > bdy_plus:  # check if any agent has exited
                    next_fire = 1
                    still_accumulating[a] = 0
                    if llr[a] > bdy_plus:
                        decisions[a] = 1
                    else:
                        decisions[a] = -1
                else:
                    next_fire = 0
                
                if fire == 1 and still_accumulating[a] == 1:
                    llr[a] += decisions[a-1] * bump[a]
                    
                if np.abs(llr[a]) > bdy_plus and still_accumulating[a] == 1:  # check if any agent has exited
                    next_fire = 1
                    still_accumulating[a] = 0
                    if llr[a] > bdy_plus:
                        decisions[a] = 1
                    else:
                        decisions[a] = -1
                else:
                    next_fire = 0
        if decisions[-1] == 1:
            correct += 1
            decided = 1
            final_time = t
        elif decisions[-1] == -1:
            decided = 1
            final_time = t
        t += 1
    print decisions
                
for a in range(num_agents):
    plt.plot(llr_store[a,:final_time+1], label=str(a+1))
plt.xlabel('Time')
plt.ylabel('LLR')
plt.legend()
plt.title('Directed Line' + str(decisions))
plt.show()