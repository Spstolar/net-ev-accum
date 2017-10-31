import numpy as np
import matplotlib.pyplot as plt
"""
This is to simply evidence accumulation for a line of agents.
"""

# How observations are drawn.
mean1 = 0.1
mean2 = -0.1
var1 = 1
var2 = 1
# alpha = 0.8
# bdy_plus = np.log(alpha / (1 - alpha))
bdy_plus = 3
bdy_minus = - bdy_plus


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

# Simulation Parameters
num_agents = 2
num_sims = int(1e1)
time_limit = int(1e3)
bump = np.zeros(num_agents)  # This is how much an agent alters LLR when previous agent decides.
for a in range(0,num_agents):
    bump[a] = (1.1 ** a) * bdy_plus
llr_store = np.zeros((num_agents, time_limit))
final_time = 0
correct = 0  # Number of correct decisions by the end agent.


def check_decision(llr, theta_minus, theta_plus):
    if llr > theta_plus:
        return 1
    elif llr < theta_minus:
        return -1
    else:
        return 0

for s in range(num_sims):
    decided = 0
    t = 0
    fire = 0
    next_fire = 0
    llr = np.zeros(num_agents)
    decisions = np.zeros(num_agents)
    still_accumulating = np.ones(num_agents)

    while decided == 0 and t < time_limit:
        obs = np.sqrt(var1) * np.random.randn(num_agents) + mean1  # make observations
        llr += compute_llr(obs, pos.prob, neg.prob) * still_accumulating  # incorporate obs for active agents

        for a in np.arange(num_agents):
            if still_accumulating[a] == 1:  # skip all of this if the agent is done
                fire = next_fire
                decisions[a] = check_decision(llr[a], bdy_minus, bdy_plus)

                if decisions[a] != 0:  # check if the observation caused a decision
                    # print str(a) + ' decided'
                    next_fire = 1
                    still_accumulating[a] = 0
                else:
                    next_fire = 0
                
                if fire == 1 and still_accumulating[a] == 1:  # if neighbor decided, then bump
                    # print 'fire'
                    llr[a] += decisions[a-1] * bump[a]
                    decisions[a] = check_decision(llr[a], bdy_minus, bdy_plus)
                    if decisions[a] != 0:  # check if the bump caused a decision
                        next_fire = 1
                        still_accumulating[a] = 0

        if s == 0:
            llr_store[:, t] = llr  # for visualization

        if decisions[-1] == 1:  # if the last agent decides, log and exit
            correct += 1
            decided = 1
            if s == 0:
                final_time = t
        elif decisions[-1] == -1:
            decided = 1
            if s == 0:
                final_time = t
        t += 1

    if s == 0:
        print decisions

print correct / float(num_sims)

for a in range(num_agents):
    plt.plot(llr_store[a,:final_time+1], label=str(a+1))
plt.xlabel('Time')
plt.ylabel('LLR')
plt.legend()
plt.title('Directed Line' + str(decisions))
plt.show()