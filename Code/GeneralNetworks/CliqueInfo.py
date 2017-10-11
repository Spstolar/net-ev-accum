import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


total_agents = 1000  # this includes the first agent to fire

survival_probability = np.load('survival_prob.npy')
r_probability = np.load('pos_LLR_prob.npy')
agreement_info = np.log(r_probability / (1 - r_probability))
T = np.size(survival_probability)

"""
probability of n choose k with prob p = stats.binom.pmf(k, n, p)
"""


def prob_k_correct(k, n, R):
    return stats.binom.pmf(k, n, R)

def first_exit_prob(n):
    return 1 - survival_probability ** (n+1)

plt.figure(1)
plt.plot(first_exit_prob(total_agents), color='red', linewidth=2)  # survival prob
plt.plot(r_probability, linewidth=2)
plt.axis([0, T, 0, 1])
plt.title('First Exit and Positive LLR Probability')

plt.figure(2)

plt.plot(agreement_info, linewidth=2)
plt.axis([0, T, 0, 3])
plt.title('Agreement Information')

plt.figure(3)
for i in np.arange(total_agents - 10, total_agents+1):
    correct = prob_k_correct(i, total_agents, r_probability)
    plt.plot(correct, linewidth=2, label= str(i) + 'agents')  # survival prob

plt.legend()
plt.axis([0, T, 0, 1])
plt.title('Probability k Correct out of ' + str(total_agents))

plt.figure(4)

expected_info = 0
for k in np.arange(total_agents +1):
    expected_info += prob_k_correct(k, total_agents, r_probability) * (2*k - total_agents) * agreement_info

plt.plot(expected_info)
plt.title('Expected Information At First Decider Time')


plt.show()