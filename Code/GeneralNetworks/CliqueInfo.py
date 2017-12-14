import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

"""
This does a few things:
1. Plots first exit probabilities.
2. Computes the information from knowing an agent has positive LLR.
3. Computes probabilities for different numbers of correct agents.
4. Plots the expected amount of information agents receive after a first-decider and agreement wave.
"""


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

plt.subplot(221)  # subplot(mni) puts the next plot in the i-th location of m by n multiplot
plt.plot(first_exit_prob(total_agents), color='red', linewidth=2)  # survival prob
r_probability[:10] = 0.5
plt.plot(r_probability, linewidth=2)
plt.axis([0, T, 0, 1 + .1])
plt.xticks([0, T/2, T-1])
plt.yticks([0, 0.5, 1])
plt.title('First Exit and Agreement')
plt.ylabel('ylabel', fontsize=18)
plt.ylabel('Probability')

plt.subplot(222)
for i in np.arange(total_agents/2, total_agents):
    line_color = (0.1, float(i) / total_agents, 1 - float(i) / total_agents )
    correct = prob_k_correct(i, total_agents-1, r_probability)[:T/2]
    plt.plot(correct, linewidth=2, label=str(i), color=line_color)  # survival prob

# plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0, prop={'size': 9})
plt.axis([0, T/2, 0, 1.1])
plt.title('k Correct out of ' + str(total_agents -1))
plt.xticks([0, T/4, T/2])
plt.yticks([0, 0.5, 1])


plt.subplot(223)
expected_info = 0
for k in np.arange(total_agents +1):
    expected_info += prob_k_correct(k, total_agents, r_probability) * (2*k - total_agents) * agreement_info
expected_info[:10] = 0
plt.plot(expected_info)
plt.title('Expected Evidence At T')
plt.xticks([0, T/2, T-1])
max_expected_info = np.max(expected_info)
plt.axis([0, T, 0, max_expected_info + 0.2])
plt.yticks([0, int(max_expected_info / 2), max_expected_info])
plt.ylabel('ylabel', fontsize=18)
plt.ylabel('Evidence')
plt.xlabel('xlabel', fontsize=18)
plt.xlabel('Time')

plt.subplot(224)
agreement_info[:10] = 0
plt.plot(agreement_info, linewidth=2)
max_agreement_info = np.max(agreement_info)
plt.axis([0, T, 0, max_agreement_info + .1])
plt.xticks([0, T/2, T-1])
plt.yticks([0, int(max_agreement_info/2), max_agreement_info])
plt.title('Agreement Information')
plt.xlabel('xlabel', fontsize=18)
plt.xlabel('Time')





plt.tight_layout()

plt.show()