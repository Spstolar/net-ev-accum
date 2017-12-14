import numpy as np
import matplotlib.pyplot as plt
"""
This handles simulation of the evidence accumulation process directly. An agent makes a predefined number of
observations and the derived information is computed exactly, rather than being approximated with a FP-solution.
"""


# Parameters for the simulation
length = 1000
p = np.e / 7  # need the ratio of p and q to be e
q = 1. / 7
s = 1 - p - q  # this has it around .47
bdy_plus = 60
bdy_minus = -10
evidence_amounts = np.array((1, 0, -1))
soc_ev = np.zeros(length)

A = np.array([[s, q], [p, s]])
start = np.array([1,0])
for i in range(length):
    A_power = np.linalg.matrix_power(A, i + 1)  # since i starts at 0
    states = np.matmul(A_power, start)
    soc_ev[i] = np.sum(states)


# Observations are drawn from the true multnomial distribution.
# [1, 0, 0] with prob p
# [0, 1, 0] with prob s
# [0, 0, 1] with prob q
obs1 = np.random.multinomial(1, [p,s,q], size=length)
obs2 = np.random.multinomial(1, [p,s,q], size=length)

#  Compute the evidence gained from each observation.
llr1 = np.matmul(obs1, evidence_amounts)
llr2 = np.matmul(obs2, evidence_amounts)

#  Compute the private evidence time series for both agents.
ev1 = np.cumsum(llr1)
ag2_private = np.cumsum(llr2)


done1 = 0  # To stop gathering evidence after hitting a threshold.
done2 = 0
ag1_fire = 0  # To record the firing time of agent 1.

for i in range(ev1.size):
    if done1 == 0:
        if ev1[i] >= bdy_plus:  # Check if 1's private info has hit a boundary
            ev1[i] = bdy_plus  # Fix it at that boundary.
            done1 = 1  # Stop accumulating.
            ag1_fire = i  # Record time.
        elif ev1[i] <= bdy_minus:
            ev1[i] = bdy_minus
            done1 = 1
            ag1_fire = i
    else:
        ev1[i] = ev1[i-1]  # When the accumulation is done, just keep it at that last threshold value.




plt.figure(figsize=(8, 4))  # Start the figure


plt.subplot(121)  # subplot(mni) puts the next plot in the i-th location of m by n multiplot
plt.scatter(np.arange(length+1), np.hstack((0,ev1)))

plt.xlabel('Time')
plt.ylabel('LLR')
plt.title('Agent 1')
# plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.axis([0, length, bdy_minus - 0.1, bdy_plus + 0.1])
# plt.grid(True)

###########
# AGENT 2 #
###########

# Now for agent 2 we compute the social evidence.
plt.subplot(122)

# Indexing Guide
# -2 -1 0 1 2 3
#     0 1 2 3
negThresh = bdy_minus
posThresh = bdy_plus
N = length  # time steps

total_state = posThresh + np.abs(negThresh) + 1 - 2  # include 0 but exclude the threshold values
stateVec = np.zeros(total_state)
stateTrackP = np.zeros((N, total_state))
stateTrackP[0, :] = np.zeros(total_state)
stateTrackP[0, np.abs(negThresh) - 1] = 1  # start off with everything concentrated at 0
# print stateTrackP[0,:]

stateTrackM = np.zeros((N, total_state))
stateTrackM[0, :] = np.zeros(total_state)
stateTrackM[0, np.abs(negThresh) - 1] = 1

negObsProbP = 1.0 / 5
posObsProbP = np.exp(1) / 5
staticObsProb = 1 - negObsProbP - posObsProbP
survivalProbPlus = np.zeros(N)
survivalProbPlus[0] = 1

negObsProbM = posObsProbP
posObsProbM = negObsProbP
survivalProbMinus = np.zeros(N)
survivalProbMinus[0] = 1

for i in range(1, N):
    stateTrackP[i,:] = (negObsProbP * np.hstack((stateTrackP[i-1, 1:], 0))
                         + posObsProbP * np.hstack((0, stateTrackP[i-1, :-1]))
                         + staticObsProb * stateTrackP[i-1, :])
    # print stateTrackP[i,:]

    stateTrackM[i, :] = (negObsProbM * np.hstack((stateTrackM[i - 1, 1:], 0))
                         + posObsProbM * np.hstack((0, stateTrackM[i-1, :-1]))
                         + staticObsProb * stateTrackM[i-1, :])

    survivalProbPlus[i] = np.sum(stateTrackP[i, :])
    survivalProbMinus[i] = np.sum(stateTrackM[i, :])

# print staticObsProb
# print posObsProbP
#
# print stateTrackP
# print stateTrackM

ag2_social = np.log(survivalProbPlus / survivalProbMinus)
ag2_social[ag1_fire:] = ev1[ag1_fire] # No non-decision evidence after the first agent decides.
# Compute the LLR by adding the private and bump to the non-decision social evidence.
ev2 = ag2_private + ag2_social

# Compute the time series by using the threshold check.
for i in range(ev2.size):
    if done2 == 0:
        if ev2[i] >= bdy_plus:
            ev2[i] = bdy_plus
            done2 = 1
            ag2_fire = i
        elif ev2[i] <= bdy_minus:
            ev2[i] = bdy_minus
            done2 = 1
            ag2_fire = i
    else:
        ev2[i] = ev2[i - 1]


max_ev = np.ceil(np.max(ag2_social))
plt.scatter(np.arange(N+1), np.hstack((0, ag2_social)), c='red', label='Soc')
plt.scatter(np.arange(ag2_fire+1), np.hstack((0,ag2_private[:ag2_fire])), c='purple', label='Priv')
plt.scatter(np.arange(N+1), np.hstack((0,ev2)), c='orange', label=r'$y^{(2)}$')

plt.legend()
plt.xlabel('Time')
# plt.xticks([0, N/2, N])
# plt.yticks([0, max_ev / 2.0, max_ev])
plt.axis([0, length, bdy_minus - 0.1, bdy_plus + 0.1])
plt.ylabel('Evidence')
# plt.title('Non-Decision Evidence for ' + r'$\theta_- = $' + str(negThresh) + r', $\theta_+ = $' + str(posThresh))
plt.title('Agent 2')

plt.tight_layout()
plt.show()

