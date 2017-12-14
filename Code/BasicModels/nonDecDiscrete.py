import numpy as np
import matplotlib.pyplot as plt

# Indexing Guide
# -2 -1 0 1 2 3
#     0 1 2 3
negThresh = -1
posThresh = 2
N = int(1e2)  # time steps

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

print staticObsProb
print posObsProbP

print stateTrackP
print stateTrackM

evidence = np.log(survivalProbPlus / survivalProbMinus)

max_ev = np.ceil(np.max(evidence))
plt.scatter(np.arange(N), evidence)
plt.xlabel('Time')
plt.xticks([0, N/2, N])
plt.yticks([0, max_ev / 2.0, max_ev])
plt.ylabel('Evidence')
plt.title('Non-Decision Evidence for ' + r'$\theta_- = $' + str(negThresh) + r', $\theta_+ = $' + str(posThresh))
plt.show()
# stateTrackM = zeros(N, total_state)
# stateTrackM(1,:) = zeros(1,total_state)
# stateTrackM(1, abs(negThresh)) = 1  #start off with everything concentrated at 0
# negObsProbM = .7
# posObsProbM = .3
# prob0HM = zeros(1,N)
# prob0HM(1) = 1
#
#
#
# for i in range(1,N):
#     stateTrackM(i,:) = negObsProbM*[stateTrackM(i-1, 2:total_state) 0] + posObsProbM*[0 stateTrackM(i-1, 1: total_state - 1)]
#     outLeft = negObsProbM*stateTrackM(i-1,1)
#     outRight = posObsProbM*stateTrackM(i-1, total_state)
#     escapeProb = outLeft + outRight
#     prob0HM(i) = 1 - escapeProb
#     stateTrackM(i,:) = stateTrackM(i,:)/sum(stateTrackM(i,:))
#
#
#
#
# logLikelihood = log( prob0HP ./ prob0HM )
# logLikelihood(1: min(abs(negThresh), posThresh )) = 0
#
# plot( 1:N, logLikelihood)
#
# hold on
# expectation_states = zeros(1,N)
# posStates = (negThresh+1):(posThresh-1)
#
# for j=1:N
#     expectation_states(j) = dot(posStates , stateTrack(j,:))
# end
#
# plt.plot(np.arange(1,N+1), expectation_states)