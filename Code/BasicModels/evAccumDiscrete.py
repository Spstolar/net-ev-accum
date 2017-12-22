import numpy as np
import matplotlib.pyplot as plt
"""
This handles simulation of the evidence accumulation process directly for two agents with discrete distribution
observations.
"""


def run_sim(length, p, q, s, th_plus, th_minus):
    # length is the number of time steps
    # p,q,s are observation probabilities, p + q + s = 1
    # th_plus is the positive decision threshold

    evidence_amounts = np.array((1, 0, -1))  # we allow three different observations

    # Observations are drawn from the true multinomial distribution.
    # [1, 0, 0] with prob p
    # [0, 1, 0] with prob s
    # [0, 0, 1] with prob q
    obs1 = np.random.multinomial(1, [p,s,q], size=length)
    obs2 = np.random.multinomial(1, [p,s,q], size=length)

    #  Compute the evidence gained from each observation.
    llr1 = np.matmul(obs1, evidence_amounts)  # this converts the observations to the beliefs
    llr2 = np.matmul(obs2, evidence_amounts)

    #  Compute the private evidence time series for both agents.
    ev1 = np.cumsum(llr1)
    ag2_private = np.cumsum(llr2)

    done1 = 0  # To stop gathering evidence after hitting a threshold.
    done2 = 0
    ag1_fire = 0  # To record the firing time of agent 1.

    for i in range(ev1.size):
        if done1 == 0:
            if ev1[i] >= th_plus:  # Check if 1's private info has hit a boundary
                ev1[i] = th_plus  # Fix it at that boundary.
                done1 = 1  # Stop accumulating.
                ag1_fire = i  # Record time.
            elif ev1[i] <= th_minus:
                ev1[i] = th_minus
                done1 = 1
                ag1_fire = i
        else:
            ev1[i] = ev1[i-1]  # When the accumulation is done, just keep it at that last threshold value.

    ###########
    # AGENT 2 #
    ###########

    # Now for agent 2 we compute the social evidence.

    # Indexing Guide for the small threshold example
    # -2 -1 0 1 2 3
    #     0 1 2 3

    total_state = th_plus + np.abs(th_minus) + 1 - 2  # include 0 but exclude the threshold values
    stateTrackP = np.zeros((length, total_state))
    stateTrackP[0, :] = np.zeros(total_state)
    stateTrackP[0, np.abs(th_minus) - 1] = 1  # start off with everything concentrated at 0

    stateTrackM = np.zeros((length, total_state))
    stateTrackM[0, :] = np.zeros(total_state)
    stateTrackM[0, np.abs(th_minus) - 1] = 1

    negObsProbP = 1.0 / 5
    posObsProbP = np.exp(1) / 5
    staticObsProb = 1 - negObsProbP - posObsProbP
    survivalProbPlus = np.zeros(length)
    survivalProbPlus[0] = 1

    negObsProbM = posObsProbP
    posObsProbM = negObsProbP
    survivalProbMinus = np.zeros(length)
    survivalProbMinus[0] = 1

    for i in range(1, length):
        stateTrackP[i,:] = (negObsProbP * np.hstack((stateTrackP[i-1, 1:], 0))
                             + posObsProbP * np.hstack((0, stateTrackP[i-1, :-1]))
                             + staticObsProb * stateTrackP[i-1, :])
        # print stateTrackP[i,:]

        stateTrackM[i, :] = (negObsProbM * np.hstack((stateTrackM[i - 1, 1:], 0))
                             + posObsProbM * np.hstack((0, stateTrackM[i-1, :-1]))
                             + staticObsProb * stateTrackM[i-1, :])

        survivalProbPlus[i] = np.sum(stateTrackP[i, :])
        survivalProbMinus[i] = np.sum(stateTrackM[i, :])

    ag2_social = np.log(survivalProbPlus / survivalProbMinus)  # This is the non-decision social evidence.

    ag2_social[ag1_fire:] = ev1[ag1_fire]  # Convert non-decision evidence to threshold after the first agent decides.

    # Compute the Agent 2 belief by adding the private and bump to the non-decision social evidence.
    ev2 = np.zeros(length)
    ag2_fire = 0  # to mark when agent 2 decides
    last_social_ev = 0

    # Compute the time series by using the threshold check.
    for i in range(ev2.size):
        if ag2_fire == 0:
            ev2[i] += ag2_private[i] + last_social_ev # include the observation

            if ev2[i] >= th_plus:  # check if this was sufficient to decide
                ev2[i] = th_plus
                ag2_fire = i
            elif ev2[i] <= th_minus:
                ev2[i] = th_minus
                ag2_fire = i
            else:
                ev2[i] += ag2_social[i] - last_social_ev # include the observation of the other agent's decision
                last_social_ev = ag2_social[i]

                if ev2[i] >= th_plus:  # check if this was sufficient to decide
                    ev2[i] = th_plus
                    ag2_fire = i
                elif ev2[i] <= th_minus:
                    ev2[i] = th_minus
                    ag2_fire = i
        else:
            ev2[i] = ev2[i - 1]


    # Finally, add in a start at 0 and don't record after decisions
    ev1 = np.hstack((0, ev1))
    ag2_private = np.hstack((0,ag2_private[:ag2_fire+1]))
    ag2_social = np.hstack((0, ag2_social))
    ev2 = np.hstack((0, ev2))

    return ev1, ag2_social, ag2_private, ev2, ag2_fire


def make_discrete_plot(ag1_belief, ag2_social, ag2_private, ag2_belief, ag2_fire, bdy_plus, bdy_minus, length):
    plt.figure(figsize=(4, 8))  # Start the figure

    plt.subplot(411)  # subplot(mni) puts the next plot in the i-th location of m by n multiplot
    plt.plot(ag1_belief, 'o-')

    plt.ylabel('Evidence')
    plt.title('Agent 1 Belief')
    plt.axis([0, length, bdy_minus - 0.1, bdy_plus + 0.1])

    plt.subplot(412)

    plt.plot(ag2_social, 'o-', c='red', label='Soc')
    plt.title('Agent 2 Social Evidence')
    plt.axis([0, length, bdy_minus - 0.1, bdy_plus + 0.1])
    plt.ylabel('Evidence')

    plt.subplot(413)
    plt.plot(ag2_private, 'o-', c='purple', label='Priv')
    plt.xticks(np.arange(0,12,2))
    plt.title('Agent 2 Private Evidence')
    max_priv = np.amax(np.hstack((ag2_private, bdy_plus)))
    min_priv = np.amin(np.hstack((ag2_private, bdy_minus)))

    plt.axis([0, length, min_priv - 0.1, max_priv + 0.1])
    plt.ylabel('Evidence')

    plt.subplot(414)
    plt.plot(ag2_belief, 'o-', c='orange', label=r'$y^{(2)}$')
    plt.title('Agent 1 Belief')

    # plt.legend()
    plt.xlabel('Time', fontsize=16)
    # plt.xticks([0, N/2, N])
    # plt.yticks([0, max_ev / 2.0, max_ev])
    plt.axis([0, length, bdy_minus - 0.1, bdy_plus + 0.1])
    plt.ylabel('Evidence')
    # plt.title('Non-Decision Evidence for ' + r'$\theta_- = $' + str(negThresh) + r', $\theta_+ = $' + str(posThresh))
    plt.title('Agent 2 Belief')

    plt.tight_layout()
    plt.show()


def main():
    # Parameters for the simulation
    length = 10
    p = np.e / 7  # need the ratio of p and q to be e
    q = 1. / 7
    s = 1 - p - q  # this has it around .47
    bdy_plus = 2
    bdy_minus = -1

    ag1_belief, ag2_social, ag2_private, ag2_belief, ag2_fire = run_sim(length, p, q, s, bdy_plus, bdy_minus)
    make_discrete_plot(ag1_belief, ag2_social, ag2_private, ag2_belief, ag2_fire, bdy_plus, bdy_minus, length)

    # Parameters for the simulation
    length = 500
    p = np.e / 7  # need the ratio of p and q to be e
    q = 1. / 7
    s = 1 - p - q  # this has it around .47
    bdy_plus = 60
    bdy_minus = -3

    ag1_belief, ag2_social, ag2_private, ag2_belief, ag2_fire = run_sim(length, p, q, s, bdy_plus, bdy_minus)
    make_discrete_plot(ag1_belief, ag2_social, ag2_private, ag2_belief, ag2_fire, bdy_plus, bdy_minus, length)

if __name__ == '__main__':
    main()
