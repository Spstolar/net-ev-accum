import numpy as np
import matplotlib.pyplot as plt

"""
Adapted from simddmfp.m
Simulates the Fokker-Planck equation for drift diffusion.
Outputs:
    1. The probability mass for the evidence accumulation process: Pp and Pm
    2. The survival probability for a single agent.
    3. The escape probability for some agent out of a group.
    4. The positive and negative path probabilities and the information gained when you know an agent has positive LLR.
"""

# model params
g = 1  # drift
s = 1  # variance
hp = 3  # upper threshold
hm = -3.0  # lower threshold
L = hp-hm  # length of the domain

# discretization parameters
dx = 0.01
Nx = int(L / dx + 1)
xvec = np.linspace(hm, hp, Nx)  # split the x-axis into dx sized bins
T = 10  # total time to run
dt = 0.001 
Nt = int(T / dt + 1)
tvec = np.linspace(0, T, Nt)  # split time into dt sized bins

# setup probability and survival arrays
Pp = np.zeros((Nx, Nt))  # Prob mass under assumption of rightward drift.
zero_ind = np.argmin(abs(xvec))  # grab the index of the zero or the x-value closest to zero
Pp[zero_ind, 0] = 1.0 / dx  # we want probability to be concentrated at the zero cell, so it should have area 1
surv_prob_P = np.zeros(Nt)
surv_prob_P[0] = 1
pos_surv_pos = np.zeros(Nt)  # probability mass above zero but not escaping given H^+
pos_surv_neg = np.zeros(Nt)  # ^^ for H^-

# Crank-Nicolson discretization of diffusion
scaling_term = (s * s) / (2 * dx * dx)
D = scaling_term * (np.eye(Nx, k=1) - 2 * np.eye(Nx, k=0) + np.eye(Nx, k=-1))  # scalar * (diagonal -2, off diagonals 1)

# discretization of advection for plus and minus drift
Ap = (-g / dx) * (np.eye(Nx, k=1) - np.eye(Nx, k=0))

# create Crank-Nicolson step matrices
diff_inverse = np.linalg.inv(np.eye(Nx) - (dt/2)*D)
Mp = np.dot(diff_inverse, (np.eye(Nx) + dt*Ap + (dt / 2)*D))

# run CN steps
for j in range(1, Nt):  # j from 1 to Nt-1
    Pp[:, j] = np.dot(Mp, Pp[:, j-1])  # probability
    surv_prob_P[j] = dx * np.sum(Pp[:, j])  # survival
    right_pos = dx * np.sum(Pp[zero_ind:, j])
    right_neg = dx * np.sum(Pp[:zero_ind, j])
    pos_surv_pos[j] = right_pos / surv_prob_P[j]
    pos_surv_neg[j] = right_neg / surv_prob_P[j]


np.save('surv_P.npy', Pp)

plt.figure(1)
plt.plot(tvec, surv_prob_P, color='red', linewidth=8)  # survival prob
plt.axis([0, T, 0, 1])
plt.title('Survival Probability')

plt.figure(2)

for num_agents in np.arange(1,100,10):
    exit_prob = 1 - surv_prob_P ** num_agents
    plt.plot(tvec, exit_prob, linewidth=2)  # survival prob
plt.axis([0, T, 0, 1])
plt.title('First Exit Probability')

plt.figure(3)
plt.plot(tvec, pos_surv_pos, color='orange', linewidth=2, label='R+')
plt.plot(tvec, pos_surv_neg, color='blue', linewidth=2, label='R-')
r_prob_ratio = np.log(pos_surv_pos / pos_surv_neg)  # log R_+ / R_-  clique info
r_prob_ratio[0:20] = 0.3

plt.plot(tvec, r_prob_ratio, color='purple', linewidth=4, label='Info')
plt.legend()

plt.show()
