import numpy as np
import matplotlib.pyplot as plt
# from scipy import stats
"""
With the probability masses for LLRs given a non-decision, sequentially update the LLR of two agents until the process equillibrates.
"""

# Probability Masses
theta_minus = -1  # the lower threshold
theta_plus = 3  # the upper threshold
x_resolution = 0.05  # how finely to sample the distributions
x_locations = np.arange(theta_minus + x_resolution, theta_plus, x_resolution)  # possible x-values, w/o boundaries
length = x_locations.size
Pp = np.zeros(length)  # the probability mass for LLR in H+ state
Pm = np.zeros(length)  # for H-

mu_plus = 1
mu_minus = .9
sigma = 1

y1 = 0
y2 = 0

lower_bound = theta_minus  # this will store the known lower bound for LLR of the other agent
upper_bound = theta_plus

# Generate some example masses
for i in range(length):
    x_val = x_locations[i]
    Pp[i] = np.exp((-0.5 / sigma) * (x_val - mu_plus)**2)
    Pm[i] = np.exp((-0.5 / sigma) * (x_val - mu_minus)**2)

# Pp /= np.sum(Pp)  # normalize
# Pm /= np.sum(Pm)

# Equillibration Process
# First compute the survival probabilities.
info = 0
y1 += info
y2 += info 

# here there would be an escape check

def check_bounds(info, lower, upper):
    global upper_bound
    global lower_bound
    
    if info >= 0:
        if theta_plus - info < upper:
            upper = theta_plus - info
    if info < 0:
        if theta_minus - info < lower:
            lower = theta_minus - info
    upper_bound = upper
    lower_bound = lower

def compute_new_info():
    plus = 0
    minus = 0
    for i in range(length):
        x_val = x_locations[i]
        if x_val > lower_bound and x_val < upper_bound:
            plus += Pp[i]
            minus += Pm[i]
    info = np.log( plus / minus )
    return info        
    
for j in range(10):
    print info
    print '(' + str(lower_bound) + ', ' + str(upper_bound) + ')'
    check_bounds(info, lower_bound, upper_bound)
    info = compute_new_info()
    
# plt.plot(x_locations, Pp)
# plt.plot(x_locations, Pm)
# plt.show()