import numpy as np
import matplotlib.pyplot as plt

sims = 10
marg_bigger = 0

for i in range(sims):
    p = np.random.uniform()
    q = 1 - p
    a1 = np.random.uniform()
    b1 = np.random.uniform(0.0, 1 - a1)
    c1 = (1 - a1) - b1
    # print a1, b1, c1
    # print a1 + b1 + c1
    a2 = np.random.uniform()
    b2 = np.random.uniform(0.0, 1 - a2)
    c2 = (1 - a2) - b2
    non_marg = np.log(p / q)
    marg = np.log( (p * a1) / (q * a2))
    marg += np.log( (p * b1) / (q * b2))
    marg += np.log( (p * c1) / (q * c2))
#    print non_marg, marg
    if (np.abs(non_marg) <= np.abs(marg)):
        marg_bigger += 1
    else:
        print p, q
        print a1, b1, c1
        print a2, b2, c2

print float(marg_bigger) / sims
