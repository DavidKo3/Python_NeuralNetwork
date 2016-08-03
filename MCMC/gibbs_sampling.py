import numpy as np
from scipy.ndimage.measurements import variance

# Example : GIBBS sampler for bivariate_normal
nSamples = 5000

mu = np.array([0 , 0])
rho = np.array([[0.8],[0.8]])

# Initialize the Gibbs sampler
propSigma = 1 # Proposal variance
minn = np.array([-3 , -3])
maxx = np.array([3  ,  3])

# Run GIBBS Sampler
t= 1

while t < nSamples:
    t= t + 1
    T= np.array([[t-1, t]])
    for iD in range(1,3) : # loop over dimensions
        # Update Samples  
        nIx = iD
        # Conditional mean
        muCond = mu(iD) + rho(iD)*(x(T))