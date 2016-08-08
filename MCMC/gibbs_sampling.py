# refer https://theclevermachine.wordpress.com/2012/11/05/mcmc-the-gibbs-sampler/
import numpy as np
# from scipy.ndimage.measurements import variance

# Example : GIBBS sampler for bivariate_normal
nSamples = 5000

mu = np.array([0 , 0])
rho = np.array([[0.8],[0.8]]) # ([rho_21, rho_12])
# Initialize the Gibbs sampler
propSigma = 1 # Proposal variance
minn = np.array([-3 , -3])
maxx = np.array([ 3 ,  3])

# Initialize samples
x= np.zeros((nSamples,2))
x[0,0]= 3*np.random.randn() -3
x[0,1]= 3*np.random.randn() -3


dims = np.array([1, 2])


# Run GIBBS Sampler
t= 0

while t < nSamples:
    t= t + 1
    T= np.array([[t-1, t]])
    for iD in range(1,3) : # loop over dimensions
        # Update Samples  
        nIx = dims != iD 
        # Conditional mean
        muCond = mu(iD-1) + rho(iD-1)*(x(T(id-1), nIx) - mu(nIx))
        # Conditional variance
        varCond = np.math.sqrt(1-np.math.pow(rho(iD)))
        # Draw from conditional 
        x[t, iD]= np.random.normal(muCond, varCond)
        