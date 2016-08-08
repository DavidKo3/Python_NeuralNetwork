# refer https://theclevermachine.wordpress.com/2012/11/05/mcmc-the-gibbs-sampler/
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
# from scipy.ndimage.measurements import variance

# Example : GIBBS sampler for bivariate_normal
nSamples = 5000

mu = np.array([0 , 0 ])
rho = np.array([0.8, 0.8]) # ([rho_21, rho_12])
print mu.shape , rho.shape
# Initialize the Gibbs sampler
propSigma = 1 # Proposal variance
minn = np.array([-3 , -3])
maxx = np.array([ 3 ,  3])

# Initialize samples
x= np.zeros((nSamples,2))
x[0,0]= 3*np.random.randn() -3
x[0,1]= 3*np.random.randn() -3

# print x
# print x[0,0], x[0,1]
dims = np.array([1, 2])


# Run GIBBS Sampler
t= 0
nSamples= 5000

print np.random.normal(0.0, 1.0 )
while t < nSamples:
    
    T= np.array([t-1, t])
    for iD in range(1,3) : # loop over dimensions
        # Update Samples  
        nIx =0
        if iD ==1:
            nIx = 1
        else:
            nIx = 0
         
        # Conditional mean
        muCond = mu[iD-1] + rho[iD-1]*( x[T[iD-1], nIx] - mu[nIx] )
        
        # Conditional variance
        varCond = np.sqrt(1 - np.power(rho[iD-1],2) )
        # Draw from conditional 
        x[t, iD-1]= np.random.normal(muCond, varCond)
        print x
    t= t + 1    
    
# Display sampling dynamics
fig= plt.figure() 
ax1 = fig.add_subplot(111)

ax1.scatter(x[:,0], x[:,1], color='blue', s=5, edgecolor='none')
ax1.grid(True)
ax1.set_ylim([-11,11])
ax1.set_xlim([-11,11])
plt.show()