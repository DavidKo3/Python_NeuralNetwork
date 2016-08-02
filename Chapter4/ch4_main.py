# Python imports
import numpy as np  # Matrix and vector computation package
import sklearn.datasets # To generate the dataset
import matplotlib.pyplot as plt  # Plotting library
from matplotlib.colors import colorConverter, ListedColormap  # Some plotting functions
from mpl_toolkits.mplot3d import Axes3D  # 3D plots
from matplotlib import cm  # Colormaps
# # Allow matplotlib to plot inside this notebook
# %matplotlib inline
# Set the seed of the numpy random number generator so that the tutorial is reproducable
np.random.seed(seed=1)

# Generate the dataset
X, t = sklearn.datasets.make_circles(n_samples=100, shuffle=False, factor=0.3, noise=0.1)
T = np.zeros((100,2)) # Define target matrix
T[t==1,1] = 1
T[t==0,0] = 1
# Separate the red and blue points for plotting
x_red = X[t==0]
x_blue = X[t==1]

print('shape of X: {}'.format(X.shape))
print('shape of T: {}'.format(T.shape))


 # Plot both classes on the x1, x2 plane
plt.plot(x_red[:,0], x_red[:,1], 'ro', label='class red')
plt.plot(x_blue[:,0], x_blue[:,1], 'bo', label='class blue')
plt.grid()
plt.legend(loc=1)
plt.xlabel('$x_1$', fontsize=15)
plt.ylabel('$x_2$', fontsize=15)
plt.axis([-1.5, 1.5, -1.5, 1.5])
plt.title('red vs blue classes in the input space')
plt.show()



# Define the logistic function
def logistic(z):
    return 1 / (1 + np.exp(-z))

# Define the softmax function
def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

# Function to compute the hidden activations
def hidden_activations(X, Wh, bh):
    return logistic(X.dot(Wh) + bh)

# Define output layer feedforward
def output_activations(H, Wo, bo):
    return softmax(H.dot(Wo) + bo)

# Define the neural network function
def nn(X, Wh, bh, Wo, bo):
    return output_activations(hidden_activations(X, Wh, bh), Wo, bo)

# Define the neural network prediction function that only returns
#  1 or 0 depending on the predicted class
def nn_predict(X, Wh, bh, Wo, bo):
    return np.around(nn(X, Wh, bh, Wo, bo))


########## 2. Vectorization of the backward step #########################################################

# Define the cost function
def cost(Y, T):
    return - np.multiply(T, np.log(Y)).sum()

# Define the error function at the output
def error_output(Y, T):
    return Y - T

# Define the gradient function for the weight parameters at the output layer
def gradient_weight_out(H, Eo):
    return  H.T.dot(Eo)

# Define the gradient function for the bias parameters at the output layer
def gradient_bias_out(Eo):
    return  np.sum(Eo, axis=0, keepdims=True)

########## Vectorization of the hidden layer backward step #########################################################

















