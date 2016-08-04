# Python imports
import numpy as np  # Matrix and vector computation package
import sklearn.datasets # To generate the dataset
import matplotlib.pyplot as plt  # Plotting library
from matplotlib.colors import colorConverter, ListedColormap  # Some plotting functions
from mpl_toolkits.mplot3d import Axes3D  # 3D plots
from matplotlib import cm  # Colormaps
from mpmath import eps
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


# Define the error function at the hidden layer
def error_hidden(H, Wo, Eo):
    # H*(1-H)*(E.Wo^T)
    return np.multiply(np.multiply(H, (1-H)), Eo.dot(Wo.T))

# Define the gradient function for the weight parameters at the hidden layer
def gradient_weight_hidden(X, Eh):
    return X.T.dot(Eh)

# Define the gradient function for the bias parameters at the output layer
def gradient_bias_hidden(Eh):
    return np.sum(Eh, axis=0, heepdims=True )



################# Gradient checking ###################################################################################
init_var = 1
# Initialize hidden layer parameters
bh = np.random.randn(1, 3)*init_var
Wh = np.random.randn(2, 3)*init_var
# Initialize output layer parameters
bo = np.random.randn(1, 2)*init_var
Wo = np.random.randn(3, 2)*init_var

# Compute the gradients by backprogation
# Compute the activations of the layers
H = hidden_activations(X, Wh, bh)
Y = output_activations(H, Wo, bo)

# Compute the gradients of the output layer
Eo  = error_output(Y, T)
JWo = gradient_weight_hidden(H, Eo)
Jbo = gradient_bias_out(Eo)
# Compute the gradients of the hidden layer
Eh  = error_hidden(H, Wo, Eo)
JWh = gradient_weight_hidden(X, Eh)
Jbh = gradient_bias_hidden(Eh) 

# Combine all parameter matrices in a list
params = [Wh, bh, Wo, bo]
# Combine all parameter gradients in a list
grad_params = [JWh, Jbh, JWo, Jbo]

# Set the small change to compute the numerical gradient
eps = 0.0001

# Check each parameter matrix
for p_idx in range(len(params)):
    # Check each parameter in each parameter matrix
    for row in range(params[p_idx].shape[0]):
        for col in range(params[p_idx].shape[1]):
            # Copy the parameter matrix and change the current parameter slightly
            p_matrix_min = params[p_idx].copy()
            p_matrix_min[row, col] -= eps
            p_matrix_plus = params[p_idx].copy()
            p_matrix_plus[row, col] += eps
            # Copy the parameter list, and change the updated parameter matrix
            params_min = params[:]
            params_min[p_idx] = p_matrix_min
            params_plus = params[:]
            params_plus[p_idx] = p_matrix_plus
            # Compute the numerical gradient
            grad_num = (cost(nn(X, *params_plus), T) - cost(nn(X, *params_min), T))/(2*eps)
            # Compute error if the numerical grade is not close to the backprop gradient
            if not np.isclose(grad_num, grad_params[p_idx][row,col]):
                raise ValueError('Numerical gradient of {:.6f} is not close to the backpropagation gradient of {:.6f}!'.format(float(grad_num), float(grad_params[p_idx][row,col])))
print('No gradient errors found')

dfdf


















