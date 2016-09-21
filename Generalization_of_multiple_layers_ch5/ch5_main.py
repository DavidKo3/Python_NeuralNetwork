# refer http://peterroelants.github.io/posts/neural_network_implementation_part05/

##  Generalization of multiple layers ############################################################

# Python imports
import numpy as np # Matrix and vector computation package
import matplotlib.pyplot as plt  # Plotting library

# Set the seed of the numpy random number generator so that the tutorial is reproducable
np.random.seed(seed=1)
from sklearn import datasets, cross_validation, metrics # data and evaluation utils
from matplotlib.colors import colorConverter, ListedColormap # some plotting functions
import itertools
import collections


# load the data from scikit-learn
digits = datasets.load_digits()

# Load the targets
# Note that the targets are stored as digits, thest need to be
# converted to one-hot-encoding for the output softmax layer
T = np.zeros((digits.target.shape[0], 10))
print "len_of_T :", len(T)
T[np.arange(len(T)), digits.target] +=1

##############################################################################
ex=np.zeros((5,5))
col=np.array([0,1,4,3,2])
print col
ex[np.arange(len(ex)), col]+=1
print ex


##############################################################################
# Divie the data into a train and test set
X_train, X_test, T_train, T_test = cross_validation.train_test_split(digits.data, T, test_size=0.4)

print "X_train\n"
print X_train
                 
# Defome the non-linear functions used
def logistic(z):
    return 1/(1+np.exp(-z))

def logistic_deriv(y):
    # Derivative of logistic function
    return np.multiply(y, (1-y))

def softmax(z):
    return np.exp(z).np.sum(np.exp(z) , axis=1, keepdims=True)

# Define tha layers used in this model
class Layer(object):
    """ Base class for the different layers.
    Defines base methods and documentation of methods."""
    def get_parmas_iter(self):
        """ Return an iterator over the parameters(if any)
          The iterator has the same order as get_params_grad.
          The elements returned by iterator are editable in-place."""
        return []
    def get_params_grad(self, X, output_grad):
        """Return a list of gradients over ther parameters.
            The list has the same order as the get_params_iter iterator.
            X is the input.
            output_grad is the gradient at the output of this layer.
          """
        return []
    def get_input_grad(self, Y, output_grad=None,T=None):   
        """Return the gradient at the inputs of this layer.
            Y is the pre-computed output of this layer ( not needed in this case)
            output-grad is the gradient at the output of this layer
            (gradient at input of next layer)
            Output layer uses targets T to compute the gradient based on the
            output error instead of output_grad"""
            
    
    
    
    
    
    
    
    
    
    
    
    
    











       
                                                