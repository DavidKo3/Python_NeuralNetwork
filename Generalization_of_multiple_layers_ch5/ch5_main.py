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
                 
                 
                                                   