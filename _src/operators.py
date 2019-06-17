import numpy as np
from .solution import *

### INITIALIZATION METHODS ###
def init_random(lb, ub, dimension):
    return np.random.uniform(lb, ub, dimension)
    
def init_zero(lb, ub, dimension):
    return np.zeros(dimension)
    
#velocity initialization
def initv_half_dif(x, lb, ub, dimension, f=0.1):
    return (np.random.uniform(lb, ub, dimension) - x)/2
    
def initv_random(x, lb, ub, dimension, f=0.1):
    return np.random.uniform(lb, ub, dimension)
    
def initv_zero(x, lb, ub, dimension, f=0.1):
    return np.zeros(dimension)

def initv_small_random(x, lb, ub, dimension, f=0.1):
    return np.random.uniform(-f, +f, dimension)
    
### OPERATORS PSO ###
def pso_velocity(x, v, gbest, pbest, w, c1, c2):
    ''' 
    Computes the new velocity of 'x'
    :param x: np.array of real values
    :param v: np.array of real values
    :param gbest: np.array of real values
    :param pbest: np.array of real values
    :param w: (inertia) velocity modifier, real value
    :param c1: (cognitive) pbest modifier, real value
    :param c2: (social) gbest modifier, real value
    :returns: np.array of real values
    '''
    r1 = np.random.random(len(x))
    r2 = np.random.random(len(x))
    
    v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
    return v
    
def pso_move(x, v):
    ''' 
    Computes the new position of 'x'
    :param x: np.array of real values
    :param v: np.array of real values
    :returns: np.array of real values
    '''
    u = x + v
    return u

### OPERATORS DE* ###
def mut_de(x1, x2, x3, beta):
    ''' 
    Creates one new solutions by the differential mutation method
    :param x1: np.array of real values
    :param x2: np.array of real values
    :param x3: np.array of real values
    :returns: np.array of real values
    '''
    u = x1 + beta*(x2 - x3)
    return u
    
def crx_exponential(x1, x2, pr):
    ''' 
    Creates two new solutions by exchanging the points between 'x1' and 'x2', the points to be exchanged are chosen by the exponential method
    :param x1: np.array of real values
    :param x2: np.array of real values
    :returns: (np.array, np.array)
    '''
    size = len(x1)
    mask = [False]*size

    i = np.random.choice(size)
    mask[i] = True #ensures at least one point
    while pr >= np.random.uniform(0, 1) and np.sum(mask) < size:
        i = i + 1
        mask[i%size] = True

    u, v = crx_exchange_points(x1, x2, mask)

    return u, v

# exchange points
def crx_exchange_points(x1, x2, points):
    ''' 
    Creates two new solutions by exchanging the points between 'x1' and 'x2' based on the given 'points'
    :param x1: np.array of real values
    :param x2: np.array of real values
    :param points: list of indices
    :returns: (np.array, np.array)
    '''
    u = np.array([_ for _ in x1])
    v = np.array([_ for _ in x2])
    
    u[points] = x2[points]
    v[points] = x1[points]
    
    return u, v

### REPAIR OPERATOR ###
def repair_truncate(x, lb, ub):
    '''
    Replaces the values in 'x' higher than 'ub' and lower than 'lb' by 'ub' and 'lb', respectively
    :param x: np.array of real values
    :param lb: lower bound
    :param ub: upper bound
    :returns: np.array of real values
    '''
    u = np.clip(x, lb, ub)
    return u
    
def repair_random(x, lb, ub):
    '''
    Replaces the values in 'x' higher than 'ub' and lower than 'lb' by a random value between lb and ub
    :param x: np.array of real values
    :param lb: lower bound
    :param ub: upper bound
    :returns: np.array of real values
    '''
    u = np.array([xi for xi in x])
    
    mask = (u<lb) + (u>ub)
    u[mask] = np.random.uniform(lb, ub, len(u[mask]))
    return u
   
#repair velocity
def repairv_zero(v, x, x1, lb, ub):
    u = np.array([xi for xi in x])
    mask = (u<lb) + (u>ub)
    
    w = np.array([vi for vi in v])
    w[mask] = 0

    return w
    
def repairv_diff(v, x, x1, lb, ub):
    return x1-x