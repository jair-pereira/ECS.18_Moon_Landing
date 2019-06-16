from src import *
import numpy as np
import datetime

def pso(n, my_func, bounds, dimension, max_nfe, w, c1, c2):
    print("0 /",max_nfe,":",datetime.datetime.now())
    ## set problem ##
    Solution.setProblem(my_func, bounds, dimension, maximize=False)
    
    ## set repair methods ##
    Solution.repair_x = op.repair_truncate
    Solution.repair_v = op.repairv_zero
    
    ## initialize position ##
    X = Solution.initialize(n)
    [Xi.setX(op.init_random(*Solution.bounds, Solution.dimension)) for Xi in X]
    ## initialize velocity ##
    [Xi.setVelocity(op.initv_zero(Xi.x, *Solution.bounds, Solution.dimension)) for Xi in X]
    ## evaluate ##
    f = my_func([Xi.x for Xi in X])
    [X[i].setFitness(f[i]) for i in range(n)]
    
    while Solution.nfe < max_nfe:
        print(Solution.nfe,"/",max_nfe,":",datetime.datetime.now())
        ## update velocity ##
        [Xi.setVelocity(op.pso_velocity(Xi.x, Xi.velocity, Solution.best.x, Xi.pbest['x'], w, c1, c2))  for Xi in X]
        ## update position ##
        [Xi.setX(op.pso_move(Xi.x, Xi.velocity))  for Xi in X]
        ## evaluate ##
        f = my_func([Xi.x for Xi in X])
        [X[i].setFitness(f[i]) for i in range(n)]
    
    print(Solution.nfe,"/",max_nfe,":",datetime.datetime.now())
    return Solution
    
def de(n, my_func, bounds, dimension, max_nfe, beta, pr):
    print("0 /",max_nfe,":",datetime.datetime.now())
    ## set problem ##
    Solution.setProblem(my_func, bounds, dimension, maximize=False)
    
    ## set repair methods ##
    Solution.repair_x = op.repair_truncate
    
    ## initialize position ##
    X = Solution.initialize(n)
    U = Solution.initialize(n)
    [Xi.setX(op.init_random(*Solution.bounds, Solution.dimension)) for Xi in X]
    
    ## evaluate ##
    f = my_func([Xi.x for Xi in X])
    [X[i].setFitness(f[i]) for i in range(n)]
    
    ## aux ##
    parents = [i for i in range(0, n)]
    
    while Solution.nfe < max_nfe:
        print(Solution.nfe,"/",max_nfe,":",datetime.datetime.now())
        
        ## mutation de##
        selection = [np.random.choice(np.delete(parents,i), 3, replace=False) for i in range(n)]
        u = np.array([op.mut_de(X[s[0]].x, X[s[1]].x, X[s[2]].x, beta) for s in selection])
        
        ## crossover exponential##
        [U[i].setX(op.crx_exponential(X[i].x, u[i], pr)[0]) for i in range(n)]
                
        ## evaluate ##
        f = my_func([Ui.x for Ui in U])
        [U[i].setFitness(f[i]) for i in range(n)]
        
        ## replacement ##
        X = [U[i] if U[i].getFitness() > X[i].getFitness() else X[i] for i in range(n)]
    
    print(Solution.nfe,"/",max_nfe,":",datetime.datetime.now())
    return Solution