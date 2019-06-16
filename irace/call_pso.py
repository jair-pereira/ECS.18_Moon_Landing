import numpy as np
import sys, argparse, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solvers import pso
import utils

# ## interface between the target-runner and the solvers for irace ###

def main(args):
    parser = argparse.ArgumentParser()
    #parser.add_argument('--nfe'  , dest='nfe'  , type=float, help="Integer   : Number of Function Evaluations")
    parser.add_argument('--n'    , dest='n'    , type=float, help="Integer   : Population size")
    parser.add_argument('--w'    , dest='w'    , type=float, help="Real value: velocity modifier")
    parser.add_argument('--c1'   , dest='c1'   , type=float, help="Real value: pbest modifier")
    parser.add_argument('--c2'   , dest='c2'   , type=float, help="Real value: gbest modifier")
    parser.add_argument('--mod'   , dest='mod'   , type=float, help="Constraind penalty")
    args = parser.parse_args()
    
    utils.IRACE = True
    utils.ALG = "PSO"
    utils.modifier = args.mod
    sol = pso(int(args.n), utils.evaluate, (0, 1), 2, 1.2e+3, args.w, args.c1, args.c2)
    
    print(sol.best.getFitness())
    return 

if __name__ == "__main__":
   np.warnings.filterwarnings('ignore')
   main(sys.argv[1:])
