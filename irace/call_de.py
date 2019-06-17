import numpy as np
import sys, argparse, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solvers import de
import utils

### interface between the target-runner and the solvers for irace ###

def main(args):
    parser = argparse.ArgumentParser()
    #parser.add_argument('--nfe'  , dest='nfe'  , type=float, help="Integer   : Number of Function Evaluations")
    parser.add_argument('--n'    , dest='n'    , type=float, help="Integer   : Population size")
    parser.add_argument('--beta' , dest='beta' , type=float, help="Real value: DE-mutation modifier")
    parser.add_argument('--pr'   , dest='pr'   , type=float, help="Real value: Crossover probability")
    parser.add_argument('--mod'   , dest='mod'   , type=float, help="Constraind penalty")
    args = parser.parse_args()
    
    utils.IRACE = True
    utils.ALG = "DE"
    utils.modifier = args.mod
    sol = de(int(args.n), utils.evaluate, (0, 1), 2, 1.2e+3, args.beta, args.pr)
    
    print(sol.best.getFitness())
    return 

if __name__ == "__main__":
   np.warnings.filterwarnings('ignore')
   main(sys.argv[1:])