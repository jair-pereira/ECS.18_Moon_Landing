import numpy as np
import subprocess
import os

ALG="FOO"
RUN=0
IRACE = False
modifier = 1

def save_history(x, objs, cons):
    GEN = 0
    while os.path.isfile(ALG+"/testrun_mop/{:03}".format(RUN)+"th_run/optimizer/interface/gen{:04}".format(GEN)+"_pop_cons_eval.txt"):
        GEN = GEN + 1
    
    f1 = ALG+"/testrun_mop/{:03}".format(RUN)+"th_run/optimizer/interface/gen{:04}".format(GEN)+"_pop_vars_eval.txt"
    f2 = ALG+"/testrun_mop/{:03}".format(RUN)+"th_run/optimizer/interface/gen{:04}".format(GEN)+"_pop_objs_eval.txt"
    f3 = ALG+"/testrun_mop/{:03}".format(RUN)+"th_run/optimizer/interface/gen{:04}".format(GEN)+"_pop_cons_eval.txt"
    
    os.makedirs(ALG+"/testrun_mop/{:03}".format(RUN)+"th_run/optimizer/interface", exist_ok=True)
    
    np.savetxt(f1, x,    delimiter='	', newline='\n')
    np.savetxt(f2, objs, delimiter='	', newline='\n')
    np.savetxt(f3, cons, delimiter='	', newline='\n')
    
    return
    
def evaluate(x):
    if os.path.isfile("pop_vars_eval.txt"):
        os.remove("pop_vars_eval.txt")
    if os.path.isfile("pop_objs_eval.txt"):
        os.remove("pop_objs_eval.txt")
    if os.path.isfile("pop_cons_eval.txt"):
        os.remove("pop_cons_eval.txt")
    #write X to a file
    np.savetxt("pop_vars_eval.txt", x, delimiter='	', newline='\n')
    
    #evaluate X
    subprocess.call(["./moon_sop", "./"])
    
    #load the result of the evaluation
    objs = np.loadtxt("pop_objs_eval.txt") #maximization of objs
    cons = np.loadtxt("pop_cons_eval.txt") #constraints: c0<0.05, c1<0.3
    
    #history
    if not IRACE:
        save_history(x, objs, cons)
    
    #penalty
    c0 = np.array([0 if x[0]<0.05 else x[0] for x in cons])
    c1 = np.array([0 if x[1]<0.3 else x[1] for x in cons])
    fit = objs + (c0+c1)*modifier
    
    return fit