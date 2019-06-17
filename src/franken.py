import numpy as np
import src
import utils
import datetime

def frk01():
    utils.ALG = "FRK01"
    utils.IRACE = False
    utils.modifier = 0.25
    n = 100
    iteration = 150
    my_func = utils.evaluate
    dimension = 2
    bounds = 0, 1
    beta = 0.00
    pr = 0.50
    w = 0.50
    c1 = 2.00
    c2 = 2.00
    pa = 0.10
    dp = 0.25

    print("0 /",iteration,":",datetime.datetime.now())

    X = np.array([src.solution(my_func, dimension, bounds) for i in range(n)])
    [Xi.initRandom() for Xi in X]
    f = my_func([Xi.x for Xi in X])
    [X[i].setFitness(f[i]) for i in range(n)]

    for it in range(iteration):
        print(it,"/",iteration,":",datetime.datetime.now())

        U = src.op.op_pso(X, src.op.select_random, src.op.mut_pso, src.op.crx_exponential)
        f = my_func([Ui.x for Ui in U])
        [U[i].setFitness(f[i]) for i in range(n)]
        X = U
        U = src.op.op_de(X, src.op.select_random, src.op.mut_cs, src.op.crx_blend)
        f = my_func([Ui.x for Ui in U])
        [U[i].setFitness(f[i]) for i in range(n)]
        X = U

    print(iteration,"/",iteration,":",datetime.datetime.now())
    return #src.solution.best.fitness


def frk02():
    utils.ALG = "FRK02"
    utils.IRACE = False
    utils.modifier = 0.00
    n = 100
    iteration = 150
    my_func = utils.evaluate
    dimension = 2
    bounds = 0, 1
    beta = 0.00
    pr = 0.50
    w = 2.00
    c1 = 0.50
    c2 = 2.00
    pa = 0.75
    dp = 0.75

    print("0 /",iteration,":",datetime.datetime.now())

    X = np.array([src.solution(my_func, dimension, bounds) for i in range(n)])
    [Xi.initRandom() for Xi in X]
    f = my_func([Xi.x for Xi in X])
    [X[i].setFitness(f[i]) for i in range(n)]
    for it in range(iteration):
        print(it,"/",iteration,":",datetime.datetime.now())
        U = src.op.op_pso(X, src.op.select_random, src.op.mut_pso, src.op.crx_exponential)
        f = my_func([Ui.x for Ui in U])
        [U[i].setFitness(f[i]) for i in range(n)]
        X = U
        U = src.op.op_de(X, src.op.select_random, src.op.mut_pso, src.op.crx_exponential)
        f = my_func([Ui.x for Ui in U])
        [U[i].setFitness(f[i]) for i in range(n)]
        X = src.op.replace_if_best(X, U)


    print(iteration,"/",iteration,":",datetime.datetime.now())
    return

def frk03():
    utils.ALG = "FRK03"
    utils.IRACE = True
    utils.modifier = 2.00
    n = 100
    iteration = 150
    my_func = utils.evaluate
    dimension = 2
    bounds = 0, 1
    beta = 0.75
    pr = 0.75
    w = 1.00
    c1 = 2.00
    c2 = 0.75
    pa = 0.10
    dp = 0.25

    print("0 /",iteration,":",datetime.datetime.now())

    X = np.array([src.solution(my_func, dimension, bounds) for i in range(n)])
    [Xi.initRandom() for Xi in X]
    f = my_func([Xi.x for Xi in X])
    [X[i].setFitness(f[i]) for i in range(n)]
    for it in range(iteration):
        print(it,"/",iteration,":",datetime.datetime.now())
        U = src.op.op_pso(X, src.op.select_random, src.op.mut_pso, src.op.crx_blend)
        f = my_func([Ui.x for Ui in U])
        [U[i].setFitness(f[i]) for i in range(n)]
        X = U
        U = src.op.op_pso(X, src.op.select_random, src.op.mut_pso, src.op.crx_blend)
        f = my_func([Ui.x for Ui in U])
        [U[i].setFitness(f[i]) for i in range(n)]
        X = src.op.replace_if_best(X, U)

    print(iteration,"/",iteration,":",datetime.datetime.now())
    return
