import matplotlib.pyplot as plt
import mosek
import numpy as np
import sys
from scipy.stats import norm


def _streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()


def kwp(data, grid_mean):
    """
    Solve Kiefer-Wolfowitz MLE primal problem.
    :param data:
    :param grid_mean:
    :return prior: estimated prior
    :return mixture: estimated mixture density
    """
    len_grid = len(grid_mean)
    sz = len(data)
    location = np.subtract.outer(data, grid_mean)

    A_raw = np.asarray([norm.pdf(location[i], scale=1) for i in range(sz)])
    A_1 = np.concatenate((A_raw, -np.identity(sz)), axis=1)
    A_2 = np.array([1] * len_grid + [0] * sz).T
    A = np.vstack((A_1, A_2))

    with mosek.Env() as env:
        env.set_Stream(mosek.streamtype.log, _streamprinter)
        with env.Task(0, 0) as task:
            task.set_Stream(mosek.streamtype.log, _streamprinter)

            num_var = sz + len_grid
            num_con = sz + 1

            bkc = [mosek.boundkey.fx] * num_con
            blc = [0] * sz + [1]
            buc = [0] * sz + [1]

            bkx = [mosek.boundkey.ra] * num_var
            blx = [0] * num_var
            bux = [1] * num_var

            task.appendvars(num_var)
            task.appendcons(num_con)

            task.putvarboundslice(0, num_var, bkx, blx, bux)
            task.putconboundslice(0, num_con, bkc, blc, buc)

            opro = [mosek.scopr.log] * sz
            oprjo = list(range(len_grid, num_var))
            oprfo = [-1] * sz
            oprgo = [1] * sz
            oprho = [0] * sz

            asub = [list(range(A.shape[0]))] * num_var
            aval = []
            # aval[j] contains the non-zero values of column j
            for i in range(0, A.shape[1]):
                aval.append(list(A[:, i]))

            for j in range(num_var):
                task.putacol(j, asub[j], aval[j])

            oprc = [mosek.scopr.ent]
            opric = [0]
            oprjc = [0]
            oprfc = [0.0]
            oprgc = [0.0]
            oprhc = [0.0]

            task.putSCeval(opro, oprjo, oprfo, oprgo, oprho,
                           oprc, opric, oprjc, oprfc, oprgc, oprhc)

            task.optimize()

            v = [0.0] * num_var
            task.getsolutionslice(
                mosek.soltype.itr,
                mosek.solitem.xx,
                0, num_var,
                v)

            prior = v[0:len_grid]
            mixture = v[len_grid:num_var]

            return prior, mixture
