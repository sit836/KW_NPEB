import numpy as np
from scipy.stats import norm
import sys

import mosek


def _streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()


def kwp(data, grid_of_mean):
    """
    Solve Kiefer-Wolfowitz MLE in its primal form.
    :param data: 1-D observation
    :param grid_of_mean: 1-D grid of means
    :return prior: estimated prior
    :return mixture: estimated mixture density
    """
    len_grid = len(grid_of_mean)
    sz = len(data)
    location = np.subtract.outer(data, grid_of_mean)

    A_raw = norm.pdf(location, scale=1)
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

            v = np.zeros(num_var)
            task.getsolutionslice(
                mosek.soltype.itr,
                mosek.solitem.xx,
                0, num_var,
                v)

            prior = v[0:len_grid]
            mixture = v[len_grid:num_var]

            return prior, mixture


def kwd(data, grid_of_mean):
    """
    Solve Kiefer-Wolfowitz MLE in its dual form.
    :param data: 1-D observation
    :param grid_of_mean: 1-D grid of means
    :return prior: estimated prior
    :return mixture: estimated mixture density
    """
    len_grid = len(grid_of_mean)
    sz = len(data)
    location = np.subtract.outer(data, grid_of_mean)
    A = norm.pdf(location, scale=1)

    print("location: ", location)


    with mosek.Env() as env:
        env.set_Stream(mosek.streamtype.log, _streamprinter)
        with env.Task(0, 0) as task:
            task.set_Stream(mosek.streamtype.log, _streamprinter)
            # task.putdouparam(mosek.dparam.intpnt_co_tol_rel_gap, 1.0e-8)

            num_var = sz
            num_con = len_grid
            # Since the actual value of Infinity is ignored, we define it solely
            # for symbolic purposes:
            inf = 0.0

            bkc = [mosek.boundkey.ra] * num_con
            blc = [0] * num_con
            buc = [num_var] * num_con

            bkx = [mosek.boundkey.lo] * num_var
            blx = [0] * num_var
            bux = [+inf] * num_var

            task.appendvars(num_var)
            task.appendcons(num_con)

            task.putvarboundslice(0, num_var, bkx, blx, bux)
            task.putconboundslice(0, num_con, bkc, blc, buc)

            opro = [mosek.scopr.log] * num_var
            oprjo = list(range(0, num_var))
            oprfo = [-1] * num_var
            oprgo = [1] * num_var
            oprho = [0] * num_var

            asub = [list(range(len_grid))] * num_var

            for j in range(num_var):
                task.putacol(j, asub[j], A[j, :])

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

            _slc = [0.0] * num_con
            task.getsolutionslice(
                mosek.soltype.itr,
                mosek.solitem.slc,
                0, num_con,
                _slc)

            _suc = [0.0] * num_con
            task.getsolutionslice(
                mosek.soltype.itr,
                mosek.solitem.suc,
                0, num_con,
                _suc)

            prior = np.array(_suc) - np.array(_slc)
            mixture = np.matmul(A, prior)

            return prior, mixture
