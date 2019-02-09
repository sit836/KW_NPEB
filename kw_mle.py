import numpy as np
from scipy.stats import norm
import sys

import mosek


def _streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()


class KWMLE:
    def __init__(self, df, len_grid=500):
        self.df = df
        self.grid_of_mean = np.linspace(min(df), max(df), len_grid)

        location = np.subtract.outer(self.df, self.grid_of_mean)
        self.norm_density = norm.pdf(location, scale=1)

    def kw_primal(self):
        """
        Solve Kiefer-Wolfowitz MLE in its primal form.
        :return prior: estimated prior
        :return mixture: estimated mixture density
        """
        len_grid = len(self.grid_of_mean)
        sz = len(self.df)

        A_1 = np.concatenate((self.norm_density, -np.identity(sz)), axis=1)
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

                self.prior = v[0:len_grid]
                self.mixture = v[len_grid:num_var]
                return self.prior, self.mixture

    def kw_dual(self):
        """
        Solve Kiefer-Wolfowitz MLE in its dual form.
        :return prior: estimated prior
        :return mixture: estimated mixture density
        """
        len_grid = len(self.grid_of_mean)
        sz = len(self.df)

        with mosek.Env() as env:
            env.set_Stream(mosek.streamtype.log, _streamprinter)
            with env.Task(0, 0) as task:
                task.set_Stream(mosek.streamtype.log, _streamprinter)

                num_var = sz
                num_con = len_grid
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
                    task.putacol(j, asub[j], self.norm_density[j, :])

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

                self.prior = np.array(_suc) - np.array(_slc)
                self.mixture = np.matmul(self.norm_density, self.prior)
                return self.prior, self.mixture

    def prediction(self, df):
        """
        Compute the posterior mean.
        :param df: 1-D dataframe
        :return: the posterior mean
        """
        location = np.subtract.outer(df, self.grid_of_mean)
        norm_density = norm.pdf(location, scale=1)
        weighted_support = self.grid_of_mean * self.prior
        mixture = np.matmul(norm_density, self.prior)
        return np.matmul(norm_density, weighted_support) / mixture
