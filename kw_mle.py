import mosek
import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack
from scipy.stats import norm


class KWMLE:
    """
    Solve 1-D Kiefer-Wolfowitz MLE by interior point methods as studied in Koenker and Mizera (2014).
    For reference, see: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.679.9137&rep=rep1&type=pdf
    """
    def __init__(self, data, stds, len_grid=500):
        self.data = data
        self.check_dtype()
        self.grid_of_mean = np.linspace(min(data), max(data), len_grid)

        location = np.subtract.outer(self.data, self.grid_of_mean)
        self.norm_density = csr_matrix([norm.pdf(location[i], scale=stds[i]) for i in range(len(data))])

    def check_dtype(self):
        if isinstance(self.data, (list, np.ndarray)):
            if isinstance(self.data, np.ndarray):
                if self.data.ndim > 1:
                    raise ValueError(f"Data must be 1-D, but got {len(self.data)}-D.")
        else:
            raise ValueError("Data must be 1-D list or np.array.")

    def fit(self):
        """
        Solve Kiefer-Wolfowitz MLE in its dual form.
        """
        len_grid = len(self.grid_of_mean)
        sz = len(self.data)

        with mosek.Env() as env:
            with env.Task(0, 0) as task:
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
                    task.putacol(j, asub[j], self.norm_density.getrow(j).toarray().flatten())

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
                self.mixture = self.norm_density.dot(self.prior)

    def prediction(self, data, stds):
        """
        Compute the posterior mean.
        """
        location = np.subtract.outer(data, self.grid_of_mean)
        norm_density = np.array([norm.pdf(location[i], scale=stds[i]) for i in range(len(data))])
        weighted_support = self.grid_of_mean * self.prior
        mixture = np.matmul(norm_density, self.prior)
        return np.matmul(norm_density, weighted_support) / mixture
