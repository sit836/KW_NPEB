import tensorflow as tf
import tensorflow_probability as tfp

tfco = tf.contrib.constrained_optimization
tfd = tfp.distributions


class kw_dual(tfco.ConstrainedMinimizationProblem):
    def __init__(self, data, weights, len_grid=300):
        self.data = data
        self.weights = weights
        self.len_grid = len_grid
        self.sz = tf.to_float(tf.shape(data)[0])

    @property
    def objective(self):
        return (-1) * tf.reduce_sum(tf.log(self.weights))

    @property
    def constraints(self):
        grid_of_mean = tf.lin_space(tf.reduce_min(self.data), tf.reduce_max(self.data), self.len_grid)
        location = self.data - grid_of_mean
        dist = tfd.Normal(loc=location, scale=1)
        normal_density = dist.prob([1])
        return tf.tensordot(tf.transpose(self.weights), normal_density, axes=1) - self.sz * tf.ones(self.len_grid)
