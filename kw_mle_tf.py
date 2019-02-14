import warnings

import numpy as np

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
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


sz = 100
tfd = tfp.distributions
data = tf.add(tfd.Normal(loc=[0], scale=[1]).sample([sz]), tfd.Normal(loc=[10], scale=[2]).sample([sz]))

weights = tf.Variable(5 * tf.ones(sz), dtype=tf.float32, name="weights")

problem = kw_dual(
    data=data,
    weights=weights,
)

n_iterations = 30000
loss_stochastic = []

# global_step = tf.Variable(0, trainable=False)
# starter_learning_rate = 1.0
# learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
#                                            1000, 0.95, staircase=True)
learning_rate = 0.8

with tf.Session() as session:
    optimizer = tfco.MultiplicativeSwapRegretOptimizer(
        optimizer=tf.train.AdagradOptimizer(learning_rate=learning_rate))
    # train_op = optimizer.minimize(problem, global_step=global_step)
    train_op = optimizer.minimize(problem)

    session.run(tf.global_variables_initializer())
    for step in range(n_iterations):
        session.run(train_op)

        if (step + 1) % 200 == 0:
            loss_stochastic.append(problem.objective.eval())
            print("Iteration ", str(step + 1))

            if len(loss_stochastic) > 10:
                if (max(abs(problem.constraints.eval())) < 1) | (np.std(loss_stochastic[-5:]) < 0.5):
                    break

    trained_weights = session.run((weights))
    print("trained_weights: \n", trained_weights)

    #
    #
    #
    import numpy as np
    from scipy.stats import norm

    grid_of_mean = np.linspace(min(data.eval()), max(data.eval()), 300)
    location = np.subtract.outer(data.eval(), grid_of_mean)
    norm_density = [norm.pdf(location[i], scale=1) for i in range(data.eval().shape[0])]

    constraints = np.matmul(np.transpose(norm_density), trained_weights).flatten()
    print("max constraints: ", max(constraints))

    import matplotlib.pyplot as plt

    plt.plot(grid_of_mean, constraints, '-o')
    plt.axhline(sz, color='black', lw=2)
    plt.show()

    plt.plot(loss_stochastic, '-o')
    plt.title('Loss')
    plt.show()
