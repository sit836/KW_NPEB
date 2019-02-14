import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import tensorflow as tf
    import tensorflow_probability as tfp

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from kw_mle_tf import kw_dual

tfd = tfp.distributions
tfco = tf.contrib.constrained_optimization

sz = 100
data = tf.add(tfd.Normal(loc=[0], scale=[1]).sample([sz]), tfd.Normal(loc=[10], scale=[2]).sample([sz]))
weights = tf.Variable(5 * tf.ones(sz), dtype=tf.float32, name="weights")

problem = kw_dual(
    data=data,
    weights=weights,
)

n_iterations = 30000
learning_rate = 0.8
loss_stochastic = []

with tf.Session() as session:
    optimizer = tfco.MultiplicativeSwapRegretOptimizer(
        optimizer=tf.train.AdagradOptimizer(learning_rate=learning_rate))
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

    # sanity check
    grid_of_mean = np.linspace(min(data.eval()), max(data.eval()), 300)
    location = np.subtract.outer(data.eval(), grid_of_mean)
    norm_density = [norm.pdf(location[i], scale=1) for i in range(data.eval().shape[0])]

    constraints = np.matmul(np.transpose(norm_density), trained_weights).flatten()

    plt.plot(grid_of_mean, constraints, '-o')
    plt.axhline(sz, color='black', lw=2)
    plt.title('Feasibility Check')
    plt.show()

    plt.plot(loss_stochastic, '-o')
    plt.title('Loss')
    plt.show()
