import tensorflow as tf
import numpy as np

class LinearRegression(object):
    def fit(self, X, y, learning_rate = 1e-6, epochs=10000):
        m, n = X.shape

        X_tf = tf.placeholder('float32')
    	y_tf = tf.placeholder('float32')
    	W_tf = tf.Variable(tf.zeros([n, 1]), name='weights')
    	b_tf = tf.Variable(0.0, name='bias')

        # Compute a linear activation with model: X*W + b. This assumes X is an
        # m x n matrix and W is an n x 1 vector (producing an m x 1 vector).
        activation = tf.add(tf.matmul(X_tf, W_tf), b_tf)

        # Setup the cost function to be standardized L2 loss (mean squared error
        # of activation vs. actual y)
        squared_error = tf.square(tf.sub(activation, y_tf))
    	cost_function = tf.div(tf.reduce_sum(squared_error), (2 * m))

        # Setup the cost optimization method as iterative gradient descent
        gradient_descent = tf.train.GradientDescentOptimizer(learning_rate)
    	optimizer = gradient_descent.minimize(cost_function)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            for i in range(epochs):
                sess.run(optimizer, feed_dict={X_tf: X, y_tf: y})

                if i % 100 == 0:
                    cost = sess.run(cost_function, feed_dict={X_tf: X, y_tf: y})
                    print 'Step #%d: %f' % (i, cost)

            self.W, self.b = sess.run([W_tf, b_tf])

    def predict(self, X):
        return np.dot(X, self.W) + self.b
