# Constants
FOREST_FIRE_DATASET_PATH = './../datasets/forest-fire-damage/prepared_ff_damage_dataset.csv'
STANDARDIZED_DATASET_DELIMETER = ','
LEARNING_RATE = 1e-6
MAX_EPOCHS = 10000

# Imports
import math
import numpy as np
import tensorflow as tf
import csv

# Declaring variables

ff_dataset = None # numpy matrix for forest fire design matrix + output data

X = np.array([]) # Input design matrix
y = np.array([]) # Output vector
m = 0 # No. of training cases
n = 0 # No. of features

# Data Preparation

def import_data(): # Import forest fire CSV data and store into a global variable
	global ff_dataset
	dataset_reader = csv.reader(open(FOREST_FIRE_DATASET_PATH, "rb"), delimiter=STANDARDIZED_DATASET_DELIMETER) # Import CSV into raw data
	next(dataset_reader, None) # Skip column headers
	raw_dataset = list(dataset_reader) # Put dataset into list type from csv reader
	ff_dataset = np.array(raw_dataset).astype(np.float32) # Convert dataset to a numpy array
	prepare_data()

def prepare_data():
	global X, y, m, n
	(m, n) = ff_dataset.shape
	n = n - 1 # Discounting output in imported dataset
	# Separating input features and output from dataset
	X = ff_dataset[:, :-1].reshape((m, n))
	y = ff_dataset[:, n].reshape((m, 1))
	y = np.log(np.add(y, np.ones(y.shape))) # Apply a transformation to the output vector of ln(x + 1) to linearize it due to data's skew to (0, 0)

# Learning

# Declare the numerical variables for our weight(s) and bias
W = 0
b = 0

def learn(): # Train a model from data using TensorFlow
	# Declare the TensorFlow versions of the weight(s) and bias variables and placeholders for input/output
	X_tf = tf.placeholder("float32")
	y_tf = tf.placeholder("float32")
	W_tf = tf.Variable(tf.zeros([n, 1]), name="weight")
	b_tf = tf.Variable(0.0, name="bias")

	activation = tf.add(tf.matmul(X_tf, W_tf), b_tf) # Compute a linear activation with model: X*W + b. This assumes X is an m x n matrix and W is an n x 1 vector (producing an m x 1 vector).
	cost_function = tf.div(tf.reduce_sum(tf.square(tf.sub(activation, y_tf))), (2 * m)) # Setup the cost function to be standardized L2 loss (mean squared error of activation vs. actual y)
	optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost_function) # Setup the cost optimization method as iterative gradient descent

	with tf.Session() as sess: # Create the session
		sess.run(tf.initialize_all_variables()) # Initialize the placeholder values

		for i in range(MAX_EPOCHS): # Perform MAX_EPOCHS steps of optimization
			sess.run(optimizer, feed_dict={X_tf: X, y_tf: y}) # Perform a single weight update

			if i % 1000 == 0:
				print 'Step #%d: %f' % (i, sess.run(cost_function, feed_dict={X_tf: X, y_tf: y}))

		# Initialize our numerical weights from the TensorFlow session versions
		global W, b
		W = sess.run(W_tf)
		b = sess.run(b_tf)

# Prediction

def predict(x): # Take a feature vector x and compute a prediction based on the weights
	return math.exp(np.add(np.dot(np.transpose(W), x), b)) - 1 # First, we use the linear hypothesis function h(x) = x^Transpose*W + b, and then use that as input to the function g(x) = e^x - 1 as the inverse of the pre-processing ln(x + 1) step

# Post-Training

def compute_cost(): # Computes the universal cost over the training set
	cost = 0
	for i in range(X.shape[0]): # Iterate through all the training cases
		cost += np.square(y[i] - predict(X[i, :])) # Get the square of discrepency between predicted and actual
	cost /= (2 * m) # Get mean of cost
	return cost

def output_results(): # Output optimization results through the console
	print 'Optimization complete'
	# print 'Final weight vector: \n', W
	# print 'Final bias value', b
	print 'Final cost is ', compute_cost()
	# for i in range(X.shape[0]):
	# 	print predict(X[i, :])

if __name__ == "__main__":
	import_data()
	learn()
	output_results()
