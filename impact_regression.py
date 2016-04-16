# Constants
FOREST_FIRE_DATASET_PATH = 'prepared_ff_dataset.csv'
STANDARDIZED_DATASET_DELIMETER = ','
LEARNING_RATE = 0.000001
TRAINING_EPOCHS = 1000

# Imports
import numpy as np
import tensorflow as tf
import csv

ff_dataset = None # numpy matrix for forest fire design matrix + output data

X = np.array([]) # input design matrix
y = np.array([]) # output vector
m = 0 # no. of training cases
n = 0 # no. of features

# Data Preparation

def import_data(): # import forest fire CSV data and store into a global variable
	global ff_dataset
	dataset_reader = csv.reader(open(FOREST_FIRE_DATASET_PATH, "rb"), delimiter=STANDARDIZED_DATASET_DELIMETER) # import CSV into raw data
	next(dataset_reader, None) # skip headers
	raw_dataset = list(dataset_reader) # put dataset into list type from csv reader
	ff_dataset = np.array(raw_dataset).astype(np.float32) # convert dataset to a numpy array
	prepare_data()

def prepare_data():
	global X, y, m, n
	(m, n) = ff_dataset.shape
	n = n - 1 # discounting output in imported dataset
	# separating input features and output from dataset
	X = ff_dataset[:, :-1].reshape((m, n))
	y = ff_dataset[:, n].reshape((m, 1))

# Learning

def learn(): # create a linear model from data
	# Create the placeholder data for the input design matrix X, output vector y, mapping weights W, and constant bias term b
	W = tf.Variable(tf.zeros([n, 1]), name="weights")
	b = tf.Variable(tf.constant(0.0, shape=[m, 1]), name="bias")
	activation = tf.add(tf.matmul(X, W), b) # Compute the linear activation of predicted_y = activation(X, W, b) = X*W + b
	cost = tf.reduce_sum(tf.square(activation - y))/(2 * m) # Set up the cost function to be standardized L2 loss (mean squared error of activation vs. actual y)
	optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost) # Set up the cost optimization method as iterative gradient descent
	with tf.Session() as sess: # Setup the session
		sess.run(tf.initialize_all_variables()) # Initialize placeholder values
		# Feed values in, begin iterative optimizing
		sess.run(optimizer)
		print "Optimization Finished!"
		print sess.run(W)
		print sess.run(cost)

if __name__ == "__main__":
	import_data()
	learn()