# Constants
FOREST_FIRE_DATASET_PATH = './../datasets/forest-fire-damage/prepared_ff_damage_dataset.csv'
STANDARDIZED_DATASET_DELIMETER = ','
LEARNING_RATE = 1e-6
MAX_EPOCHS = 10000

# Imports
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

def linear_activation(W_tf, b_tf):
	# Compute a linear activation with model: X*W + b
 	return tf.add(tf.matmul(X, W_tf), b_tf)

def learn(): # Train a model from data
	# Declare the TensorFlow versions of the weight(s) and bias variables
	W_tf = tf.Variable(tf.zeros([n, 1]), name="weight")
	b_tf = tf.Variable(tf.constant(0.0, shape=[1, 1]), name="bias")

	activation = linear_activation(W_tf, b_tf) # We will use a linear model
	cost_function = tf.reduce_sum(tf.square(activation - y))/(2 * m) # Setup the cost function to be standardized L2 loss (mean squared error of activation vs. actual y)
	optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost_function) # Setup the cost optimization method as iterative gradient descent

	with tf.Session() as sess: # Create the session
		sess.run(tf.initialize_all_variables()) # Initialize the placeholder values

		for i in range(MAX_EPOCHS): # Perform MAX_EPOCHS steps of optimization
			if i % 100 == 0:
				print 'Step #%d: %f' % (i, sess.run(cost_function))
			sess.run(optimizer) # Perform a single weight update

		# Initialize our numerical weights from the TensorFlow session versions
		global W, b
		W = sess.run(W_tf)
		b = sess.run(b_tf)
		print 'Optimization completed with cost %f', sess.run(cost_function)

def output_results(): # Output optimization results through the console
	print 'Final weight vector %f', W
	print 'Final bias value %f', b

if __name__ == "__main__":
	import_data()
	learn()
	output_results()
