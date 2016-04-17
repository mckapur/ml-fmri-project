# Constants
FOREST_FIRE_DATASET_PATH = './../datasets/forest-fire-damage/prepared_ff_damage_dataset.csv'
STANDARDIZED_DATASET_DELIMETER = ','
LEARNING_RATE = 1e-4

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

# Learning

# Declare the numerical variables for our weights and biases in log(X*W_in + b_in)*W_out + b_out
W_out = 0
W_in = 0
b_out = 0
b_in = 0

def logarithmic_activation(W_out_tf, W_in_tf, b_out_tf, b_in_tf):
	# Compute a logarithmic activation with different coefficients for shifts and stretching, with model: log(X*W_in + b_in)*W_out + b_out
	inner = tf.add(tf.matmul(X, W_in_tf), b_in_tf) # Compute the X*W_in + b_in component
	composite = tf.add(tf.matmul(tf.log(inner), W_out_tf), b_out_tf) # Compute ln()*X_out + b_out composite
 	return composite

def learn(): # Train a model from data
	# Declare the TensorFlow versions of the weights and biases
	W_out_tf = tf.Variable(tf.constant(0.0, shape=[1, 1]), name="vertical_stretch")
	W_in_tf = tf.Variable(tf.ones([n, 1]), name="horizontal_stretch")
	b_out_tf = tf.Variable(tf.constant(0.0, shape=[m, 1]), name="vertical_shift")
	b_in_tf = tf.Variable(tf.constant(0.0, shape=[m, 1]), name="horizontal_shift")

	activation = logarithmic_activation(W_out_tf, W_in_tf, b_out_tf, b_in_tf) # We will use a logarithmic transformation/model
	cost_function = tf.reduce_sum(tf.square(activation - y))/(2 * m) # Setup the cost function to be standardized L2 loss (mean squared error of activation vs. actual y)
	optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost_function) # Setup the cost optimization method as iterative gradient descent

	with tf.Session() as sess: # Create the session
		sess.run(tf.initialize_all_variables()) # Initialize the placeholder values

		for i in range(10000):
			if i % 100 == 0:
				print 'Step #%d: %f' % (i, sess.run(cost_function))
			sess.run(optimizer) # Perform optimization

		# Initialize our numerical weights from the TF session versions
		global W_out, W_in, b_out, b_in
		W_out = sess.run(W_out_tf)
		W_in = sess.run(W_in_tf)
		b_out = sess.run(b_out_tf)
		b_in = sess.run(b_in_tf)
		print sess.run(cost_function)

def output_results(): # output optimization results through the console
	print W_out
	print W_in
	print b_out
	print b_in

if __name__ == "__main__":
	import_data()
	learn()
	# output_results()
