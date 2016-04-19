# Imports
import numpy as np
import load_data
from linear_regression import LinearRegression
from matplotlib.mlab import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Take an output vector y and apply a reverse log transform
def reverse_transform(y):
	return np.exp(y) - 1

# Computes the non-transformed cost on all training samples
def compute_cost(X, y, predictions):
	# Generate all predictions
	predictions = reverse_transform(predictions)
	y = reverse_transform(y)

	return np.sum((predictions - y) ** 2)/(2 * X.shape[0])

if __name__ == "__main__":
	X, y = load_data.import_data()

	# model = LinearRegression()
	# model.fit(X, y)

	res = PCA(X).Y
	x_data = []
	y_data = []
	z_data = []

	# y = reverse_transform(y)

	for i in range(len(y)):
		x_data.append(res[i, 0])
		y_data.append(res[i, 1])
		z_data.append(y[i])

	fig1 = plt.figure() # Make a plotting figure
	ax = Axes3D(fig1) # use the plotting figure to create a Axis3D object.
	ax.scatter(x_data, y_data, z_data, 'bo')

	ax.set_zlabel('Transformed area')
	plt.show()
