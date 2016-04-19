# Imports
import math
import numpy as np
import csv
import load_data
from linear_regression import LinearRegression

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

	model = LinearRegression()
	model.fit(X, y)
	
	print compute_cost(X, y, model.predict(X))
