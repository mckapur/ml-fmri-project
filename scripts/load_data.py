import csv
import numpy as np

'''
Load and preprocess raw data from CSV into usable format that we can feed into
whatever model we want to.
'''

# Load raw data from CSV, return numpy array of data from csv
def import_data(path='./../datasets/forest-fire-damage/prepared_ff_damage_dataset.csv', delimeter=','):
    dataset_reader = csv.reader(open(path, 'rb'), delimiter=delimeter)
    next(dataset_reader, None) # Skip column headers
    raw_dataset = list(dataset_reader)

    ff_dataset = np.array(raw_dataset).astype(np.float32)

    # Format the data into features/targets
    m, n = ff_dataset.shape
    n = n - 1 # Discounting output in imported dataset

    # Separating input features and output from dataset
    X = ff_dataset[:, :-1].reshape((m, n))
    y = ff_dataset[:, n].reshape((m, 1))

    # Apply a transformation of ln(x + 1) to the output vector to linearize it
    # because the output is skewed towards 0
    y = np.log(np.add(y, np.ones(y.shape)))

    return (X, y)
