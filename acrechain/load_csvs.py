import numpy as np


def load_accelerometer_csv(file_path, delimiter=","):
    return np.loadtxt(fname=file_path, delimiter=delimiter, dtype="float")
