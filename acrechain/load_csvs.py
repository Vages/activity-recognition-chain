import numpy as np


def load_accelerometer_csv(file_path, delimiter=","):
    return np.loadtxt(fname=file_path, delimiter=delimiter, dtype="float")


def load_label_csv(file_path):
    return np.loadtxt(fname=file_path, dtype="int")
