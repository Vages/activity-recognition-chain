import numpy as np
import pickle
import itertools
import os
from time import time

import pandas.errors
import pandas as pd
from .conversion import timesync_from_cwa
from .segment_and_calculate_features import segment_acceleration_and_calculate_features

model_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

model_paths = {
    100: os.path.join(model_folder, "healthy_3.0s_model_100.0hz.pickle"),
    50: os.path.join(model_folder, "healthy_3.0s_model_50.0hz.pickle"),
    25: os.path.join(model_folder, "healthy_3.0s_model_25.0hz.pickle"),
    20: os.path.join(model_folder, "healthy_3.0s_model_20.0hz.pickle"),
    10: os.path.join(model_folder, "healthy_3.0s_model_10.0hz.pickle"),
    5: os.path.join(model_folder, "healthy_3.0s_model_5.0hz.pickle"),
    4: os.path.join(model_folder, "healthy_3.0s_model_4.0hz.pickle"),
    2: os.path.join(model_folder, "healthy_3.0s_model_2.0hz.pickle"),
    1: os.path.join(model_folder, "healthy_3.0s_model_1.0hz.pickle"),
}

models = dict()

for hz in model_paths:
    with open(model_paths[hz], "rb") as f:
        models[hz] = pickle.load(f)

window_length = 3.0
overlap = 0.0


def complete_end_to_end_prediction(back_cwa, thigh_cwa, end_result_path, sampling_frequency=100,
                                   minutes_to_read_in_a_chunk=15):
    a = time()
    back_csv_path, thigh_csv_path, time_csv_path = timesync_from_cwa(back_cwa, thigh_cwa)
    b = time()
    print("TIME: Conversion and sync:", format(b - a, ".2f"), "s")
    a = time()
    predictions = load_csv_and_extract_features(back_csv_path, thigh_csv_path, sampling_frequency,
                                                minutes_to_read_in_a_chunk)
    b = time()
    print("TIME: Feature extraction and prediction:", format(b - a, ".2f"), "s")
    time_stamp_skip = int(sampling_frequency * window_length * (1.0 - overlap))
    a = time()
    with open(time_csv_path, "r") as t:
        time_stamp_lines = [_.strip() for _ in itertools.islice(t, 0, None, time_stamp_skip)]

    output_lines = [tsl + ", " + str(pred) + "\n" for tsl, pred in zip(time_stamp_lines, predictions)]

    with open(end_result_path, "w") as ef:
        ef.writelines(output_lines)
    b = time()
    print("TIME: Writing to disk:", format(b - a, ".2f"), "s")

    for tmp_file in [back_csv_path, thigh_csv_path, time_csv_path]:
        os.remove(tmp_file)


def load_csv_and_extract_features(back_csv_path, thigh_csv_path, sampling_frequency, minutes_to_read_in_a_chunk):
    number_of_samples_in_a_window = int(sampling_frequency * window_length)
    number_of_windows_to_read = int(round(minutes_to_read_in_a_chunk * 60 / window_length))
    number_of_samples_to_read = number_of_samples_in_a_window * number_of_windows_to_read

    window_start = 0

    predictions = []

    while True:
        try:
            this_back_window = pd.read_csv(back_csv_path, skiprows=window_start, nrows=number_of_samples_to_read,
                                           delimiter=",", header=None).as_matrix()
            this_thigh_window = pd.read_csv(thigh_csv_path, skiprows=window_start, nrows=number_of_samples_to_read,
                                            delimiter=",", header=None).as_matrix()

            window_start += number_of_samples_to_read

            back_features = segment_acceleration_and_calculate_features(this_back_window,
                                                                        sampling_rate=sampling_frequency,
                                                                        window_length=window_length, overlap=overlap)
            thigh_features = segment_acceleration_and_calculate_features(this_thigh_window,
                                                                         sampling_rate=sampling_frequency,
                                                                         window_length=window_length, overlap=overlap)

            boths_features = np.hstack((back_features, thigh_features))
            this_windows_predictions = models[sampling_frequency].predict(boths_features)
            predictions.append(this_windows_predictions)
        except pandas.errors.EmptyDataError:  # There are no more lines to read
            break

    predictions = np.hstack(predictions)
    return predictions


if __name__ == "__main__":
    cwa_1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "S03_LB.cwa")
    cwa_2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "S03_RT.cwa")
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "timestamped_predictions.csv")

    complete_end_to_end_prediction(cwa_1, cwa_2, output_path, minutes_to_read_in_a_chunk=60)
