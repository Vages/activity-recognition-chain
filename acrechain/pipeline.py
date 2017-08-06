import numpy as np
import pickle
import itertools
import os
from time import time
from .conversion import timesync_from_cwa
from .load_csvs import load_accelerometer_csv
from .segment_and_calculate_features import segment_acceleration_and_calculate_features

my_folder = os.path.dirname(os.path.abspath(__file__))
default_model_path = os.path.join(my_folder, "healthy_3s_model.pickle")

model_paths = {100: os.path.join(my_folder, "healthy_3s_model.pickle"),
               50: os.path.join(my_folder, "healthy_3s_model_50hz.pickle")}

models = dict()

for hz in model_paths:
    with open(model_paths[hz], "rb") as f:
        models[hz] = pickle.load(f)

window_length = 3.0
overlap = 0.0


def complete_end_to_end_prediction(back_cwa, thigh_cwa, end_result_path, sampling_frequency=100):
    a = time()
    back_csv_path, thigh_csv_path, time_csv_path = timesync_from_cwa(back_cwa, thigh_cwa)
    b = time()
    print("TIME: Conversion and sync:", format(b-a, ".2f"), "s")
    a = time()
    back_acceleration, thigh_acceleration = load_accelerometer_csv(back_csv_path), load_accelerometer_csv(
        thigh_csv_path)
    back_features = segment_acceleration_and_calculate_features(back_acceleration, sampling_rate=sampling_frequency,
                                                                window_length=window_length, overlap=overlap)
    thigh_features = segment_acceleration_and_calculate_features(thigh_acceleration, sampling_rate=sampling_frequency,
                                                                 window_length=window_length, overlap=overlap)
    all_features = np.hstack([back_features, thigh_features])
    b = time()
    print("TIME: Feature extraction:", format(b-a, ".2f"), "s")

    a = time()
    predictions = models[sampling_frequency].predict(all_features)
    b = time()
    print("TIME: Prediction:", format(b - a, ".2f"), "s")
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
