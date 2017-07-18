from __future__ import print_function

import os
import pandas as pd

OMCONVERT_LOCATION = os.path.join(os.path.dirname(os.path.abspath(__file__)), "axivity_dependencies", "omconvert", "omconvert")
TIMESYNC_LOCATION = os.path.join(os.path.dirname(os.path.abspath(__file__)), "axivity_dependencies", "timesync", "timesync")


def run_omconvert(input_cwa, output_wav=None, output_csv=None):
    import subprocess

    omconvert = OMCONVERT_LOCATION

    command = [omconvert, input_cwa]

    if output_csv is not None:
        command += ['-csv-file', output_csv]

    if output_wav is not None:
        command += ['-out', output_wav]

    if not os.path.exists(OMCONVERT_LOCATION):
        print("Did not find a compiled version of OMconvert. "
              "Building OMconvert from source. This may take a while.")
        omconvert_directory = os.path.join(os.path.dirname(omconvert))
        make_call = ["make", "-C", omconvert_directory]
        subprocess.call(make_call)

    subprocess.call(command)


def timesync_from_cwa(cwas, output_csv, clean_up=True, nrows=None):
    import subprocess

    timesync = TIMESYNC_LOCATION

    output_folder = os.path.split(output_csv)[0]

    master_cwa = cwas[0]

    slave_cwas = cwas[1:]

    master_wav = os.path.splitext(master_cwa)[0] + ".wav"
    run_omconvert(master_cwa, output_wav=master_wav)

    wav_files = [master_wav]
    csv_files = []

    if not os.path.exists(TIMESYNC_LOCATION):
        print("Did not find a compiled version of Timesync. "
              "Building Timesync from source. This may take a while.")
        timesync_directory = os.path.join(os.path.dirname(timesync))
        make_call = ["make", "-C", timesync_directory]
        subprocess.call(make_call)

    for i, s in enumerate(slave_cwas):
        s_prefix = os.path.splitext(s)[0]
        slave_wav = s_prefix + ".wav"
        wav_files.append(slave_wav)
        run_omconvert(s, output_wav=slave_wav)

        # Synchronize them and make them a CSV
        tmp_output_path = os.path.join(output_folder, s_prefix + ".csv")

        csv_files.append(tmp_output_path)
        subprocess.call([timesync, master_wav, slave_wav, "-csv", tmp_output_path])

    first_csv_file = csv_files[0]
    first_dataframe = pd.read_csv(first_csv_file, parse_dates=[0], header=None, nrows=nrows)

    if len(csv_files) == 1:
        os.rename(first_csv_file, output_csv)
        csv_files.remove(first_csv_file)
        output_dataframe = first_dataframe
    else:
        data_frames = [first_dataframe]
        for next_csv in csv_files[1:]:
            data_frames.append(pd.read_csv(next_csv, header=None, usecols=[4, 5, 6], nrows=nrows))

        output_dataframe = pd.concat(data_frames, axis=1, ignore_index=True)
        output_dataframe.to_csv(output_csv, header=False, index=False)

    if clean_up:
        print("Cleaning up files")
        for f in csv_files + wav_files:
            subprocess.call(["rm", f])

    return output_dataframe
