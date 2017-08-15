from __future__ import print_function

import os
import pandas as pd
import subprocess
import time

OMCONVERT_SCRIPT_LOCATION = os.path.join(os.path.dirname(os.path.abspath(__file__)), "axivity_dependencies",
                                         "omconvert", "omconvert")
TIMESYNC_LOCATION = os.path.join(os.path.dirname(os.path.abspath(__file__)), "axivity_dependencies",
                                 "timesync", "timesync")


def run_omconvert(input_cwa, output_wav_path=None, output_csv_path=None):
    omconvert_script = OMCONVERT_SCRIPT_LOCATION

    shell_command = [omconvert_script, input_cwa]

    if output_wav_path is not None:
        print("WAV file will be output to", output_wav_path)
        shell_command += ['-out', output_wav_path]

    if output_csv_path is not None:
        print("CSV file will be output to", output_csv_path)
        shell_command += ['-csv-file', output_csv_path]

    if not os.path.exists(OMCONVERT_SCRIPT_LOCATION):
        print("Did not find a compiled version of OMconvert. "
              "Building OMconvert from source. This may take a while.")
        omconvert_directory = os.path.join(os.path.dirname(omconvert_script))
        make_omconvert_call = ["make", "-C", omconvert_directory]
        subprocess.call(make_omconvert_call)

    subprocess.call(shell_command)
    print("OMconvert finished.")


def timesync_from_cwa(master_cwa, slave_cwa, master_csv=None, slave_csv=None, time_csv=None, clean_up=True):
    timesync_script = TIMESYNC_LOCATION

    output_folder = os.path.dirname(master_cwa)

    master_basename_without_extension = os.path.splitext(os.path.basename(master_cwa))[0]
    slave_basename_without_extension = os.path.splitext(os.path.basename(slave_cwa))[0]

    master_wav = os.path.splitext(master_cwa)[0] + ".wav"
    slave_wav = os.path.splitext(slave_cwa)[0] + ".wav"

    if master_csv is None:
        master_csv = os.path.splitext(master_cwa)[0] + ".csv"

    if slave_csv is None:
        slave_csv = os.path.splitext(slave_cwa)[0] + ".csv"

    if time_csv is None:
        time_csv = os.path.splitext(master_cwa)[0] + "_timestamps.csv"

    print("Converting master and slave CWA files to intermediary WAV files")
    run_omconvert(master_cwa, output_wav_path=master_wav)
    run_omconvert(slave_cwa, output_wav_path=slave_wav)

    timesync_output_path = os.path.join(output_folder,
                                        master_basename_without_extension + "_" + slave_basename_without_extension + "_timesync_output.csv")

    intermediary_files = [master_wav, slave_wav, timesync_output_path]

    if not os.path.exists(TIMESYNC_LOCATION):
        print("Did not find a compiled version of Timesync. "
              "Building Timesync from source. This may take a while.")
        timesync_directory = os.path.dirname(timesync_script)
        make_call = ["make", "-C", timesync_directory]
        subprocess.call(make_call)

    print("Running Timesync")
    subprocess.call([timesync_script, master_wav, slave_wav, "-csv", timesync_output_path])

    newline = "\n"
    with open(timesync_output_path, 'r') as synchronized_csv_filereader, \
            open(time_csv, 'w') as time_csv_file_writer, \
            open(master_csv, 'w') as master_csv_file_writer, \
            open(slave_csv, 'w') as slave_csv_file_writer:
        for line in synchronized_csv_filereader:
            line_as_str_array = line.strip().split(",")
            time_line_data = line_as_str_array[0]
            master_line_data = line_as_str_array[1:4]
            slave_line_data = line_as_str_array[4:7]

            time_csv_file_writer.write(time_line_data + newline)
            master_csv_file_writer.write(",".join(master_line_data) + newline)
            slave_csv_file_writer.write(",".join(slave_line_data) + newline)

    """
    synchronized_data_frame = pd.read_csv(timesync_output_path, parse_dates=[0], header=None)
    
    
    synchronized_data_frame.to_csv(master_csv, header=False, index=False, columns=[1, 2, 3])
    print("Master accelerometer values saved to", master_csv)
    synchronized_data_frame.to_csv(slave_csv, header=False, index=False, columns=[4, 5, 6])
    print("Slave accelerometer values saved to", slave_csv)
    synchronized_data_frame.to_csv(time_csv, header=False, index=False, columns=[0])
    print("Time stamps saved to", time_csv)
    """

    if clean_up:
        print("Removing intermediary files", intermediary_files)
        for f in intermediary_files:
            subprocess.call(["rm", f])
    else:
        print("'clean_up' parameter set to false. "
              "The following intermediary files will remain on disk:", intermediary_files)

    return master_csv, slave_csv, time_csv


if __name__ == "__main__":
    cwa_1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "S03_LB.cwa")
    cwa_2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "S03_RT.cwa")
    timesync_from_cwa(cwa_1, cwa_2)
