# activity-recognition-chain
An implementation of the Activity Recognition Chain described in Bulling et al.’s 2014 paper “A Tutorial on Human Activity Recognition Using Body-Worn Inertial Sensors” with Scikit-learn’s Random Forests Classifiers

## Requirements

- Python 3 (tested with Python 3.5)
- Linux

## Installation
```
pip install git+https://github.com/Vages/activity-recognition-chain.git
```

## Usage
### Default
To get predictions from two CWA files sampled at 100 Hz:

```python
from acrechain import complete_end_to_end_prediction

complete_end_to_end_prediction("/path/to/back.cwa", "/path/to/thigh.cwa", "/path/to/output.csv")
```

The file at `/path/to/output.csv` should look something like this:

```
2016-11-22 12:00:00.000, 7
2016-11-22 12:00:03.000, 7
2016-11-22 12:00:06.000, 8
2016-11-22 12:00:09.000, 8
2016-11-22 12:00:12.000, 8
2016-11-22 12:00:15.000, 8
2016-11-22 12:00:18.000, 8
2016-11-22 12:00:21.000, 8
2016-11-22 12:00:24.000, 8
2016-11-22 12:00:27.000, 8
2016-11-22 12:00:30.000, 8
2016-11-22 12:00:33.000, 8
2016-11-22 12:00:36.000, 8
2016-11-22 12:00:39.000, 8
2016-11-22 12:00:42.000, 8
2016-11-22 12:00:45.000, 8
…
```

### Other sampling frequencies
To predict for other sampling rates, e.g. 50 Hz, pass them to the function using the `sampling_frequency` parameter:

```python
from acrechain import complete_end_to_end_prediction

complete_end_to_end_prediction(back_cwa, thigh_cwa, end_result_path, sampling_frequency=50)
```

### Adjust the number of minutes that are read at a time
The function makes the predictions for chunks of the file at a time to conserve memory. 
The default size of a chunk is 15 minutes, but this parameter can be adjusted using the `minutes_to_read_in_a_chunk` 
parameter.

For example, to predict for 30 minutes at a time, you would type the following.

```python
from acrechain import complete_end_to_end_prediction

complete_end_to_end_prediction(back_cwa, thigh_cwa, end_result_path, minutes_to_read_in_a_chunk=30)
```