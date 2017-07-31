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
To get predictions from two CWA files:

```python
from acrechain import complete_end_to_end_prediction

complete_end_to_end_prediction(back_cwa, thigh_cwa, end_result_path)
```
