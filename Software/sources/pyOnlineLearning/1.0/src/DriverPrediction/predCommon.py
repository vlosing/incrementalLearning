__author__ = 'viktor'
import numpy as np
TURNING_LABELS = [0, 1, 2]
TURNING_LABELS_SYMBOLS = ['|', '<', '>']
STRAIGHT_LABEL_IDX = 0
LEFT_TURN_LABEL_IDX = 1
RIGHT_TURN_LABEL_IDX = 2


LABEL_NAMES=['pass', 'stop', 'turn', 'turn left', 'turn right']
PASS_LABEL=0
STOP_LABEL=1
TURNING_LABEL=2
TURNING_LEFT_LABEL=3
TURNING_RIGHT_LABEL=4

TIME_IDX = 0
GPSLAT_IDX = 1
GPSLONG_IDX = 2
VELOCITY_IDX = 3
ACCEL_IDX = 4
AVS_IDX = 5
STOP_DISTANCE_IDX = 6
FEATURE_COLUMN_NAMES = np.array(['Time', 'Latitude', 'Longitude', 'Velocity', 'Acceleration', 'AVS', 'Stop_Distance'])
