#!/usr/bin/env python
"""
Cheap and dirty script to examine an experiment log
"""
import numpy as np

# Process experiment log into something useful
normal_1_array = []
normal_2_array = []
mixture_array = []
log_array = []
with open("experiment.log", "r") as file:
    for i, line in enumerate(file):
        if i < 10000:
            continue
        if i == 10180:
            break
        array = np.array(line[11:].split(", "), dtype=np.float)
        normal_1_array.append(-array[0])
        normal_2_array.append(-array[1])
        mixture_array.append(array[2])
        log_array.append(-array[3])
