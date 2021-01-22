#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Eric Einspänner

This program is free software.
"""
import os
import re
import csv
from itertools import repeat
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt


#%% functions
# function for sorting lists, containing strings with numbers (https://stackoverflow.com/a/48413757)
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key, reverse=False)

def func(x, a, b):
    return a * np.exp(-b * x)

def fit_exp_linear(x, y, C=0):
    y = y - C
    y = np.log(y)
    K, A_log = np.polyfit(x, y, 1)
    A = np.exp(A_log)
    return A, K # k = Anstieg


#%% Iod-131
# data path and files
data_path = 'C:/Users/Eric/Documents/GitHub/HalfLife/Data/Iod/'
data_files = [f for f in os.listdir(data_path) if f.endswith(".csv")]


# define lists for channels and counts
channels = [[] for i in repeat(None, len(data_files))]
counts = [[] for i in repeat(None, len(data_files))]


# read measured dataset in lists
j = 0
for file in sorted_alphanumeric(data_files):
    with open(data_path + file, "r") as f:
        reader = csv.reader(f, delimiter=";")
        start = 14
        for i, line in enumerate(reader):
            if i > start:
                channels[j].append(float(line[0]))
                counts[j].append(float(line[1]))
    j += 1


# sum of all counts over the intervall (energy window)
decay_sum_counts = []
for block in counts:
    decay_sum_counts.append(sum(block[803:810]))


# convert time points into seconds and calculate ds
dates = [dt.datetime(2020,7,10), dt.datetime(2020,11,6), dt.datetime(2020,11,18)]
ds = [0]
for time in dates[1:]:
    ds.append((time-dates[0]).total_seconds())


# use exp fit function
A, K = fit_exp_linear(np.asarray(ds), np.asarray(decay_sum_counts))
half_life = round(np.log(2)/(-K*60*60*24), 1)


# define x axis cvalues for fit-plot
tmin, tmax = 0, ds[len(ds)-1]
num = 20
t = np.linspace(tmin, tmax, num)


# plot counts and fit
plt.title('Calculated Half-Life: ' + str(half_life) + 'd')
plt.plot(ds, decay_sum_counts, '.', label='Measured Points')
plt.plot(t, func(t, A, -K), color='green', label='Fitted Function:\n y = %0.2f e^{-ln(2)/%0.2f t}' % (A,half_life))
plt.xlabel('dt [s]')
plt.ylabel('Counts')
plt.legend()
plt.show()