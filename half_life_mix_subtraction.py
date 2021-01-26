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

def fit_exp_linear_iod(x, y, C):
    y = y - C
    y = np.log(y)
    K, A_log = np.polyfit(x, y, 1)
    A = np.exp(A_log)
    return A, K # k = slope

def fit_exp_linear_lu(x, y, C=0):
    y = y - C
    y = np.log(y)
    K, A_log = np.polyfit(x, y, 1)
    A = np.exp(A_log)
    return A, K


#%% Mixture
# data path and files
data_path = 'C:/Users/Eric/Documents/GitHub/HalfLife/Data/Mix/'
data_files = [f for f in os.listdir(data_path) if f.endswith(".csv")]
print(sorted_alphanumeric(data_files))

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
                if float(line[1]) < 0:
                    counts[j].append(float(0))
                else:
                    counts[j].append(float(line[1]))
    j += 1


x_iod_min, x_iod_max = 803, 810
x_lu_min, x_lu_max = 56, 62
# sum of all counts over the intervall (energy window: Iod, Lu)
decay_sum_counts_iod = []
decay_sum_counts_lu = []
for block in counts:
    decay_sum_counts_iod.append(sum(block[x_iod_min:x_iod_max])) #(sum(block[804:810]))
    decay_sum_counts_lu.append(sum(block[x_lu_min:x_lu_max]))
#decay_sum_counts_iod = [520.0, 377.0, 172.0, 120.0, 87.0, 80.0]


# convert time points into seconds and calculate ds
dates = [dt.datetime(2020,10,7), dt.datetime(2020,10,13), dt.datetime(2020,10,27), dt.datetime(2020,11,6), dt.datetime(2020,11,18), dt.datetime(2020,11,30)]
ds_begin = [0]
ds_end = []
for time in dates[1:2]:
    ds_begin.append((time-dates[0]).total_seconds())

for time in dates[3:]:
    ds_end.append((time-dates[0]).total_seconds())


# use exp fit function (Lu)
A_lu, K_lu = fit_exp_linear_lu(np.asarray(ds_end), np.asarray(decay_sum_counts_lu[3:]))
half_life_lu = round(np.log(2)/(-K_lu*60*60*24), 1)

# use exp fit function (Iod)
A_iod, K_iod = fit_exp_linear_iod(np.asarray(ds_begin), np.asarray(decay_sum_counts_iod[0:2]), A_lu)
half_life_iod = round(np.log(2)/(-K_iod*60*60*24), 1)


# define x axis cvalues for fit-plot
tmin, tmax = 0, max(ds_end)
num = 20
t = np.linspace(tmin, tmax, num)

A_ges, K_ges = fit_exp_linear_lu(t, func(t, A_iod, -K_iod) + func(t, A_lu, -K_lu))
half_life_ges = round(np.log(2)/(-K_ges*60*60*24), 1)

print('--- Results ---')
print('Iod activity:', (round(A_iod*0.932, 2)))
print('Lu activity:', (round(A_lu*1.018, 2)))
print('---------------')

# plot counts and fit
fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)

ax1.set_title('Half-Life (I): ' + str(half_life_iod) + 'd ; Half-Life (Lu): ' + str(half_life_lu) + 'd')
ax1.plot(ds_begin, decay_sum_counts_iod[0:2], '.', label='Counts (Iod window)')
ax1.plot(ds_end, decay_sum_counts_iod[3:], 'x', label='Counts (Iod window)')
ax1.plot(t, func(t, A_iod, -K_iod), color='green', label='Fit (Iod):\n y = %0.2f e^{-ln(2)/%0.2f t}' % (A_iod,half_life_iod))
ax1.plot(t, func(t, A_lu, -K_lu), color='red', label='Fit (Lu):\n y = %0.2f e^{-ln(2)/%0.2f t}' % (A_lu,half_life_lu))
ax1.plot(t, (func(t, A_iod, -K_iod) + func(t, A_lu, -K_lu)), color='black', label='Iod + Lu: %0.2f Bq, %0.2f d' % (A_ges,half_life_ges))
ax1.set_ylabel('counts')
ax1.legend(bbox_to_anchor=(1.05, 1.1), fancybox=True, shadow=True)
