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
from scipy.optimize import minimize
import datetime as dt


#%% functions
# function for sorting lists, containing strings with numbers (https://stackoverflow.com/a/48413757)
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key, reverse=False)

def lin(x, a, b):
    return a * x + b

#%% data path and files
data_path_lu = 'C:/Users/Eric/Documents/GitHub/HalfLife/Data/Lu/10000Bq_20200916_300s.csv'
data_path_iod = 'C:/Users/Eric/Documents/GitHub/HalfLife/Data/Iod/1000Bq_20201007_300s.csv'
data_path_mix = 'C:/Users/Eric/Documents/GitHub/HalfLife/Data/Mix/I-131_500Bq_Lu-177m_200Bq_300s_1.csv'

#%% read data
channels_lu = []
counts_lu = []
with open(data_path_lu, "r") as f:
    reader = csv.reader(f, delimiter=";")
    start = 14
    for i, line in enumerate(reader):
        if i > start:
            channels_lu.append(float(line[0]))
            counts_lu.append(float(line[1]))
new_counts_lu = np.asarray([i/max(counts_lu)/300 for i in counts_lu])
print(sum(new_counts_lu[0:]))

channels_iod = []
counts_iod = []
with open(data_path_iod, "r") as f:
    reader = csv.reader(f, delimiter=";")
    start = 14
    for i, line in enumerate(reader):
        if i > start:
            channels_iod.append(float(line[0]))
            counts_iod.append(float(line[1]))
new_counts_iod = np.asarray([i/max(counts_iod)/300 for i in counts_iod])
print(sum(new_counts_iod[0:]))

channels_mix = []
counts_mix = []
with open(data_path_mix, "r") as f:
    reader = csv.reader(f, delimiter=";")
    start = 14
    for i, line in enumerate(reader):
        if i > start:
            channels_mix.append(float(line[0]))
            counts_mix.append(float(line[1]))
new_counts_mix = np.asarray(counts_mix)

#%% calculations
# converting channels to energies
energy_channels_lu = []
for channel in channels_lu:
    energy = lin(channel, 0.46079, 0)
    energy_channels_lu.append(energy)

energy_channels_iod = []
for channel in channels_iod:
    energy = lin(channel, 0.46079, 0)
    energy_channels_iod.append(energy)

energy_channels_mix = []
for channel in channels_mix:
    energy = lin(channel, 0.46079, 0)
    energy_channels_mix.append(energy)


# objective function: Least square
# x is the array containing the wanted coefficients: c_lu, c_iod
def obj_func(x, counts_mix, new_counts_lu, new_counts_iod):
    y_pred = (x[0] * new_counts_lu) + (x[1] * new_counts_iod)
    return np.sum((counts_mix - y_pred)**2)


#%% scipy: minimize
# xinit = initial point
xinit = np.array([0, 0])
res = minimize(fun=obj_func, args=(counts_mix, new_counts_lu, new_counts_iod), x0=xinit)
print(res.x)

# plots
fig = plt.figure(figsize=(16,12))
ax1 = fig.add_subplot(4,1,1)
ax2 = fig.add_subplot(4,1,2)
ax3 = fig.add_subplot(4,1,3)

ax1.plot(np.asarray(energy_channels_lu), new_counts_lu, label='Lu-Spektrum (normiert)')
ax1.legend()
ax2.plot(np.asarray(energy_channels_iod), new_counts_iod, label='Iod-Spektrum (normiert)')
ax2.legend()
ax3.plot(np.asarray(energy_channels_mix), counts_mix, label='Mix')
ax3.legend()
ax4 = fig.add_subplot(4,1,4)
ax4.plot(np.asarray(energy_channels_mix), res.x[0]*new_counts_lu, color='red', label='Lu (berechnet')
ax4.plot(np.asarray(energy_channels_mix), res.x[1]*new_counts_iod, color='green', label='Iod (berechnet)')
ax4.plot(np.asarray(energy_channels_mix), (res.x[0]*new_counts_lu+res.x[1]*new_counts_iod), color='black', label='Lu+Iod')
ax4.legend()