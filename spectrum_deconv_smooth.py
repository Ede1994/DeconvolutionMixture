#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Eric Einspänner

This program is free software.
"""
import re
import csv

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.integrate import simps
from scipy.signal import savgol_filter

#%% functions
# function for sorting lists, containing strings with numbers (https://stackoverflow.com/a/48413757)
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key, reverse=False)

def lin(x, a, b):
    return a * x + b

# objective function: Least square
# x is the array containing the wanted coefficients: c_lu, c_iod
def obj_func(x, counts_mix, new_counts_lu, new_counts_iod):
    y_pred = (x[0] * new_counts_lu) + (x[1] * new_counts_iod)
    return np.sum((counts_mix - y_pred)**2)

# callback function for more scipy.optimize.minimize infos
def callbackF(x):
    global Nfeval
    print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}'.format(Nfeval, x[0], x[1], obj_func(x, counts_mix, new_counts_lu, new_counts_iod)))
    Nfeval += 1

#%% data path and files

data_path_lu = 'C:/Users/Eric/Documents/GitHub/HalfLife/Data/Lu/10000Bq_20200916_300s.csv'
data_path_iod = 'C:/Users/Eric/Documents/GitHub/HalfLife/Data/Iod/1000Bq_20201007_300s.csv'
#data_path_mix = 'C:/Users/Eric/Documents/GitHub/HalfLife/Data/Mix/I-131_500Bq_Lu-177m_200Bq_300s_4.csv'
#data_path_mix = 'C:/Users/Eric/Documents/GitHub/HalfLife/Data/Mix2/AWM_MIX_100vs100_3600s.csv'
data_path_mix = 'C:/Users/Eric/Documents/GitHub/HalfLife/Data/Mix2/AWM_MIX_50vs97_3600s.csv'
#data_path_mix = 'C:/Users/Eric/Documents/GitHub/HalfLife/Data/Mix2/AWM_MIX_5vs86_3600s.csv'

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
new_counts_lu = np.asarray([i/sum(counts_lu) for i in counts_lu])
# Savitzky-Golay filter for smoothing
new_counts_lu_smooth = savgol_filter(new_counts_lu, 21, 3) # window size, polynomial order

channels_iod = []
counts_iod = []
with open(data_path_iod, "r") as f:
    reader = csv.reader(f, delimiter=";")
    start = 14
    for i, line in enumerate(reader):
        if i > start:
            channels_iod.append(float(line[0]))
            counts_iod.append(float(line[1]))
new_counts_iod = np.asarray([i/sum(counts_iod) for i in counts_iod])
new_counts_iod_smooth = savgol_filter(new_counts_iod, 21, 3)


channels_mix = []
counts_mix = []
bg_mix = []
with open(data_path_mix, "r") as f:
    reader = csv.reader(f, delimiter=";")
    start = 14
    for i, line in enumerate(reader):
        if i > start:
            channels_mix.append(float(line[0]))
            counts_mix.append(float(line[1]))
            bg_mix.append(float(line[2]))
new_counts_mix = np.asarray(np.subtract(counts_mix, bg_mix))
new_counts_mix_smooth = savgol_filter(new_counts_mix, 31, 3)

#%% Converting channels to energies

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

#%% Optimization

# scipy: minimize
Nfeval = 1

print('\n--- Start Optimization ---')
print('{0:4s}       {1:9s}      {2:9s}       {3:9s}'.format('Iter', ' c_Lu', ' c_Iod', 'obj. Func.'))

# without smoothing
xinit = np.array([0, 0])
#res = minimize(fun=obj_func, args=(counts_mix, new_counts_lu, new_counts_iod), x0=xinit, method='Nelder-Mead',\
#               tol=0.001, callback=callbackF, options={'maxiter':2000 ,'disp': True})

# with smoothing  
res = minimize(fun=obj_func, args=(new_counts_mix_smooth, new_counts_lu_smooth, new_counts_iod_smooth), x0=xinit, method='Nelder-Mead',\
               tol=0.001, callback=callbackF, options={'maxiter':2000 ,'disp': True})
print('---------------------------')

#%% Calculations

# Define specific windows
# Lu boundaries, A window (width: 72)
Lu_peak = 126 #np.argmax(new_counts_lu)
Lu_min, Lu_max = Lu_peak - 36, Lu_peak + 36
# Iod boundaries, E window (width: 236)
Iod_peak = 791 #np.argmax(new_counts_iod)
Iod_min, Iod_max = Iod_peak - 118, Iod_peak + 118

print(sum(new_counts_mix_smooth[Iod_min:Iod_max]))

print('\n--- Energy Windows ---')
print('Lu: Peak {}, Min {}, Max {}'.format(Lu_peak, Lu_min, Lu_max))
print('Iod: Peak {}, Min {}, Max {}'.format(Iod_peak, Iod_min, Iod_max))
print('---------------------')

# results of normalization
print('\n--- Normalization ---')
print('Area Lu sopectrum (norm.):', round(simps(new_counts_lu_smooth, channels_lu),3))
print('Area Iod spectrum (norm.):', round(simps(new_counts_iod_smooth, channels_iod),3))
print('---------------------')

print('\n--- Optimized Factors ---')
print('c_Lu:', res.x[0])
print('c_Iod:', res.x[1])
print('-------------------------')

# Calculation of specific activities
Lu_act = (simps((res.x[0]*new_counts_lu_smooth)[Lu_min:Lu_max], channels_lu[Lu_min:Lu_max]) / 3600) * 12.68
Iod_act = (simps((res.x[1]*new_counts_iod_smooth)[Iod_min:Iod_max], channels_iod[Iod_min:Iod_max]) / 3600) * 16.5
print('\n--- Calculated Activities ---')
print('Lu activity [Bq]:', round(Lu_act , 2))
print('Iod activity [Bq]:', round(Iod_act, 2))
print('-----------------------------')

# Calculation of the coefficient of determination (R^2)
y_mean = sum(new_counts_mix_smooth)/float(len(new_counts_mix_smooth))
ss_tot = sum((yi-y_mean)**2 for yi in new_counts_mix_smooth)
ss_err = sum((yi-fi)**2 for yi,fi in zip(new_counts_mix_smooth,(res.x[0]*new_counts_lu_smooth+res.x[1]*new_counts_iod_smooth)))
r2 = 1 - (ss_err/ss_tot)
print('\n--- Coefficient Of Determination ---')
print('R^2:', round(r2, 3))
print('------------------------------------')

#%% plots

fig = plt.figure(figsize=(16,15))
ax1 = fig.add_subplot(5,1,1)
ax2 = fig.add_subplot(5,1,2)
ax3 = fig.add_subplot(5,1,3)
ax4 = fig.add_subplot(5,1,4)
ax5 = fig.add_subplot(5,1,5)

# Lu Spectrum (normalized)
ax1.plot(np.asarray(energy_channels_lu), new_counts_lu, label='Lu Spectrum (normalized)')
ax1.plot(np.asarray(energy_channels_lu), new_counts_lu_smooth, label='Lu Spectrum (smooth)')
ax1.set_ylabel('x_i/sum(x_i)')
ax1.legend()

# Iod Spectrum (normalized)
ax2.plot(np.asarray(energy_channels_iod), new_counts_iod, label='Iod Spectrum (normalized)')
ax2.plot(np.asarray(energy_channels_iod), new_counts_iod_smooth, label='Iod Spectrum (smooth)')
ax2.set_ylabel('x_i/sum(x_i)')
ax2.legend()

# Mixture (measured)
ax3.plot(np.asarray(energy_channels_mix), counts_mix, label='Mixture (measured)')
ax3.plot(np.asarray(energy_channels_mix), new_counts_mix_smooth, label='Mixture (smooth)')
ax3.set_ylabel('counts')
ax3.legend()

# Lu + Iod (Calculated)
ax4.plot(np.asarray(energy_channels_lu), res.x[0]*new_counts_lu_smooth, color='red', label='Lu (calculated+smooth)')
ax4.plot(np.asarray(energy_channels_iod), res.x[1]*new_counts_iod_smooth, color='green', label='Iod (calculated+smooth)')
ax4.plot(np.asarray(energy_channels_mix), (res.x[0]*new_counts_lu_smooth+res.x[1]*new_counts_iod_smooth), color='black', label='Lu + Iod (smooth)')
ax4.set_ylabel('counts')
ax4.legend()

# Diff
ax5.plot(np.asarray(energy_channels_mix), new_counts_mix_smooth, label='Mixture (smooth)')
ax5.plot(np.asarray(energy_channels_mix), (res.x[0]*new_counts_lu_smooth+res.x[1]*new_counts_iod_smooth), color='black', label='Lu + Iod (smooth)')
ax5.plot(np.asarray(energy_channels_mix), (new_counts_mix_smooth-(res.x[0]*new_counts_lu_smooth+res.x[1]*new_counts_iod_smooth)), color='red', label='Diff')
ax5.set_xlabel('Energy (keV)')
ax5.set_ylabel('counts')
ax5.legend()