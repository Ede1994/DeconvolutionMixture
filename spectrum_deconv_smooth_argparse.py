#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Eric Einspänner

Spectrum deconvolution for 2 components: Lu177m, I131
Supports argparse for Shell integration

This program is free software.
"""
import re
import os
import csv
import string
import argparse

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize, Bounds
from scipy.integrate import simps, trapezoid
from scipy.signal import savgol_filter # Savitzky-Golay filter for smoothing


#%% functions

# function for sorting lists, containing strings with numbers (https://stackoverflow.com/a/48413757)
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key, reverse=False)

# optimize smoothing filter
def optimized_smoothing(counts_arr):
    winsize = 7 # must be larger than polynomial order
    for num in range(1000):
        counts_smooth_arr = savgol_filter(counts_arr, winsize, 2) # window size, polynomial order

        y_mean = sum(counts_arr)/float(len(counts_arr))
        ss_tot = sum((yi-y_mean)**2 for yi in counts_arr)
        ss_err = sum((yi-fi)**2 for yi,fi in zip(counts_arr, counts_smooth_arr))
        r2 = 1 - (ss_err/ss_tot)
    
        if r2 > 0.91:
            break
        elif winsize >= 15:
            break
        else:
            winsize += 2
        
    return winsize, counts_smooth_arr, r2

# linear func for transformation of channels to eneries
def lin(x, a, b):
    return a * x + b

# objective function: Least square
# x is the array containing the wanted coefficients: c_lu, c_iod
def obj_func(x, new_counts_mix_smooth, new_counts_lu_smooth, new_counts_iod_smooth):
    y_pred = (x[0] * new_counts_lu_smooth) + (x[1] * new_counts_iod_smooth)
    return np.sum((new_counts_mix_smooth - y_pred)**2)

# callback function for more scipy.optimize.minimize infos
def callbackF(x):
    global Nfeval
    print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}'.format(Nfeval, x[0], x[1], obj_func(x, new_counts_mix_smooth, new_counts_lu, new_counts_iod)))
    Nfeval += 1


#%% data path and files

parser = argparse.ArgumentParser(description='Spectrum deconvolution for a mixture')
parser.add_argument('-rLu', '--refLu', help='Reference file for Lu-spectrum')
parser.add_argument('-rIod', '--refIod', help='Reference file for Iod-spectrum')
parser.add_argument('-sMix', '--sMixture', help='File for Mixture-spectrum')
parser.add_argument('-t', '--time', help='Measuring time for mixture spectrum', type=float)
args = parser.parse_args()

# define measuring time
dt = args.time


#%% Lu: pure (reference)spectrum
channels_lu = []
counts_lu = []

with open(args.refLu, "r") as f:
    reader = csv.reader(f, delimiter=";")

    # choose the right start parameter (first channel) and set energy2channel CF
    for finder in reader:
        if finder == []:
            continue   
        if finder[0] == 'Kalibrierfaktor:':
            CF_lu = float(finder[1])
        if finder[0] == 'Kanal':
            number = reader.line_num

with open(args.refLu, "r") as f:
    reader = csv.reader(f, delimiter=";")
    
    # choose the right start parameter (first channel)
    for line in reader:
        if line == []:
            continue
        if line[0] == 'Kanal':
            break

    for i, line in enumerate(reader):
        channels_lu.append(int(line[0]))
        # avoid negative counts
        if float(line[1]) < 0.0:
            counts_lu.append(0)
        else:
            counts_lu.append(float(line[1]))

# normalization
new_counts_lu = np.asarray([i/sum(counts_lu) for i in counts_lu])

#savgol filter
winsize_lu, new_counts_lu_smooth, r2_lu  = optimized_smoothing(new_counts_lu)

print('\n---- Lu177m ----')
print('Start Channel: {}, End Channel: {}'.format(channels_lu[0], channels_lu[-1]))
print('CF lu spectrum:', CF_lu)
print('Window size:', winsize_lu)
print('R^2:', round(r2_lu, 3))
print('---------------------')


#%% Iod: pure (reference)spectrum
channels_iod = []
counts_iod = []
bg_iod = []
with open(args.refIod, "r") as f:
    reader = csv.reader(f, delimiter=";")

    # choose the right start parameter (first channel) and set energy2channel CF
    for finder in reader:
        if finder == []:
            continue   
        if finder[0] == 'Kalibrierfaktor:':
            CF_iod = float(finder[1])
        if finder[0] == 'Kanal':
            number = reader.line_num

with open(args.refIod, "r") as f:
    reader = csv.reader(f, delimiter=";")

    # read dataset and fill lists
    for i, line in enumerate(reader):
        if i > number-1:
            channels_iod.append(int(line[0]))
            counts_iod.append(float(line[1]))

            # depends on separate background column
            if len(line) == 2:
                bg_iod = np.linspace(0, 0, len(counts_iod))
            if len(line) == 3:
                bg_iod.append(float(line[2]))

counts_iod = np.asarray(np.subtract(counts_iod, bg_iod))
# avoid negative counts
counts_iod[counts_iod < 0] = 0
# normalization
new_counts_iod = np.asarray([i/sum(counts_iod) for i in counts_iod])

#savgol filter
winsize_iod, new_counts_iod_smooth, r2_iod  = optimized_smoothing(new_counts_iod)

print('\n---- Iod ----')
print('Start Channel: {}, End Channel: {}'.format(channels_iod[0], channels_iod[-1]))
print('CF iod spectrum:', CF_iod)
print('Window size:', winsize_iod)
print('R^2:', round(r2_iod, 3))
print('---------------------')


#%% Mix spectrum
channels_mix = []
counts_mix = []
bg_mix = []
with open(args.sMixture, "r") as f:
    reader = csv.reader(f, delimiter=";")

    # choose the right start parameter (first channel) and set energy2channel CF
    for finder in reader:
        if finder == []:
            continue
        if finder[0] == 'Kalibrierfaktor:':
            CF_mix = float(finder[1])
        if finder[0] == 'Kanal':
            number = reader.line_num

with open(args.sMixture, "r") as f:
    reader = csv.reader(f, delimiter=";")

    # read dataset and fill lists
    for i, line in enumerate(reader):
        if i > number-1:
            channels_mix.append(int(line[0]))
            counts_mix.append(float(line[1]))

            # depends on separate background column
            if len(line) == 2:
                bg_mix = np.linspace(0, 0, len(counts_mix))
            if len(line) == 3:
                bg_mix.append(float(line[2]))

new_counts_mix = np.asarray(np.subtract(counts_mix, bg_mix))

# avoid negative counts
new_counts_mix[new_counts_mix < 0] = 0

#savgol filter
winsize_mix, new_counts_mix_smooth, r2_mix  = optimized_smoothing(new_counts_mix)

print('\n---- Mixture ----')
print('Start Channel: {}, End Channel: {}'.format(channels_mix[0], channels_mix[-1]))
print('CF mix spectrum:', CF_mix)
print('Window size:', winsize_mix)
print('R^2:', round(r2_mix, 3))
print('---------------------')


#%% Converting channels to energies

# converting channels to energies
energy_channels_lu = []
for channel in channels_lu:
    energy = lin(channel, CF_lu, 0)
    energy_channels_lu.append(energy)

energy_channels_iod = []
for channel in channels_iod:
    energy = lin(channel, CF_iod, 0)
    energy_channels_iod.append(energy)

energy_channels_mix = []
for channel in channels_mix:
    energy = lin(channel, CF_mix, 0)
    energy_channels_mix.append(energy)


#%% Optimization
# scipy: minimize

Nfeval = 1

print('\n--- Start Optimization ---')
print('{0:4s}       {1:9s}      {2:9s}       {3:9s}'.format('Iter', ' c_Lu', ' c_Iod', 'obj. Func.'))

# initial values
xinit = np.array([0, 0])

# bounds
bnds = Bounds([0.0, 0.0], [1000000000., 1000000000.])

# optimize minimize
# Unconstrained minimization
res = minimize(fun=obj_func, args=(new_counts_mix_smooth, new_counts_lu_smooth, new_counts_iod_smooth), x0=xinit, method='BFGS',\
               tol=0.0001, callback=callbackF, options={'maxiter':1000 ,'disp': True})

if res.x[0] < 0:
    res.x[0] = 0
elif res.x[1] < 0:
    res.x[1] = 0

# Bound-Constrained minimization 
#res = minimize(fun=obj_func, args=(new_counts_mix_smooth, new_counts_lu_smooth, new_counts_iod_smooth), x0=xinit, method='L-BFGS-B',\
#               bounds=bnds, tol=0.0001, callback=callbackF, options={'maxiter':1000 ,'disp': True})

print('--- End Optimization ---')

print('\n---------------------------\n')

print('Convergence Message:\n', res)


#%% Calculations

### Define specific windows
# Lu boundaries, A window (width: 72)
Lu_peak_A = int(59/CF_lu) #np.argmax(new_counts_lu)
Lu_min_A, Lu_max_A = Lu_peak_A - 36, Lu_peak_A + 36
Lu_width_A = Lu_max_A - Lu_min_A

# Lu boundaries, D window (width: 298)
Lu_peak_D = int(169.5/CF_lu) #np.argmax(new_counts_lu)
Lu_min_D, Lu_max_D = Lu_peak_D - 149, Lu_peak_D + 149
Lu_width_D = Lu_max_D - Lu_min_D

# Iod boundaries, E window (width: 236)
Iod_peak = int(370/CF_iod) #np.argmax(new_counts_iod)
Iod_min, Iod_max = Iod_peak - 117, Iod_peak + 118
Iod_width = Iod_max - Iod_min

### Choose the right calibration factor 
# Lu: depends on cps in window A
cps_lu_winA = sum(counts_lu[Lu_min_A:Lu_max_A]) / dt
cps_lu_winD = sum(counts_lu[Lu_min_D:Lu_max_D]) / dt

lu_factor_A = 12.68
lu_factor_D = 4.81

# =============================================================================
# if cps_lu_winA < 50:
#     lu_factor = 12.68
# elif 50 <= cps_lu_winA < 500:
#     lu_factor = 11.5
# elif 500 <= cps_lu_winA:
#     lu_factor = 10.88
# =============================================================================

# Iod: depends on cps in window E
cps_iod_winE = sum(new_counts_mix_smooth[Iod_min:Iod_max]) / dt

if cps_iod_winE < 100:
    iod_factor = 18.7
elif 100 <= cps_iod_winE < 500:
    iod_factor = 18.2
elif 500 <= cps_iod_winE < 5000:
    iod_factor = 17.7
elif 5000 <= cps_iod_winE < 10000:
    iod_factor = 17.
elif cps_iod_winE >= 10000:
    iod_factor = 16.5


### Calculation of specific activities
# Lu - A window
Lu_act_A = (simps((res.x[0]*new_counts_lu_smooth[Lu_min_A:Lu_max_A]), channels_lu[Lu_min_A:Lu_max_A]) / dt) * lu_factor_A
Lu_act_A2 = (trapezoid((res.x[0]*new_counts_lu_smooth[Lu_min_A:Lu_max_A]), channels_lu[Lu_min_A:Lu_max_A]) / dt) * lu_factor_A

# Lu - D window
Lu_act_D = (simps((res.x[0]*new_counts_lu_smooth[Lu_min_D:Lu_max_D]), channels_lu[Lu_min_D:Lu_max_D]) / dt) * lu_factor_D
Lu_act_D2 = (trapezoid((res.x[0]*new_counts_lu_smooth[Lu_min_D:Lu_max_D]), channels_lu[Lu_min_D:Lu_max_D]) / dt) * lu_factor_D

# Iod - E window
Iod_act = (simps((res.x[1]*new_counts_iod_smooth[Iod_min:Iod_max]), channels_iod[Iod_min:Iod_max]) / dt) * iod_factor
Iod_act2 = (trapezoid((res.x[1]*new_counts_iod_smooth[Iod_min:Iod_max]), channels_iod[Iod_min:Iod_max]) / dt) * iod_factor


### Calculation of the coefficient of determination (R^2)
y_mean = sum(new_counts_mix_smooth)/float(len(new_counts_mix_smooth))
ss_tot = sum((yi-y_mean)**2 for yi in new_counts_mix_smooth)
ss_err = sum((yi-fi)**2 for yi,fi in zip(new_counts_mix_smooth,(res.x[0]*new_counts_lu_smooth+res.x[1]*new_counts_iod_smooth)))
r2 = 1 - (ss_err/ss_tot)


#%% Print results

# print energy windows
print('\n--- Energy Windows ---')
print('Lu - window A: Peak {}, Min {}, Max {}, Width {}'.format(Lu_peak_A, Lu_min_A, Lu_max_A, Lu_width_A))
print('Lu - window D: Peak {}, Min {}, Max {}, Width {}'.format(Lu_peak_D, Lu_min_D, Lu_max_D, Lu_width_D))
print('Iod - window E: Peak {}, Min {}, Max {}, Width {}'.format(Iod_peak, Iod_min, Iod_max, Iod_width))
print('---------------------')

# results of normalization; area should be 1
print('\n--- Normalization ---')
print('Area Lu sopectrum (norm.):', round(simps(new_counts_lu_smooth, channels_lu),3))
print('Area Iod spectrum (norm.):', round(simps(new_counts_iod_smooth, channels_iod),3))
print('---------------------')

# print results of optimizer (c_lu. c_iod) 
print('\n--- Optimized Factors ---')
print('c_Lu:', res.x[0])
print('c_Iod:', res.x[1])
print('-------------------------')

# print the calibration factors (depends on energy windows)
print('\n--- Calibration Factors ---')
print('Lu: counts window A: {} -> CF: {}'.format(round(cps_lu_winA, 1), lu_factor_A))
print('Lu: counts window D: {} -> CF: {}'.format(round(cps_lu_winD, 1), lu_factor_D))
print('Iod: counts window E: {} -> CF: {}'.format(round(cps_iod_winE, 1), iod_factor))
print('-------------------------')

# print specific activities
print('\n--- Calculated Activities ---')
print('Lu activity - window A [Bq]: {} (Simpson)'.format(round(Lu_act_A , 2)))
print('Lu activity - window A [Bq]: {} (Trapezoid)'.format(round(Lu_act_A2 , 2)))
print('Lu activity - window D [Bq]: {} (Simpson)'.format(round(Lu_act_D , 2)))
print('Lu activity - window D [Bq]: {} (Trapezoid)'.format(round(Lu_act_D2 , 2)))
print('Iod activity - window E [Bq]: {} (Simpson)'.format(round(Iod_act, 2)))
print('Iod activity - window E [Bq]: {} (Trapezoid)'.format(round(Iod_act2, 2)))
print('-----------------------------')

# print the coefficient of determination (R^2)
print('\n--- Coefficient Of Determination ---')
print('R^2:', round(r2, 3))
print('------------------------------------')


#%% plots

# define plots
fig = plt.figure(figsize=(16,15))
ax1 = fig.add_subplot(5,1,1)
ax2 = fig.add_subplot(5,1,2)
ax3 = fig.add_subplot(5,1,3)
ax4 = fig.add_subplot(5,1,4)
ax5 = fig.add_subplot(5,1,5)

# Lu Spectrum (normalized)
ax1.plot(np.asarray(energy_channels_lu), new_counts_lu, label='[$^{177m}$Lu]Lu spectrum (normalized)')
ax1.plot(np.asarray(energy_channels_lu), new_counts_lu_smooth, label='[$^{177m}$Lu]Lu spectrum (smooth)')
#ax1.set_xlabel('Energy [keV]')
ax1.set_ylabel('$cts_i$ / $\sum_i(cts_i)$')
ax1.legend()
ax1.text(0.01, 0.85, 'A', transform=ax1.transAxes, size=20, weight='bold')

# Iod Spectrum (normalized)
ax2.plot(np.asarray(energy_channels_iod), new_counts_iod, label='[$^{131}$I]I spectrum (normalized)')
ax2.plot(np.asarray(energy_channels_iod), new_counts_iod_smooth, label='[$^{131}$I]I spectrum (smooth)')
#ax2.set_xlabel('Energy [keV]')
ax2.set_ylabel('$cts_i$ / $\sum_i(cts_i)$')
ax2.legend()
ax2.text(0.01, 0.85, 'B', transform=ax2.transAxes, size=20, weight='bold')

# Mixture (measured)
ax3.plot(np.asarray(energy_channels_mix), new_counts_mix, label='Mixture (measured)')
ax3.plot(np.asarray(energy_channels_mix), new_counts_mix_smooth, label='Mixture (smooth)')
#ax3.set_xlabel('Energy [keV]')
ax3.set_ylabel('cts')
ax3.legend()
ax3.text(0.01, 0.85, 'C', transform=ax3.transAxes, size=20, weight='bold')

# Lu + Iod (Calculated)
ax4.plot(np.asarray(energy_channels_lu), res.x[0]*new_counts_lu_smooth, color='red', label='[$^{177m}$Lu]Lu spectrum (calculated+smooth)')
ax4.plot(np.asarray(energy_channels_iod), res.x[1]*new_counts_iod_smooth, color='green', label='[$^{131}$I]I spectrum (calculated+smooth)')
ax4.plot(np.asarray(energy_channels_mix), (res.x[0]*new_counts_lu_smooth+res.x[1]*new_counts_iod_smooth), color='black', label='[$^{177m}$Lu]Lu + [$^{131}$I]I (calculated)')
#ax4.set_xlabel('Energy [keV]')
ax4.set_ylabel('cts')
ax4.legend()
ax4.text(0.01, 0.85, 'D', transform=ax4.transAxes, size=20, weight='bold')

# Diff
ax5.plot(np.asarray(energy_channels_mix), new_counts_mix_smooth, label='Mixture (smooth)')
ax5.plot(np.asarray(energy_channels_mix), (res.x[0]*new_counts_lu_smooth+res.x[1]*new_counts_iod_smooth), color='black', label='[$^{177m}$Lu]Lu + [$^{131}$I]I (calculated)')
ax5.plot(np.asarray(energy_channels_mix), (new_counts_mix_smooth-(res.x[0]*new_counts_lu_smooth+res.x[1]*new_counts_iod_smooth)), color='red', label='Difference')
ax5.set_xlabel('Energy [keV]')
ax5.set_ylabel('cts')
ax5.legend()
ax5.text(0.01, 0.85, 'E', transform=ax5.transAxes, size=20, weight='bold')
