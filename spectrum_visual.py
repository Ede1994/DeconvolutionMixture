#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Eric Einspänner

Spectrum deconvolution for 2 components: Lu177m, I131

This program is free software.
"""
import os
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize, Bounds
from scipy.signal import savgol_filter
from scipy.integrate import simps, trapezoid

#%% constants
# Lu177m energy window D (width: 298)
dict_Lu177m_winD = {'Lu177m_peak': 362,
                    'Lu177m_min' : 213,
                    'Lu177m_max' : 511,
                    'Lu177m_width': 298}

# I131 energy window E (width: 235)
dict_I131_winE =    {'I131_peak': 802,
                    'I131_min' : 685,
                    'I131_max' : 920,
                    'I131_width': 235}


#%% data path and files
py_path = os.getcwd()

data_path_lu177 = py_path + '/Data/Reference/AWM_Lu177_300Bq_3600s.csv'
data_path_lu177m = py_path + '/Data/Reference/AWM_Lu177m_7000Bq_3600s_040221.csv'
data_path_iod = py_path + '/Data/Reference/AWM_I131_7000Bq_3600s_170221.csv'

data_path_mix = py_path + '/Data/Mix_sample1/AWM_MIX_100vs100_3600s.csv'

#%%
def replace_and_convert(item: str, replace_map: dict={';': '', ':': ''}) -> float:
    '''
    Replace/delete special chars and then convert it to float
    '''
    try: 
        for k, v in replace_map.items():
            item = item.replace(k, v)
    except TypeError as e:
        print(e)
    try:
        return float(item)
    except ValueError:
        return np.NaN

def optimized_smoothing(df_col):
    '''
    Definition of a smoothing filter (Savitzky-Golay filter)
    '''
    # create array of df column
    arr = df_col.to_numpy()

    # define winsize (must be larger than polynomial order)
    winsize = 7

    # while loop to find "optimal" winsize
    while winsize < 100:
        # Savitzky-Golay filter for smoothing: array, window size, polynomial order
        smooth_arr = savgol_filter(arr, winsize, 2)

        # calculate R^2
        y_mean = sum(arr)/float(len(arr))
        ss_tot = sum((yi-y_mean)**2 for yi in arr)
        ss_err = sum((yi-fi)**2 for yi,fi in zip(arr, smooth_arr))
        r2 = 1 - (ss_err/ss_tot)
    
        # define abort criterion
        if r2 > 0.91:
            break
        elif winsize >= 15:
            break
        else:
            winsize += 2
        
    return smooth_arr, winsize, r2

def obj_func(x, counts_mix_smooth, counts_lu_smooth, counts_iod_smooth):
    '''
    objective function: Least square
    x is the array containing the wanted coefficients: c_lu, c_iod
    '''
    y_pred = (x[0] * counts_lu_smooth) + (x[1] * counts_iod_smooth)
    return np.sum((counts_mix_smooth - y_pred)**2)

def callbackF(x):
    '''
    callback function for more scipy.optimize.minimize infos
    '''
    global Nfeval
    print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}'.format(Nfeval, x[0], x[1], obj_func(x, df_mix_data['Smoothed'], df_Lu177m_data['Smoothed'], df_I131_data['Smoothed'])))
    Nfeval += 1

def calc_activity(df_data, min_ch, max_ch, c, dt, CF):
    '''
    Calculate the activity (integral under the curve)
    '''
    _simps = simps(df_data['Smoothed'].mul(c).iloc[min_ch-1:max_ch],
                    df_data.index[min_ch-1:max_ch]) / dt * CF
    _trapezoid = trapezoid(df_data['Smoothed'].mul(c).iloc[min_ch-1:max_ch],
                    df_data.index[min_ch-1:max_ch]) / dt * CF
    return _simps, _trapezoid

def abw_spektr(name, df_path):
    '''
    Extract header informations and spectrum datat from csv-file
    '''
    # read the csv data set from waster water measuring station
    df = pd.read_csv(df_path, encoding = "ISO-8859-1", delimiter=':;', engine='python', header=None, on_bad_lines='skip')
    col_names = list(df.columns)

    # extract the CF factor from "header" (convert channels -> energies) and convert it to float
    CF = replace_and_convert(df.loc[df[col_names[0]] == 'Kalibrierfaktor', col_names[1]].iloc[0]) #Messzeit Spektrum (s):

    # extract the measuring time
    dt = replace_and_convert(df.loc[df[col_names[0]] == 'Messzeit Spektrum (s)', col_names[1]].iloc[0])

    # get the index where the spectrum data begins
    idx = df.index[df[col_names[0]] == ';Spektrum;Nullspektrum']

    # start to load the spectrum data, set col names and split
    df_data = df.iloc[idx[0]:, 0].str.split(';', expand=True)
    df_data = df_data.reset_index(drop=True)
    df_data.columns = df_data.iloc[0]
    df_data = df_data.drop([0,1])

    # If zero spectrum has not yet been taken into account, then subtract it from the measured counts,
    # otherwise this has already been done automatically by the measuring station.
    if 'Nullspektrum' in col_names:
        df_data['Total'] = df_data['Spektrum'].astype(float) - df_data['Nullspektrum'].astype(float)
    else:
        df_data['Total'] = df_data['Spektrum'].astype(float)
    
    # set negative values to zero
    df_data[df_data['Total'] < 0] = 0

    # normalize
    df_data['Normalize'] = df_data['Total'] / df_data['Total'].sum()

    # smoothed
    df_data['Smoothed'], winsize, r2 = optimized_smoothing(df_data['Total'])

    print('\n---- {} ----'.format(name))
    print('CF spectrum {}: {}'.format(name, CF))
    print('Window size:', winsize)
    print('R^2:', round(r2, 3))
    print('---------------------')

    return df_data, CF, dt

def CF_lu177m_winD():
    '''
    Choose the correct CF factor for Lu177m in energy window D (depends on cts)
    '''
    lu177m_factor = 4.81

    return lu177m_factor

def CF_iod_winE(cps):
    '''
    Choose the correct CF factor for I131 in energy window E (depends on cts)
    '''
    if cps < 100:
        iod_factor = 18.7
    elif 100 <= cps < 500:
        iod_factor = 18.2
    elif 500 <= cps < 5000:
        iod_factor = 17.7
    elif 5000 <= cps < 10000:
        iod_factor = 17.
    elif cps >= 10000:
        iod_factor = 16.5

    return iod_factor

#%% Lu177 and Lu177m: pure (reference-)spectrum
df_Lu177_data, CF_Lu177, dt_Lu177 = abw_spektr('Lu177', data_path_lu177)
df_Lu177m_data, CF_Lu177m, dt_Lu177m = abw_spektr('Lu177m', data_path_lu177m)
df_I131_data, CF_I131, dt_I131 = abw_spektr('I131', data_path_iod)
df_mix_data, CF_mix, dt_mix = abw_spektr('mix', data_path_mix)

#%% Optimization
Nfeval = 1
# initial values
xinit = np.array([0, 0])
# bounds
bnds = Bounds([0.0, 0.0], [1000000000., 1000000000.])

print('\n--- Start Optimization ---')
print('{0:4s}       {1:9s}      {2:9s}       {3:9s}'.format('Iter', ' c_Lu', ' c_Iod', 'obj. Func.'))

# optimize minimize
# Unconstrained minimization
res = minimize(fun=obj_func, args=(df_mix_data['Smoothed'], df_Lu177m_data['Smoothed'], df_I131_data['Smoothed']), x0=xinit, method='BFGS',\
               tol=0.0001, callback=callbackF, options={'maxiter':1000 ,'disp': True})
# Bound-Constrained minimization 
#res = minimize(fun=obj_func, args=(new_counts_mix_smooth, new_counts_lu_smooth, new_counts_iod_smooth), x0=xinit, method='L-BFGS-B',\
#               bounds=bnds, tol=0.0001, callback=callbackF, options={'maxiter':1000 ,'disp': True})

# if factors smaller than 0 are set equal to zero
if res.x[0] < 0:
    res.x[0] = 0
if res.x[1] < 0:
    res.x[1] = 0

print('--- End Optimization ---')
print('\n---------------------------\n')
print('Convergence Message:\n', res)

#%% 
cps_iod_winE = df_I131_data['Smoothed'].mul(res.x[1]).iloc[dict_I131_winE['I131_min']-1:dict_I131_winE['I131_max']].sum() / dt_I131
iod_factor = CF_iod_winE(cps_iod_winE)

# Iod_act_simps = simps(df_I131_data['Smoothed'].mul(res.x[1]).iloc[dict_I131_winE['I131_min']-1:dict_I131_winE['I131_max']], df_I131_data.index[dict_I131_winE['I131_min']-1:dict_I131_winE['I131_max']]) / dt_I131 * iod_factor
# Iod_act_trapezoid = trapezoid(df_I131_data['Smoothed'].mul(res.x[1]).iloc[dict_I131_winE['I131_min']-1:dict_I131_winE['I131_max']], df_I131_data.index[dict_I131_winE['I131_min']-1:dict_I131_winE['I131_max']]) / dt_I131 * iod_factor



Iod_act_simps, Iod_act_trapezoid = calc_activity(df_I131_data, dict_I131_winE['I131_min'], dict_I131_winE['I131_max'], res.x[1], dt_I131, iod_factor)

#%% print results
# print specific activities
print('\n--- Calculated Activities ---')
# print('Lu activity - window A [Bq]: {} (Simpson)'.format(round(Lu_act_A , 2)))
# print('Lu activity - window A [Bq]: {} (Trapezoid)'.format(round(Lu_act_A2 , 2)))
# print('Lu activity - window D [Bq]: {} (Simpson)'.format(round(Lu_act_D , 2)))
# print('Lu activity - window D [Bq]: {} (Trapezoid)'.format(round(Lu_act_D2 , 2)))
print('Iod activity - window E [Bq]: {} (Simpson)'.format(round(Iod_act_simps, 2)))
print('Iod activity - window E [Bq]: {} (Trapezoid)'.format(round(Iod_act_trapezoid, 2)))
print('-----------------------------')

#%% plots

# define plots
fig1 = plt.figure(figsize=(16,15))
ax1 = fig1.add_subplot(4,1,1)
ax2 = fig1.add_subplot(4,1,2)
ax3 = fig1.add_subplot(4,1,3)
ax4 = fig1.add_subplot(4,1,4)

# 
ax1.plot(df_Lu177_data.index * CF_Lu177, df_Lu177_data['Normalize'], label='[$^{177}$Lu]Lu spectrum')
ax1.plot(df_Lu177m_data.index * CF_Lu177m, df_Lu177m_data['Normalize'], label='[$^{177m}$Lu]Lu spectrum')
ax1.plot(df_Lu177_data.index * CF_Lu177, df_Lu177m_data['Normalize']-df_Lu177_data['Normalize'], label='[$^{177}$Lu]Lu - [$^{177m}$Lu]Lu spectrum')
ax1.set_ylabel('$cts_i$ / $\sum_i(cts_i)$', fontsize=16)
ax1.set_xticks(range(0,1000,50))
ax1.legend(fancybox=True, framealpha=0.1, fontsize = 'large')
ax1.text(0.01, 0.85, 'A', transform=ax1.transAxes, size=20, weight='bold')

# 
ax2.plot(df_I131_data.index * CF_I131, df_I131_data['Normalize'], label='[$^{131}$I]I spectrum')
ax2.set_ylabel('$cts_i$ / $\sum_i(cts_i)$', fontsize=16)
ax2.set_xticks(range(0,1000,50))
ax2.legend(fancybox=True, framealpha=0.1, fontsize = 'large')
ax2.text(0.01, 0.85, 'B', transform=ax2.transAxes, size=20, weight='bold')

#
ax3.plot(df_mix_data.index * CF_mix, df_mix_data['Total'], label='mix spectrum')
ax3.plot(df_mix_data.index * CF_mix, df_mix_data['Smoothed'], label='smoothed spectrum')
ax3.set_ylabel('cts', fontsize=16)
ax3.set_xticks(range(0,1000,50))
ax3.legend(fancybox=True, framealpha=0.1, fontsize = 'large')
ax3.text(0.01, 0.85, 'C', transform=ax3.transAxes, size=20, weight='bold')

# Lu + Iod (Calculated)
ax4.plot(df_Lu177m_data.index * CF_Lu177m, df_Lu177m_data['Smoothed'].mul(res.x[0]), color='red', label='[$^{177m}$Lu]Lu Spektrum (berechnet+geglättet)')
ax4.plot(df_I131_data.index * CF_I131, df_I131_data['Smoothed'].mul(res.x[1]), color='green', label='[$^{131}$I]I Spektrum (berechnet+geglättet)')
ax4.plot(df_mix_data.index * CF_mix, (df_Lu177m_data['Smoothed'].mul(res.x[0])+df_I131_data['Smoothed'].mul(res.x[1])), color='black', label='[$^{177m}$Lu]Lu + [$^{131}$I]I')
ax4.set_xlabel('Energy [keV]')
ax4.set_xticks(range(0,1000,50))
ax4.set_ylabel('cts', fontsize=16)
ax4.legend(fancybox=True, framealpha=0.1, fontsize = 'large')
ax4.text(0.01, 0.85, 'D', transform=ax4.transAxes, size=20, weight='bold')

# show plot
plt.show()