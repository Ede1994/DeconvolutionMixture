#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Eric Einspänner

Spectrum visualizing

This program is free software.
"""
import os
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter


#%% data path and files
py_path = os.getcwd()

data_path_lu177 = py_path + '/Data/Reference/AWM_Lu177_300Bq_3600s.csv'
data_path_lu177m = py_path + '/Data/Reference/AWM_Lu177m_7000Bq_3600s_040221.csv'
data_path_iod = py_path + '/Data/Reference/AWM_I131_7000Bq_3600s_170221.csv'

data_path_mix = py_path + '/Data/Mix_sample1/AWM_MIX_100vs100_3600s.csv'

#%%
def replace_and_convert(item: str, replace_map: dict={';': '', ':': ''}) -> float:
    '''
    Replace special chars with '' and then convert it to float
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
        
    return smooth_arr

def abw_spektr(df_path):
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
    df_data['Smoothed'] = optimized_smoothing(df_data['Total'])

    return df_data, CF

#%% Lu177 and Lu177m: pure (reference-)spectrum
df_Lu177_data, CF_Lu177 = abw_spektr(data_path_lu177)
df_Lu177m_data, CF_Lu177m = abw_spektr(data_path_lu177m)
df_I131_data, CF_I131 = abw_spektr(data_path_iod)
df_mix_data, CF_mix = abw_spektr(data_path_mix)

#%% plots

# define plots
fig1 = plt.figure(figsize=(16,15))
ax1 = fig1.add_subplot(3,1,1)
ax2 = fig1.add_subplot(3,1,2)
ax3 = fig1.add_subplot(3,1,3)

# 
ax1.plot(df_Lu177_data.index * CF_Lu177, df_Lu177_data['Normalize'], label='[$^{177}$Lu]Lu spectrum')
ax1.plot(df_Lu177_data.index * CF_Lu177m, df_Lu177m_data['Normalize'], label='[$^{177m}$Lu]Lu spectrum')
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
ax3.set_ylabel('$cts_i$ / $\sum_i(cts_i)$', fontsize=16)
ax3.set_xticks(range(0,1000,50))
ax3.set_xlabel('Energy [keV]', fontsize=16)
ax3.legend(fancybox=True, framealpha=0.1, fontsize = 'large')
ax3.text(0.01, 0.85, 'C', transform=ax3.transAxes, size=20, weight='bold')

# show plot
plt.show()