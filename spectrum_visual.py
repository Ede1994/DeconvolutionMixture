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


#%% data path and files
py_path = os.getcwd()

data_path_lu177 = py_path + '/Data/Reference/AWM_Lu177_300Bq_3600s.csv'
data_path_lu177m = py_path + '/Data/Reference/AWM_Lu177m_7000Bq_3600s_040221.csv'
data_path_iod = py_path + '/Data/Reference/AWM_I131_7000Bq_3600s_170221.csv'

data_path_mix = py_path + '/Data/Mix_sample1/AWM_MIX_100vs100_3600s.csv'

#%% 
def abw_spektr(df_path):
    '''
    Extract header informations and spectrum datat from csv-file
    '''
    #
    df = pd.read_csv(df_path, encoding = "ISO-8859-1", delimiter=':;', engine='python', on_bad_lines='skip')
    col_names = list(df.columns)

    #
    if ';' in df.loc[df[col_names[0]] == 'Kalibrierfaktor', col_names[1]].iloc[0]:
        CF = float(df.loc[df[col_names[0]] == 'Kalibrierfaktor', col_names[1]].iloc[0].replace(';',''))
    else:
        CF = float(df.loc[df[col_names[0]] == 'Kalibrierfaktor', col_names[1]].iloc[0])

    idx = df.index[df[col_names[0]] == ';Spektrum;Nullspektrum']

    # 
    df_data = df.iloc[idx[0]:, 0].str.split(';', expand=True)
    df_data = df_data.reset_index(drop=True)
    df_data.columns = df_data.iloc[0]
    df_data = df_data.drop([0,1])

    # 
    if 'Nullspektrum' in col_names:
        df_data['Total'] = df_data['Spektrum'].astype(float) - df_data['Nullspektrum'].astype(float)
    else:
        df_data['Total'] = df_data['Spektrum'].astype(float)

    # normalize
    df_data['Normalize'] = df_data['Total'] / df_data['Total'].sum()

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
#ax1.plot(df_mix_data.index * CF_mix, df_mix_data['Normalize'], label='mix spectrum')
ax1.set_ylabel('$cts_i$ / $\sum_i(cts_i)$', fontsize=16)
ax1.legend(fancybox=True, framealpha=0.1, fontsize = 'large')
ax1.text(0.01, 0.85, 'A', transform=ax1.transAxes, size=20, weight='bold')

# 
ax2.plot(df_I131_data.index * CF_I131, df_I131_data['Normalize'], label='[$^{131}$I]I spectrum')
ax2.set_ylabel('$cts_i$ / $\sum_i(cts_i)$', fontsize=16)
ax2.legend(fancybox=True, framealpha=0.1, fontsize = 'large')
ax2.text(0.01, 0.85, 'B', transform=ax2.transAxes, size=20, weight='bold')

#
ax3.plot(df_mix_data.index * CF_mix, df_mix_data['Total'], label='mix spectrum')
ax3.set_ylabel('$cts_i$ / $\sum_i(cts_i)$', fontsize=16)
ax3.set_xlabel('Energy [keV]', fontsize=16)
ax3.legend(fancybox=True, framealpha=0.1, fontsize = 'large')
ax3.text(0.01, 0.85, 'C', transform=ax3.transAxes, size=20, weight='bold')

# show plot
plt.show()