#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Eric Einspänner

GUI, Spectrum deconvolution for 2 components: Lu177m, I131

This program is free software.
"""
import re
import os
import csv

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from scipy.optimize import minimize, Bounds
from scipy.integrate import simps
from scipy.signal import savgol_filter

import tkinter as tk
from tkinter import filedialog

#%% Definitions

### iteration counter
Nfeval = 1

### definition of figures
fig = Figure(figsize=(16, 15), dpi=60)
ax1 = fig.add_subplot(5,1,1)
ax2 = fig.add_subplot(5,1,2)
ax3 = fig.add_subplot(5,1,3)
ax4 = fig.add_subplot(5,1,4)
ax5 = fig.add_subplot(5,1,5)

### Define specific windows
# Lu boundaries, A window (width: 72)
Lu_peak = 126 #np.argmax(new_counts_lu)
Lu_min, Lu_max = Lu_peak - 36, Lu_peak + 36

# Iod boundaries, E window (width: 236)
Iod_peak = 791 #np.argmax(new_counts_iod)
Iod_min, Iod_max = Iod_peak - 118, Iod_peak + 118


#%% Functions

# function for sorting lists, containing strings with numbers (https://stackoverflow.com/a/48413757)
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key, reverse=False)

# optimize smoothing filter
def optimized_smoothing(counts_arr):
    winsize = 5 # must be larger than polynomial order
    for num in range(1000):
        counts_smooth_arr = savgol_filter(counts_arr, winsize, 3) # window size, polynomial order

        y_mean = sum(counts_arr)/float(len(counts_arr))
        ss_tot = sum((yi-y_mean)**2 for yi in counts_arr)
        ss_err = sum((yi-fi)**2 for yi,fi in zip(counts_arr, counts_smooth_arr))
        r2 = 1 - (ss_err/ss_tot)
    
        if r2 > 0.991:
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


#%% Read reference (pure) spectrums

# reference (pure) spectrum for iod and lu
py_path = os.getcwd()

data_path_lu = py_path + '/Data/Lu/10000Bq_20200916_300s.csv'
data_path_iod = py_path + '/Data/Iod/1000Bq_20201007_300s.csv'

### Lu: pure (reference)spectrum
channels_lu = []
counts_lu = []
with open(data_path_lu, "r") as f:
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

### Iod: pure (reference)spectrum
channels_iod = []
counts_iod = []
with open(data_path_iod, "r") as f:
    reader = csv.reader(f, delimiter=";")

    # choose the right start parameter (first channel)
    for line in reader:
        if line == []:
            continue
        if line[0] == 'Kanal':
            break

    for i, line in enumerate(reader):
        channels_iod.append(int(line[0]))
        # avoid negative counts
        if float(line[1]) < 0.0:
            counts_iod.append(0)
        else:
            counts_iod.append(float(line[1]))

# normalization
new_counts_iod = np.asarray([i/sum(counts_iod) for i in counts_iod])

#savgol filter
new_counts_iod_smooth = savgol_filter(new_counts_iod, 11, 3)


#%% Functions for GUI

# load patdata and measures from file
def load_data(file):
    global channels_mix, counts_mix, bg_mix, new_counts_mix, new_counts_mix_smooth

    # Mix spectrum
    channels_mix = []
    counts_mix = []
    bg_mix = []
    with open(file, "r") as f:
        reader = csv.reader(f, delimiter=";")
        
        # choose the right start parameter (first channel)
        for line in reader:
            if line == []:
                continue
            if line[0] == 'Kanal':
                break

        # read dataset and fill lists
        for i, line in enumerate(reader):
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


# spectrum deconvolution
def spectrum_deconv():
    dt = int(entry_dt.get())
    # scipy: minimize

    print('\n--- Start Optimization ---')
    print('{0:4s}       {1:9s}      {2:9s}       {3:9s}'.format('Iter', ' c_Lu', ' c_Iod', 'obj. Func.'))

    # initial values
    xinit = np.array([0, 0])

    # bounds
    bnds = Bounds([0.0, 0.0], [10000000., 10000000.])

    # optimize minimize 
    res = minimize(fun=obj_func, args=(new_counts_mix_smooth, new_counts_lu_smooth, new_counts_iod_smooth), x0=xinit, method='L-BFGS-B',\
                   bounds=bnds, tol=0.001, callback=callbackF, options={'maxiter':2000 ,'disp': True})

    print('---------------------------')
    
    global c_Lu, c_Iod
    c_Lu = res.x[0]
    c_Iod = res.x[1]

    ### Choose the right calibration factor 
    # Lu: depends on cps in window A
    cps_lu_winA = (sum(counts_lu[Lu_min:Lu_max])/300.)

    if cps_lu_winA < 50:
        lu_factor = 12.68
    elif 50 <= cps_lu_winA < 500:
        lu_factor = 11.5
    elif 500 <= cps_lu_winA:
        lu_factor = 10.88

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
    Lu_act = round((simps((res.x[0]*new_counts_lu_smooth)[Lu_min:Lu_max], channels_lu[Lu_min:Lu_max]) / dt) * lu_factor, 2)
    Iod_act = round((simps((res.x[1]*new_counts_iod_smooth)[Iod_min:Iod_max], channels_iod[Iod_min:Iod_max]) / dt) * iod_factor, 2)

    ### Calculation of the coefficient of determination (R^2)
    y_mean = sum(new_counts_mix_smooth)/float(len(new_counts_mix_smooth))
    ss_tot = sum((yi-y_mean)**2 for yi in new_counts_mix_smooth)
    ss_err = sum((yi-fi)**2 for yi,fi in zip(new_counts_mix_smooth,(res.x[0]*new_counts_lu_smooth+res.x[1]*new_counts_iod_smooth)))
    r2 = round(1 - (ss_err/ss_tot), 3)

    ### results; add in label areas
    label_areaCalcActivityLu177m.config(text=str(Lu_act))
    label_areaCalcActivityI131.config(text=str(Iod_act))
    label_areaR2.config(text=str(r2))

    ### converting channels to energies, for plots
    energy_channels_lu = []
    for channel in channels_lu:
        energy = lin(channel, 0.46079, 0)
        energy_channels_lu.append(energy)

    energy_channels_iod = []
    for channel in channels_iod:
        energy = lin(channel, 0.46079, 0)
        energy_channels_iod.append(energy)

    energy_channels_mix = []
    for channel in channels_iod:
        energy = lin(channel, 0.46079, 0)
        energy_channels_mix.append(energy)

    ### Define plots    
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()
    ax5.clear()
    
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
    ax3.plot(np.asarray(energy_channels_mix), new_counts_mix, label='Mixture (measured)')
    ax3.plot(np.asarray(energy_channels_mix), new_counts_mix_smooth, label='Mixture (smooth)')
    ax3.set_ylabel('counts')
    ax3.legend()

    # Lu + Iod (Calculated)
    ax4.plot(np.asarray(energy_channels_lu), c_Lu*new_counts_lu_smooth, color='red', label='Lu (calculated+smooth)')
    ax4.plot(np.asarray(energy_channels_iod), c_Iod*new_counts_iod_smooth, color='green', label='Iod (calculated+smooth)')
    ax4.plot(np.asarray(energy_channels_mix), (c_Lu*new_counts_lu_smooth+c_Iod*new_counts_iod_smooth), color='black', label='Lu + Iod (smooth)')
    ax4.set_ylabel('counts')
    ax4.legend()

    # Diff
    ax5.plot(np.asarray(energy_channels_mix), new_counts_mix_smooth, label='Mixture (smooth)')
    ax5.plot(np.asarray(energy_channels_mix), (c_Lu*new_counts_lu_smooth+c_Iod*new_counts_iod_smooth), color='black', label='Lu + Iod (smooth)')
    ax5.plot(np.asarray(energy_channels_mix), (new_counts_mix_smooth-(c_Lu*new_counts_lu_smooth+c_Iod*new_counts_iod_smooth)), color='red', label='Diff')
    ax5.set_xlabel('Energy (keV)')
    ax5.set_ylabel('counts')
    ax5.legend()
    
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=4, rowspan=6, columnspan=5, sticky=tk.W + tk.E + tk.N + tk.S, padx=1, pady=1)


#%% Buttons

# do nothing button
def donothing():
    filewin = tk.Toplevel(root)
    button = tk.Button(filewin,
                       text='''Do nothing button'''
                       )
    button.pack()


# text for help
def helps():
    filewin = tk.Toplevel(root)
    filewin.title("Help")
    S = tk.Scrollbar(filewin)
    T = tk.Text(filewin, height=10, width=100)
    S.pack(side=tk.RIGHT , fill=tk.Y)
    T.pack(side=tk.LEFT, fill=tk.Y)
    S.config(command=T.yview)
    T.config(yscrollcommand=S.set)
    quote = '''For descriptions around the program read help.txt please! Or follow us on GitHub.'''
    T.insert(tk.END, quote)


# text for impressum
def impressum():
    filewin = tk.Toplevel(root)
    filewin.title("Impressum")
    S = tk.Scrollbar(filewin)
    T = tk.Text(filewin, height=10, width=100)
    S.pack(side=tk.RIGHT , fill=tk.Y)
    T.pack(side=tk.LEFT, fill=tk.Y)
    S.config(command=T.yview)
    T.config(yscrollcommand=S.set)
    quote = '''Author: Eric Einspänner, Clinic of Radiology and Nuclear Medicine, UMMD (Germany)
This software is distributed under an open source license, see LICENSE.txt for details.'''
    T.insert(tk.END, quote)


# load file from explorer
def buttonImport():
    # open dialog
    file = filedialog.askopenfile(title="Load data...", mode='r', filetypes=[("All files", "*.*")])

    # extract filename as str
    filename = str(file.name)

    # insert    
    entry_spectrum.insert(tk.END, str(filename))

    # call load data function
    load_data(filename)
    

# load file from explorer
def button_SpectrumDeconvolution():
    spectrum_deconv()


#%% GUI

root = tk.Tk()
root.title("Spectrum Deconvolution")
root.geometry("1920x1080")

### define menu
menubar = tk.Menu(root)

# file menu
filemenu = tk.Menu(menubar, tearoff=0)
filemenu.add_command(label="Import...", command=buttonImport)
filemenu.add_separator()
filemenu.add_command(label="Save as...", command=donothing)
filemenu.add_separator()
filemenu.add_command(label="Exit", command=root.quit)
menubar.add_cascade(label="File", menu=filemenu)

# edit menu
editmenu = tk.Menu(menubar, tearoff=0)
editmenu.add_command(label="Delete All", command=donothing)
menubar.add_cascade(label="Edit", menu=editmenu)

# help menu
helpmenu = tk.Menu(menubar, tearoff=0)
helpmenu.add_command(label="Help Index", command=helps)
helpmenu.add_command(label="Impressum", command=impressum)
menubar.add_cascade(label="Help", menu=helpmenu)

root.config(menu=menubar)

### Buttons, Labels and entries
# Import button
buttonImport = tk.Button(text='Import Data', width='10', bg='green', command=buttonImport)
buttonImport.grid(row=0, column=0, padx='5', pady='5')

# spectrum path
label_spectrum = tk.Label(root, text="Path Spectrum:").grid(row=1)
entry_spectrum = tk.Entry(root)
entry_spectrum.grid(row=1, column=1, ipadx=100, padx=15)

# measuring time
label_dt = tk.Label(root, text="Measuring Time [s]:").grid(row=2)
entry_dt = tk.Entry(root)
entry_dt.grid(row=2, column=1, padx=15, sticky=tk.W)
entry_dt.insert(10, int(3600))

# Calculation button
buttonImport = tk.Button(text='Spectrum Deconvolution', width='20', bg='yellow', command=button_SpectrumDeconvolution)
buttonImport.grid(row=3, column=0, padx='5', pady='5')

# results
label_CalcActivityLu177m = tk.Label(root, text="Calculated Activity Lu177m (Bq):").grid(row=4, column=0)
label_areaCalcActivityLu177m = tk.Label(root, bg='gray', width='12', text="")
label_areaCalcActivityLu177m.grid(row=4, column=1)

label_CalcActivityI131 = tk.Label(root, text="Calculated Activity I131 (Bq):").grid(row=5, column=0)
label_areaCalcActivityI131 = tk.Label(root, bg='gray', width='12', text="")
label_areaCalcActivityI131.grid(row=5, column=1)

label_R2 = tk.Label(root, text="R^2:").grid(row=6, column=0)
label_areaR2 = tk.Label(root, bg='gray', width='12', text="")
label_areaR2.grid(row=6, column=1)

# plot areas
plot_frame = tk.Frame(width=600, height=500, bg="grey", colormap="new")
plot_frame.grid(row=0, column=4, rowspan=6, columnspan=5, sticky=tk.W + tk.E + tk.N + tk.S, padx=1, pady=1)

# run GUI
root.mainloop()