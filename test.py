# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 08:36:38 2021

@author: Eric
"""

import re
import os
import csv

import numpy as np
from scipy.optimize import minimize, Bounds
from scipy.integrate import simps
from scipy.signal import savgol_filter

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

### Fonts
LARGE_FONT= ("Verdana", 12)
NORM_FONT = ("Helvetica", 10)
SMALL_FONT = ("Helvetica", 8)

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

# spectrum deconvolution
def spectrum_deconv():
    dt = 3600.
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
    fig = Figure(figsize=(16, 15), dpi=60)    
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

#%% Read reference (pure) spectrums

# reference (pure) spectrum for iod and lu
py_path = os.getcwd()

data_path_lu = py_path + '/Data/Reference/AWM_Lu177m_10000Bq_300s_160920.csv'
data_path_iod = py_path + '/Data/Reference/AWM_I131_7000Bq_3600s_170221.csv'

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

# Iod: pure (reference)spectrum
channels_iod = []
counts_iod = []
bg_iod = []
with open(data_path_iod, "r") as f:
    reader = csv.reader(f, delimiter=";")

    # choose the right start parameter (first channel)
    for line in reader:
        if line == []:
            continue
        if line[0] == 'Kanal':
            break

    # read dataset and fill lists
    for i, line in enumerate(reader):
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

#%% Definitions for GUI

def popupmsg(msg):
    popup = tk.Tk()
    popup.wm_title("!")
    label = ttk.Label(popup, text=msg, font=NORM_FONT)
    label.pack(side="top", fill="x", pady=10)
    B1 = ttk.Button(popup, text="Okay", command = popup.destroy)
    B1.pack()
    popup.mainloop()

def Help():
    popup = tk.Tk()
    popup.wm_title("Help")
    label = ttk.Label(popup, text='''For descriptions around the program read help.txt please! Or follow us on GitHub.''', font=NORM_FONT)
    label.pack(side="top", fill="x", pady=10)
    B1 = ttk.Button(popup, text="Okay", command = popup.destroy)
    B1.pack()
    popup.mainloop()

def Impressum():
    popup = tk.Tk()
    popup.wm_title("Impressum")
    label = ttk.Label(popup, text='''Author: Eric Einspaenner, Clinic of Radiology and Nuclear Medicine, UMMD (Germany)
This software is distributed under an open source license, see LICENSE.txt for details.''', font=NORM_FONT)
    label.pack(side="top", fill="x", pady=10)
    B1 = ttk.Button(popup, text="Okay", command = popup.destroy)
    B1.pack()
    popup.mainloop()

#%% Spectrum Conv classes
class SpectrumDeconv(tk.Tk):

    def __init__(self, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)

        tk.Tk.iconbitmap(self,default="tmp/nuclear.ico")
        tk.Tk.wm_title(self, "Spectrum Deconvolution")

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand = True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        
        ### define menu
        menubar = tk.Menu(container)

        # file menu
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Import...", command = lambda: popupmsg("Not supported just yet!"))
        filemenu.add_separator()
        filemenu.add_command(label="Save as...", command = lambda: popupmsg("Not supported just yet!"))
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=container.quit)
        menubar.add_cascade(label="File", menu=filemenu)

        # edit menu
        editmenu = tk.Menu(menubar, tearoff=0)
        editmenu.add_command(label="Delete All", command = lambda: popupmsg("Not supported just yet!"))
        menubar.add_cascade(label="Edit", menu=editmenu)
        
        # help menu
        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="Help", command = Help)
        helpmenu.add_command(label="Impressum", command = Impressum)
        menubar.add_cascade(label="Help", menu=helpmenu)

        tk.Tk.config(self, menu=menubar)

        self.frames = {}

        for F in (StartPage, Nuclids_2, Nuclids_3):

            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    # Button: show frame
    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()
    
    # Button: load file from explorer
    def buttonImport(self):
        # open dialog
        file = filedialog.askopenfile(title="Load data...", mode='r', filetypes=[("All files", "*.*")])

        # extract filename as str
        filename = str(file.name)

        # insert
        print(filename)
        tk.messagebox.showinfo(title='Import', message='Loading file successfully:\n' + filename)
        #entry_spectrum.insert(tk.END, str(filename))

        # call load data function
        load_data(filename)
    
    # Button: call calculation function
    def button_SpectrumDeconvolution(self):
        spectrum_deconv()

        
class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        
        label = ttk.Label(self, text="Home", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button = ttk.Button(self, text="2 nuclids: Lu177m, I131",
                            command=lambda: controller.show_frame(Nuclids_2))
        button.pack()

        button2 = ttk.Button(self, text="3 nuclids: Lu177m, Lu177, I131",
                            command=lambda: controller.show_frame(Nuclids_3))
        button2.pack()


class Nuclids_2(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = ttk.Label(self, text="2 nuclids: Lu177m, I131", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        buttonHome = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        buttonHome.pack()
        
        buttonImport = ttk.Button(self, text="Import", command=controller.buttonImport)
        buttonImport.pack()
    
        buttonCalc = ttk.Button(self, text="Start Calculation", command=controller.button_SpectrumDeconvolution)
        buttonCalc.pack()

        canvas = FigureCanvasTkAgg(fig, self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)


class Nuclids_3(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = ttk.Label(self, text="3 nuclids: Lu177m, Lu177, I131", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        buttonHome = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        buttonHome.pack()
        
        f = Figure(figsize=(5,5), dpi=100)
        a = f.add_subplot(111)
        a.plot([1,2,3,4,5,6,7,8],[5,6,1,3,8,9,3,5])

        canvas = FigureCanvasTkAgg(f, self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)


app = SpectrumDeconv()
app.mainloop()