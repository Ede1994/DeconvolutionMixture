# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 08:36:38 2021

@author: Eric
"""

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


LARGE_FONT= ("Verdana", 12)
NORM_FONT = ("Helvetica", 10)
SMALL_FONT = ("Helvetica", 8)

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
    popup.wm_title("!")
    label = ttk.Label(popup, text='''For descriptions around the program read help.txt please! Or follow us on GitHub.''', font=NORM_FONT)
    label.pack(side="top", fill="x", pady=10)
    B1 = ttk.Button(popup, text="Okay", command = popup.destroy)
    B1.pack()
    popup.mainloop()

def Impressum():
    popup = tk.Tk()
    popup.wm_title("!")
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

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()

        
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
        
        buttonImport = ttk.Button(self, text="Import",
                            command=lambda: controller.buttonImport)
        buttonImport.pack()
        
        label_spectrum = ttk.Label(self, text="Path Spectrum:")
        label_spectrum.pack(pady=10,padx=10)
        entry_spectrum = ttk.Entry(self)
        entry_spectrum.pack(pady=10,padx=10)


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