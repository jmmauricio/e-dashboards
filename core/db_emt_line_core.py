# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import numpy as np
import ipywidgets
import matplotlib.pyplot as plt


def intplot(file,xmin=0,xmax=200,ymin=300,ymax=550,step=1.0e-6):
    Long = 200.0
    N_pos = 200
    
    fig, axes = plt.subplots(nrows=1,ncols=1, figsize=(6, 3), dpi=100)

    data = np.load(file)

    V_s = data['X'][:,0:-1:2]
    axes.plot(range(N_pos),V_s[0,:]/1e3,'r')

    line_plot = axes.plot(range(N_pos),V_s[0,:]/1e3)
    #axes.legend()
    axes.set_xlabel('Posición (m)')
    fig.tight_layout()

    axes.set_ylim((ymin,ymax))
    axes.set_xlim((xmin,xmax))

    axes.grid(True)

    fig.tight_layout()

    sld_T  = ipywidgets.FloatSlider(orientation='horizontal',description = "Time $(\mu s)$", 
                                    value=0, min=0,max= data['t'][-1]*1e6, 
                                    step=step,continuous_update=False)

    def update(change):

        t = sld_T.value   
        it = np.searchsorted(data['t'][:,0], t*1e-6)
        line_plot[0].set_data(range(N_pos),V_s[it,:]/1e3)
        fig.canvas.draw_idle()

    sld_T.observe(update, names='value')

    layout_row1 = ipywidgets.HBox([fig.canvas])
    layout_row2 = ipywidgets.HBox([sld_T])

    layout = ipywidgets.VBox([layout_row1,layout_row2])
    return layout
# -


