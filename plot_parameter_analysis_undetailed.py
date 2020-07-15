# coding: utf-8

# external library imports
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
import scipy.stats

# local library imports
#from result_reader import plotmigrationpatterns as myplt

from plot_tools_and_loader import loadData
from plot_tools_and_loader import getNb_rows_cols
from plot_tools_and_loader import fig_pixel_definition
from plot_tools_and_loader import font_dict
from plot_tools_and_loader import MultiCellTrajectoryAnalysis
from plot_tools_and_loader import mycolorscale
from plot_tools_and_loader import mycolorscaleSimplified
from plot_tools_and_loader import mycolorscale_P_values


# Lets clear the console
os.system("cls")

pearson_correlation = True
rank_correlatation = True
nb_cols = 1


parameter_ranges, parameter_list, my_subplot_titles, myCells, multiCellTA = loadData()
nb_rows, nb_cols = getNb_rows_cols(multiCellTA.get_label_list(), nb_cols)

print(my_subplot_titles)

#https://realpython.com/numpy-scipy-pandas-correlation-python/


mesurement_list = ("distance of travel", "distance from origin", "tortuosity", "average curved speed","average speed (straight line)")
for title_str in multiCellTA.parameters_path_analysis_keys_list:
    fig1 = make_subplots(rows=nb_rows, cols=2, subplot_titles=multiCellTA.label_detailed_list)
    row_nb = 1
    for label in multiCellTA.get_label_list():
        x = multiCellTA[label]
        y = multiCellTA[title_str][label]
        #
        if label != 'l1':
            xm = np.reshape(x,(100, 10), order='F')
            xm = xm[0,0:10]
            ym = np.reshape(y,(100, 10), order='F')
            y_075 = np.quantile(ym, 0.75, axis=0)
            y_mean = np.quantile(ym, 0.5, axis=0)
            y_025 = np.quantile(ym, 0.25, axis=0)
            y_095 = np.quantile(ym, 0.95, axis=0)
            y_005 = np.quantile(ym, 0.05, axis=0)
            y_std = np.std(ym, axis=0)
            fig1.add_trace(go.Scatter(x=xm,y=y_095, fill='toself', mode= 'none', fillcolor='rgba(0, 0, 0, 0.0)'), row=row_nb, col=1)
            fig1.add_trace(go.Scatter(x=xm,y=y_005, fill='tonexty', mode= 'none', fillcolor='rgba(255, 218, 193, 1)'), row=row_nb, col=1)
            fig1.add_trace(go.Scatter(x=xm,y=y_075, fill='toself', mode= 'none', fillcolor='rgba(0, 0, 0, 0.0)'), row=row_nb, col=1)
            fig1.add_trace(go.Scatter(x=xm,y=y_025, fill='tonexty', mode= 'none', fillcolor='rgba(255, 154, 162, 1)'), row=row_nb, col=1)
            fig1.add_trace(go.Scatter( x=x, y=y, mode='markers', marker=dict(color='rgba(255, 255, 255, 1)')), row=row_nb, col=1)
            fig1.add_trace(go.Scatter(x=xm,y=y_mean, line_color='rgba(99, 110, 250, 1)'), row=row_nb, col=1)
            fig1.add_trace(go.Scatter(x=xm,y=y_std, line_color='rgba(99, 110, 250, 1)'), row=row_nb, col=2)
        else:
            fig1.add_trace(go.Scatter( x=x, y=y, mode='markers', marker=dict(color='rgba(99, 110, 250, 1)')), row=row_nb, col=1)
        fig1.update_xaxes(title=label, row=row_nb, col=1)
        fig1.update_yaxes(title=title_str, row=row_nb, col=1)
        fig1.update_xaxes(title=label, row=row_nb, col=2)
        fig1.update_yaxes(title="SD (" + title_str + ")", row=row_nb, col=2, range=[0,1.1*max(y_std)])

        print("row: ", row_nb)
        row_nb = row_nb + 1
        

        # We format the figure

    fig1.update_layout(
        title=title_str,
        height=nb_rows * fig_pixel_definition, # height of the plot
        width=2.5 * fig_pixel_definition, # The number here is the width of the plot (sum of the widths of all subplots), we thus multiply by the number of columns by the width of a single subplot (3*600)
        font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f")
        )
    fig1.show()
    