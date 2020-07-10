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
    fig1 = make_subplots(rows=nb_rows, cols=nb_cols, subplot_titles=multiCellTA.label_detailed_list)
    row_nb = 1
    for label in multiCellTA.get_label_list():
        x = multiCellTA[label]
        y = multiCellTA[title_str][label]
        fig1.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='markers'),
                row=row_nb,
                col=1)
        print("row: ", row_nb)
        row_nb = row_nb + 1

        # We format the figure

        fig1.update_layout(
            title=title_str,
            height=nb_rows * fig_pixel_definition, # height of the plot
            width=nb_cols * fig_pixel_definition, # The number here is the width of the plot (sum of the widths of all subplots), we thus multiply by the number of columns by the width of a single subplot (3*600)
            font=dict(
                    family="Courier New, monospace",
                    size=18,
                    color="#7f7f7f")
            )
        fig1.update_xaxes(title="time in seconds")
        fig1.update_yaxes(title="Mean Square Displacement")
    fig1.show()