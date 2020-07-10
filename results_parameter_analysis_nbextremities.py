#!/usr/bin/python
# coding: utf-8

# external library imports
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
from math import floor

# local library imports
from result_reader.result_loader import load_results
from plot_tools_and_loader import font_dict
from plot_tools_and_loader import assign_subplot

# Loading data : opening the csv file that contains the data
filename = os.path.join(os.getcwd(),'..\\results_parameter_analysis_nbextremities.csv')
print("uploading file: {}".format(filename))
myCells = load_results(filename)

# Ploting: we create a 15 subplots to visually compare the migration paterns generated based on the number of extremities
rows = 4
cols = 4
subplot_titles=("2 extremities", "3 extremities", "4 extremities", "5 extremities", "6 extremities", "7 extremities",
                "8 extremities", "9 extremities", "10 extremities", "11 extremities", "12 extremities", "13 extremities",
                "14 extremities", "15 extremities", "16 extremities")

fig2 = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles) #  ( how to define subplots: https://plot.ly/python/subplots/ )


for cell in myCells:
    nvalue = myCells[cell].nbExtremities - 2 # we subtract 2 because we start with 2 extremities
    row, col = assign_subplot(rows, cols, nvalue)
    nbSteps = myCells[cell].nbSteps
    fig2.add_trace(go.Scatter(
            x=myCells[cell].x[0:nbSteps+1]-myCells[cell].x[0],
            y=myCells[cell].y[0:nbSteps+1]-myCells[cell].y[0],
            name="cell {} ".format(cell),
            line_color='rgba(120,30,30,.2)'),
            row=row,
            col=col)
    
# We format the figure
fig2.update_layout(
    title_text="Subplots",
    title="Cells random migration",
    xaxis_title="migration along the x in µm",
    yaxis_title="migration along the y in µm",
    height=2400, # height of the plot
    width=2400, # The number here is the width of the plot (sum of the widths of all subplots), we thus multiply by the number of columns by the width of a single subplot (3*600)
    font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f")
    )
fig2.update_xaxes(title="in µm", range=[-150, 150])
fig2.update_yaxes(title="in µm", range=[-150, 150])

print("Number of simulations : {}".format(len(myCells)))

fig2.show()