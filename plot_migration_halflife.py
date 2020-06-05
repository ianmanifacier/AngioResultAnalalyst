#!/usr/bin/python
# coding: utf-8

# external library imports
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

# local library imports
from result_reader import plotmigrationpatterns as myplt


# define subplot titles


print(os.getcwd())
filename = os.path.join(os.getcwd(),'results.csv')
print("uploading file: {}".format(filename))
my_subplot_titles, myCells = myplt.load_results(filename)

limxy = 100

fig1 = make_subplots(rows=1, cols=3, subplot_titles=my_subplot_titles)
k0  = 0


for cell in myCells:
        k0 += 1
        nbSteps = myCells[cell].nbSteps
        print(myCells[cell].label)
        fig1.add_trace(go.Scatter(
            x=myCells[cell].x[nbSteps:nbSteps+1]-myCells[cell].x[0],
            y=myCells[cell].y[nbSteps:nbSteps+1]-myCells[cell].y[0],
            name="cell {} ".format(cell),
            line_color='rgba(120,30,30,.2)'),
            row=1,
            col=myCells[cell].labelRefNb)

# We format the figure    
fig1.update_layout(
    title_text="Subplots",
    title="Cells random migration",
    xaxis_title="migration along the x in µm",
    yaxis_title="migration along the y in µm",
    height=600, # height of the plot
    width=1800, # The number here is the width of the plot (sum of the widths of all subplots), we thus multiply by the number of columns by the width of a single subplot (3*600)
    font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f")
    )
fig1.update_xaxes(title="in µm", range=[-limxy, limxy])
fig1.update_yaxes(title="in µm", range=[-limxy, limxy])

print("Number of simulations : {}".format(len(myCells)))
fig1.show()



""" Histogram representing distances """

fig2 = make_subplots(rows=1, cols=3, subplot_titles=my_subplot_titles)
k0  = 0

#fig2 = go.Figure()
for cell in myCells:
        k0 += 1
        nbSteps = myCells[cell].nbSteps
        fig2.add_trace(go.Scatter(
            x=myCells[cell].x[0:nbSteps+1]-myCells[cell].x[0],
            y=myCells[cell].y[0:nbSteps+1]-myCells[cell].y[0],
            name="cell {} ".format(cell),
            line_color='rgba(120,30,30,.2)'),
            row=1,
            col=myCells[cell].labelRefNb)

# We format the figure    
fig2.update_layout(
    title_text="Subplots",
    title="Cells random migration",
    xaxis_title="migration along the x in µm",
    yaxis_title="migration along the y in µm",
    height=600, # height of the plot
    width=1800, # The number here is the width of the plot (sum of the widths of all subplots), we thus multiply by the number of columns by the width of a single subplot (3*600)
    font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f")
    )

fig2.update_xaxes(title="in µm", range=[-limxy, limxy])
fig2.update_yaxes(title="in µm", range=[-limxy, limxy])

print("Number of simulations : {}".format(len(myCells)))
fig2.show()
