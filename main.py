#!/usr/bin/python
# coding: utf-8

# external library imports
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

# local library imports
from result_reader import plotmigrationpatterns as myplt

filename = os.path.join(os.getcwd(),'..\\results.csv')
print("uploading file: {}".format(filename))
myCells = myplt.load_results(filename)


""" Histogram representing distances """
x = np.empty([ len(myCells), ], dtype=float)
y = np.empty([ len(myCells), ], dtype=float)
distances = np.empty([ len(myCells), ], dtype=float)

i = int(0)
for cell in myCells:
    x[i] = myCells[cell].x[ myCells[cell].nbSteps ]
    y[i] = myCells[cell].y[ myCells[cell].nbSteps ]
    distances[i] = myCells[cell].distance()
    i += 1
myplt.historgram(distances)

""" Histogram of x, y and distance """
x0 = np.ones(x.shape, dtype=float)
x_centered = x - x[0]*x0

y0 = np.ones(y.shape, dtype=float)
y_centered = y - y[0]*y0

data_list = [x_centered, y_centered]
data_labels = ["x" , "y"]
myplt.historgramFromFist(data_list, data_labels)



""" Lets make two distribution (with k_edge=0 and k_edge!=0 )"""
nbCell_K_egde_0 = int(0)
for cell in myCells: 
    if (myCells[cell].K_stiffness_edge == 0): # if K_stiffness_edge IS NULL
        nbCell_K_egde_0 += 1

x_k0 = np.empty([ nbCell_K_egde_0 , ], dtype=float)
y_k0 = np.empty([ nbCell_K_egde_0, ], dtype=float)
distances_k0 = np.empty([ nbCell_K_egde_0, ], dtype=float)

x_k = np.empty([ len(myCells) - nbCell_K_egde_0, ], dtype=float)
y_k = np.empty([ len(myCells) - nbCell_K_egde_0, ], dtype=float)
distances_k = np.empty([ len(myCells) - nbCell_K_egde_0, ], dtype=float)


i_k0 = int(0)
i_k = int(0)
for cell in myCells: 
    if (myCells[cell].K_stiffness_edge == 0): # if K_stiffness_edge IS NULL
        x_k0[i_k0] = myCells[cell].x[ myCells[cell].nbSteps ]
        y_k0[i_k0] = myCells[cell].y[ myCells[cell].nbSteps ]
        distances_k0[i_k0] = myCells[cell].distance()
        i_k0 += 1
    else: # if K_stiffness_edge is NON null
        x_k[i_k] = myCells[cell].x[ myCells[cell].nbSteps ]
        y_k[i_k] = myCells[cell].y[ myCells[cell].nbSteps ]
        distances_k[i_k] = myCells[cell].distance()        
        i_k += 1

""" Distances ( k_edge=0 || k_edge!=0 || ALL ) """
data_list = [distances_k0, distances_k, distances]
data_labels = ["K edge is null" , "K edge is non null", "all conditions"]
myplt.historgramFromFist(data_list, data_labels)


# tuto: https://plot.ly/python/histograms/
fig1 = go.Figure()
fig1.add_trace(go.Histogram(x=distances_k0))
fig1.add_trace(go.Histogram(x=distances_k))
fig1.add_trace(go.Histogram(x=distances))
fig1.update_layout(barmode='overlay')
fig1.update_layout(title=go.layout.Title(text="Over lay of distributions"))
fig1.update_xaxes(range=[0, 150])
fig1.update_yaxes(range=[0, 500])
fig1.show()

fig2 = make_subplots(rows=1, cols=3)

k0  = 0
kk1 = 0
kk5 = 0


#fig2 = go.Figure()
for cell in myCells:
    if round(myCells[cell].K_stiffness_edge) == 0 and round(myCells[cell].K_stiffness_central) == 1:
        k0 += 1
        nbSteps = myCells[cell].nbSteps
        fig2.add_trace(go.Scatter(
            x=myCells[cell].x[0:nbSteps+1]-myCells[cell].x[0],
            y=myCells[cell].y[0:nbSteps+1]-myCells[cell].y[0],
            name="cell {} ".format(cell),
            line_color='rgba(120,30,30,.2)'),
            row=1,
            col=1)
    elif round(myCells[cell].K_stiffness_edge) == 1 and round(myCells[cell].K_stiffness_central) == 1:  
        kk1 += 1
        nbSteps = myCells[cell].nbSteps
        fig2.add_trace(go.Scatter(
            x=myCells[cell].x[0:nbSteps+1]-myCells[cell].x[0],
            y=myCells[cell].y[0:nbSteps+1]-myCells[cell].y[0],
            name="cell {} ".format(cell),
            line_color='rgba(30,100,30,.2)'),
            row=1,
            col=2)
    elif round(myCells[cell].K_stiffness_edge) == 5 and round(myCells[cell].K_stiffness_central) == 5:
        kk5 += 1
        nbSteps = myCells[cell].nbSteps
        fig2.add_trace(go.Scatter(
            x=myCells[cell].x[0:nbSteps+1]-myCells[cell].x[0],
            y=myCells[cell].y[0:nbSteps+1]-myCells[cell].y[0],
            name="cell {} ".format(cell),
            line_color='rgba(30,30,100,.2)'),
            row=1,
            col=3)
    
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
fig2.update_xaxes(title="in µm", range=[-150, 150])
fig2.update_yaxes(title="in µm", range=[-150, 150])

print(" k0  = {}".format(k0))
print(" kk1 = {}".format(kk1))
print(" kk5 = {}".format(kk5))

fig2.show()