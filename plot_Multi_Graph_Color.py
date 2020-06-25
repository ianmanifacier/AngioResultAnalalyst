#!/usr/bin/python
# coding: utf-8

# external library imports
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
import scipy.stats
import random

# local library imports
#from result_reader import plotmigrationpatterns as myplt
from plot_tools_and_loader import *

showGraph = True

print("\n\n\n ********************************************** \n  ******************************************** \n   ****************************************** \n\n")
print("     Number of simulations : ", len(myCells), "(total number of cells)" )
print("     Number of labels (conditions) : ", len(my_subplot_titles) )
print("     Number of cells per condition : ", int( len(myCells)/len(my_subplot_titles) ) )
print("\n     my_subplot_titles :")
i = 0
for sub_title in my_subplot_titles:
    print("         " + str(i) + " : " + str(sub_title))
    i = i + 1

def matchColor(valueInterval, value, labelstr, continuous=False):
    if type(value) == str():
        if value.isnumeric():
            value = float(value)
    if type(value) is int or type(value) is float:
        v_min = float(valueInterval[0])
        v_max = float(valueInterval[-1])
        if v_min != v_max:
            if continuous:
                B = 255*(value-v_min)/(v_max-v_min)
                R = 255 - B
                B = round(B)
                R = round(R)
                if R>255: R=255
                if R<0: R=0
                if B>255: B=255
                if B<0: B=0
                return "rgba(" + str(R) + ",0," + str(B) + ",.4)"
            else:
                if value < v_min + (v_max-v_min)/3.0:
                    return 'rgba(0,0,255,.8)'
                if value < v_min + 2*(v_max-v_min)/3.0:
                    return 'rgba(0,255,0,.8)'
                else:
                    return 'rgba(255,0,0,.8)'
        else:
            print("v_min == v_max")
            return 'rgba(120,30,30,.4)'
    print("Color not matched: label ", labelstr)
    return 'rgba(120,30,30,.4)'


def matchCellColor(parameter_ranges, cell):
    valueInterval = parameter_ranges[cell.label_x]
    try:
        value = getattr(cell,cell.label_x)
    except AttributeError:
        if cell.label == "l1":
            value = getattr(cell,'l1min')
        else:
            raise "error: found in MatchCellColor probably due to inappropriate label assignement"
    return matchColor(valueInterval, value, labelstr=cell.label_x)
                



if showGraph:
    fig1 = make_subplots(rows=nb_rows, cols=nb_cols, subplot_titles=my_subplot_titles)
    k0  = 0

    for cell in myCells:
            k0 += 1
            nbSteps = myCells[cell].nbSteps
            row_nb, col_nb = get_row_col_nb(myCells[cell].labelRefNb, nb_cols)
            #print("k0 = ", k0, " row_nb = ", row_nb," col_nb = ", col_nb)
            fig1.add_trace(go.Scatter(
                x=myCells[cell].x[nbSteps:nbSteps+1]-myCells[cell].x[0],
                y=myCells[cell].y[nbSteps:nbSteps+1]-myCells[cell].y[0],
                name="cell {} ".format(cell),
                line_color=matchCellColor(parameter_ranges, myCells[cell])),
                row=row_nb,
                col=col_nb)

    # We format the figure    
    fig1.update_layout(
        title_text="Cells random migration (end points)",
        title="Cells random migration (end points)",
        height=nb_rows * fig_pixel_definition, # height of the plot
        width=nb_cols * fig_pixel_definition, # The number here is the width of the plot (sum of the widths of all subplots), we thus multiply by the number of columns by the width of a single subplot (3*600)
        font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f")
        )
    fig1.update_xaxes(title="migration along the x in µm", range=[-limxy, limxy])
    fig1.update_yaxes(title="migration along the y in µm", range=[-limxy, limxy])
    fig1.show()


if showGraph:
    """ Histogram representing distances """
    fig2 = make_subplots(rows=nb_rows, cols=nb_cols, subplot_titles=my_subplot_titles)
    k0  = 0

    #fig2 = go.Figure()
    for cell in myCells:
            k0 += 1
            nbSteps = myCells[cell].nbSteps
            row_nb, col_nb = get_row_col_nb(myCells[cell].labelRefNb, nb_cols)
            fig2.add_trace(go.Scatter(
                x=myCells[cell].x[0:nbSteps+1]-myCells[cell].x[0],
                y=myCells[cell].y[0:nbSteps+1]-myCells[cell].y[0],
                name="cell {} ".format(cell),
                line_color='rgba(120,30,30,.2)'),
                row=row_nb,
                col=col_nb)

    # We format the figure    
    fig2.update_layout(
        title_text="Cells random migration (trajectories)",
        title="Cells random migration (trajectories)",
        height=nb_rows * fig_pixel_definition, # height of the plot
        width=nb_cols * fig_pixel_definition, # The number here is the width of the plot (sum of the widths of all subplots), we thus multiply by the number of columns by the width of a single subplot (3*600)
        font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f")
        )

    fig2.update_xaxes(title="migration along the x in µm", range=[-limxy, limxy])
    fig2.update_yaxes(title="migration along the y in µm", range=[-limxy, limxy])
    fig2.show()


if showGraph:
    """ Graph bars transition time representing distances """
    fig3 = make_subplots(rows=nb_rows, cols=nb_cols, subplot_titles=my_subplot_titles)
    k0  = 0

    v_minimum_dict = dict()
    v_mean_dict = dict()
    v_maximum_dict = dict()

    limxy_bar_max = 0

    for cell in myCells:
            k0 += 1
            nbSteps = myCells[cell].nbSteps
            # Minimum
            if myCells[cell].labelRefNb in v_minimum_dict:
                v_minimum_dict[ myCells[cell].labelRefNb ]= np.append(v_minimum_dict[ myCells[cell].labelRefNb ], myCells[cell].immatureN1TransitionPeriod_minimum)
            else:
                v_minimum_dict[ myCells[cell].labelRefNb ] = np.empty([0, ], dtype=float)
            # Mean
            if myCells[cell].labelRefNb in v_mean_dict:
                v_mean_dict[ myCells[cell].labelRefNb ]= np.append(v_mean_dict[ myCells[cell].labelRefNb ], myCells[cell].immatureN1TransitionPeriod_mean)
            else:
                v_mean_dict[ myCells[cell].labelRefNb ] = np.empty([0, ], dtype=float)
            # Maximum
            if myCells[cell].labelRefNb in v_maximum_dict:
                v_maximum_dict[ myCells[cell].labelRefNb ]= np.append( v_maximum_dict[ myCells[cell].labelRefNb ], myCells[cell].immatureN1TransitionPeriod_maximum)
            else:
                v_maximum_dict[ myCells[cell].labelRefNb ] = np.empty([0, ], dtype=float)


    for labelRefNb in range(1,len(my_subplot_titles)+1):
            limxy_bar_max = max( limxy_bar_max, v_maximum_dict[labelRefNb].max())
            row_nb, col_nb = get_row_col_nb(labelRefNb, nb_cols)
            fig3.add_trace( go.Bar(x=['minimum','mean','maximum'], y=[ v_minimum_dict[labelRefNb].mean(), v_mean_dict[labelRefNb].mean(), v_maximum_dict[labelRefNb].mean()], error_y=dict(
                    type='data',
                    symmetric=False,
                    array=[v_minimum_dict[labelRefNb].max()- v_minimum_dict[labelRefNb].mean(), v_mean_dict[labelRefNb].max() - v_mean_dict[labelRefNb].mean(), v_maximum_dict[labelRefNb].max() - v_maximum_dict[labelRefNb].mean()],
                    arrayminus=[ v_minimum_dict[labelRefNb].mean() - v_minimum_dict[labelRefNb].min(), v_mean_dict[labelRefNb].mean() - v_mean_dict[labelRefNb].min(), v_maximum_dict[labelRefNb].mean() - v_maximum_dict[labelRefNb].min()])),
                row=row_nb,
                col=col_nb)

    # We format the figure
    fig3.update_layout(
        title_text="Subplots",
        title="Cells random migration",
        height=nb_rows * fig_pixel_definition, # height of the plot
        width=nb_cols * fig_pixel_definition, # The number here is the width of the plot (sum of the widths of all subplots), we thus multiply by the number of columns by the width of a single subplot (3*600)
        font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f")
        )

    fig3.update_xaxes(title="immature N1 Transition Period")
    fig3.update_yaxes(title="time in seconds", range=[0, 1.1*limxy_bar_max])
    fig3.show()


if showGraph:
    """ Graph bars transition time representing distances """
    fig4 = make_subplots(rows=nb_rows, cols=nb_cols, subplot_titles=my_subplot_titles)
    k0  = 0

    v_minimum_dict = dict()
    v_mean_dict = dict()
    v_maximum_dict = dict()

    limxy_bar_max = 0

    for cell in myCells:
        k0 += 1
        nbSteps = myCells[cell].nbSteps
        # Minimum
        if myCells[cell].labelRefNb in v_minimum_dict:
            v_minimum_dict[ myCells[cell].labelRefNb ]= np.append(v_minimum_dict[ myCells[cell].labelRefNb ], myCells[cell].intermediateN1TransitionPeriod_minimum)
        else:
            v_minimum_dict[ myCells[cell].labelRefNb ] = np.empty([0, ], dtype=float)
        # Mean
        if myCells[cell].labelRefNb in v_mean_dict:
            v_mean_dict[ myCells[cell].labelRefNb ]= np.append(v_mean_dict[ myCells[cell].labelRefNb ], myCells[cell].intermediateN1TransitionPeriod_mean)
        else:
            v_mean_dict[ myCells[cell].labelRefNb ] = np.empty([0, ], dtype=float)
        # Maximum
        if myCells[cell].labelRefNb in v_maximum_dict:
            v_maximum_dict[ myCells[cell].labelRefNb ]= np.append( v_maximum_dict[ myCells[cell].labelRefNb ], myCells[cell].intermediateN1TransitionPeriod_maximum)
        else:
            v_maximum_dict[ myCells[cell].labelRefNb ] = np.empty([0, ], dtype=float)
    

    for labelRefNb in range(1,len(my_subplot_titles)+1):
        limxy_bar_max = max( limxy_bar_max, v_maximum_dict[labelRefNb].max())
        row_nb, col_nb = get_row_col_nb(labelRefNb, nb_cols)
        fig4.add_trace( go.Bar(x=['minimum','mean','maximum'], y=[ v_minimum_dict[labelRefNb].mean(), v_mean_dict[labelRefNb].mean(), v_maximum_dict[labelRefNb].mean()], error_y=dict(
                type='data',
                symmetric=False,
                array=[v_minimum_dict[labelRefNb].max()- v_minimum_dict[labelRefNb].mean(), v_mean_dict[labelRefNb].max() - v_mean_dict[labelRefNb].mean(), v_maximum_dict[labelRefNb].max() - v_maximum_dict[labelRefNb].mean()],
                arrayminus=[ v_minimum_dict[labelRefNb].mean() - v_minimum_dict[labelRefNb].min(), v_mean_dict[labelRefNb].mean() - v_mean_dict[labelRefNb].min(), v_maximum_dict[labelRefNb].mean() - v_maximum_dict[labelRefNb].min()])),
            row=row_nb,
            col=col_nb)

    # We format the figure
    fig4.update_layout(
        title_text="Subplots",
        title="intermediate N1 Transition Period",
        height=nb_rows * fig_pixel_definition, # height of the plot
        width=nb_cols * fig_pixel_definition, # The number here is the width of the plot (sum of the widths of all subplots), we thus multiply by the number of columns by the width of a single subplot (3*600)
        font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f")
        )

    fig3.update_xaxes(title="intermediate N1 Transition Period",)
    fig4.update_yaxes(title="time in seconds", range=[0, 1.1*limxy_bar_max])
    fig4.show()


if showGraph:
    """ Graph bars transition mature transition periode """
    fig5 = make_subplots(rows=nb_rows, cols=nb_cols, subplot_titles=my_subplot_titles)
    k0  = 0

    v_minimum_dict = dict()
    v_mean_dict = dict()
    v_maximum_dict = dict()

    limxy_bar_max = 0

    for cell in myCells:
        k0 += 1
        nbSteps = myCells[cell].nbSteps
        # Minimum
        if myCells[cell].labelRefNb in v_minimum_dict:
            v_minimum_dict[ myCells[cell].labelRefNb ]= np.append(v_minimum_dict[ myCells[cell].labelRefNb ], myCells[cell].matureN1TransitionPeriod_minimum)
        else:
            v_minimum_dict[ myCells[cell].labelRefNb ] = np.empty([0, ], dtype=float)
        # Mean
        if myCells[cell].labelRefNb in v_mean_dict:
            v_mean_dict[ myCells[cell].labelRefNb ]= np.append(v_mean_dict[ myCells[cell].labelRefNb ], myCells[cell].matureN1TransitionPeriod_mean)
        else:
            v_mean_dict[ myCells[cell].labelRefNb ] = np.empty([0, ], dtype=float)
        # Maximum
        if myCells[cell].labelRefNb in v_maximum_dict:
            v_maximum_dict[ myCells[cell].labelRefNb ]= np.append( v_maximum_dict[ myCells[cell].labelRefNb ], myCells[cell].matureN1TransitionPeriod_maximum)
        else:
            v_maximum_dict[ myCells[cell].labelRefNb ] = np.empty([0, ], dtype=float)


    for labelRefNb in range(1,len(my_subplot_titles)+1):
        limxy_bar_max = max( limxy_bar_max, v_maximum_dict[labelRefNb].max())
        row_nb, col_nb = get_row_col_nb(labelRefNb, nb_cols)
        fig5.add_trace( go.Bar(x=['minimum','mean','maximum'], y=[ v_minimum_dict[labelRefNb].mean(), v_mean_dict[labelRefNb].mean(), v_maximum_dict[labelRefNb].mean()], error_y=dict(
                type='data',
                symmetric=False,
                array=[v_minimum_dict[labelRefNb].max()- v_minimum_dict[labelRefNb].mean(), v_mean_dict[labelRefNb].max() - v_mean_dict[labelRefNb].mean(), v_maximum_dict[labelRefNb].max() - v_maximum_dict[labelRefNb].mean()],
                arrayminus=[ v_minimum_dict[labelRefNb].mean() - v_minimum_dict[labelRefNb].min(), v_mean_dict[labelRefNb].mean() - v_mean_dict[labelRefNb].min(), v_maximum_dict[labelRefNb].mean() - v_maximum_dict[labelRefNb].min()])),
            row=row_nb,
            col=col_nb)

    # We format the figure
    fig5.update_layout(
        title_text="Mature N1 Transition Period",
        title="Mature N1 Transition Period",
        height=nb_rows * fig_pixel_definition, # height of the plot
        width=nb_cols * fig_pixel_definition, # The number here is the width of the plot (sum of the widths of all subplots), we thus multiply by the number of columns by the width of a single subplot (3*600)
        font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f")
        )

    fig3.update_xaxes(title="Mature N1 Transition Period")
    fig5.update_yaxes(title="time in seconds", range=[0, 1.1*limxy_bar_max])
    fig5.show()

if showGraph:
    """ Length of All Interaction (sum of all current length) """
    fig6 = make_subplots(rows=nb_rows, cols=nb_cols, subplot_titles=my_subplot_titles)
    k0  = 0

    traceMerge = np.empty([myCells[1].nbSteps, 1], dtype=float)
    limxy_fig6 = 0

    traceMerge_dict = dict()
    
    for cell in myCells:
        k0 += 1
        nbSteps = myCells[cell].nbSteps
        row_nb, col_nb = get_row_col_nb(myCells[cell].labelRefNb, nb_cols)
        limxy_fig6 = max(limxy_fig6, myCells[cell].calculateTheLengthOfAllInteractions[0:nbSteps+1].max())
        fig6.add_trace(go.Scatter(
            x=myCells[cell].t, #x=np.arange(0,nbSteps+1),
            y=myCells[cell].calculateTheLengthOfAllInteractions,
            name="cell {} ".format(cell),
            line_color='rgba(120,30,30,.2)'),
            row=row_nb,
            col=col_nb)
        if (row_nb, col_nb) not in traceMerge_dict:
            traceMerge_dict[row_nb, col_nb] = [myCells[cell].calculateTheLengthOfAllInteractions]
        else:
            traceMerge_dict[row_nb, col_nb] = np.append(traceMerge_dict[row_nb, col_nb], [myCells[cell].calculateTheLengthOfAllInteractions], axis=0)
        
    print("traceMerge.shape = ", traceMerge_dict[1, 1].shape)
    #print("traceMean.shape = ", traceMean.shape)
    #print("length = ", len(myCells[cell].calculateTheLengthOfAllInteractions))
    #print("width = ", myCells[cell].calculateTheLengthOfAllInteractions.size)
    for key in traceMerge_dict:
        fig6.add_trace(go.Scatter(
            x=myCells[1].t,
            y=traceMerge_dict[key].min(axis=0),
            name="cell {} ".format(cell),
            line_color='rgba(150,150,150,255)'),
            row=key[0],
            col=key[1])
        fig6.add_trace(go.Scatter(
            x=myCells[1].t,
            y=traceMerge_dict[key].mean(axis=0),
            name="cell {} ".format(cell),
            line_color='rgba(0,0,0,255)'),
            row=key[0],
            col=key[1])
        fig6.add_trace(go.Scatter(
            x=myCells[1].t,
            y=traceMerge_dict[key].max(axis=0),
            name="cell {} ".format(cell),
            line_color='rgba(150,150,150,255)'),
            row=key[0],
            col=key[1])
        fig6.add_trace(go.Scatter(
            x=myCells[1].t,
            y=traceMerge_dict[key].mean(axis=0) + traceMerge_dict[key].std(axis=0),
            name="cell {} ".format(cell),
            line_color='rgba(150,150,150,255)'),
            row=key[0],
            col=key[1])
        fig6.add_trace(go.Scatter(
            x=myCells[1].t,
            y=traceMerge_dict[key].mean(axis=0) - traceMerge_dict[key].std(axis=0),
            name="cell {} ".format(cell),
            line_color='rgba(150,150,150,255)'),
            row=key[0],
            col=key[1])

    # We format the figure    
    fig6.update_layout(
            title_text="Length of All Interaction (sum of all current length)",
            title="Length of All Interaction (sum of all current length)",
            height=nb_rows * fig_pixel_definition, # height of the plot
            width=nb_cols * fig_pixel_definition, # The number here is the width of the plot (sum of the widths of all subplots), we thus multiply by the number of columns by the width of a single subplot (3*600)
            font=dict(
                    family="Courier New, monospace",
                    size=18,
                    color="#7f7f7f")
            )
    fig6.update_xaxes(title="time (seconds)", range=[0, myCells[1].t[myCells[1].nbSteps] ]) #fig6.update_xaxes(title="time steps", range=[0, myCells[1].nbSteps+1])
    fig6.update_yaxes(title="Length total in µm", range=[0, limxy_fig6])
    fig6.show()

if showGraph:
    """ Length of All B1s """
    fig7 = make_subplots(rows=nb_rows, cols=nb_cols, subplot_titles=my_subplot_titles)
    k0  = 0

    traceMerge_dict = dict()

    limxy_fig7 = 0

    for cell in myCells:
        k0 += 1
        nbSteps = myCells[cell].nbSteps
        row_nb, col_nb = get_row_col_nb(myCells[cell].labelRefNb, nb_cols)
        limxy_fig7 = max(limxy_fig7, myCells[cell].calculateTheLengthOfAllB1s[0:nbSteps+1].max())
        fig7.add_trace(go.Scatter(
                x=myCells[cell].t, #x=np.arange(0,nbSteps+1),
                y=myCells[cell].calculateTheLengthOfAllB1s,
                name="cell {} ".format(cell),
                line_color='rgba(120,30,30,.2)'),
            row=row_nb,
            col=col_nb)

        if (row_nb, col_nb) not in traceMerge_dict:
            traceMerge_dict[row_nb, col_nb] = [myCells[cell].calculateTheLengthOfAllB1s]
        else:
            traceMerge_dict[row_nb, col_nb] = np.append(traceMerge_dict[row_nb, col_nb], [myCells[cell].calculateTheLengthOfAllB1s], axis=0)
    
    for key in traceMerge_dict:
        fig7.add_trace(go.Scatter(
            x=myCells[1].t,
            y=traceMerge_dict[key].min(axis=0),
            name="cell {} ".format(cell),
            line_color='rgba(150,150,150,255)'),
            row=key[0],
            col=key[1])
        fig7.add_trace(go.Scatter(
            x=myCells[1].t,
            y=traceMerge_dict[key].mean(axis=0),
            name="cell {} ".format(cell),
            line_color='rgba(0,0,0,255)'),
            row=key[0],
            col=key[1])
        fig7.add_trace(go.Scatter(
            x=myCells[1].t,
            y=traceMerge_dict[key].max(axis=0),
            name="cell {} ".format(cell),
            line_color='rgba(150,150,150,255)'),
            row=key[0],
            col=key[1])
        fig7.add_trace(go.Scatter(
            x=myCells[1].t,
            y=traceMerge_dict[key].mean(axis=0) + traceMerge_dict[key].std(axis=0),
            name="cell {} ".format(cell),
            line_color='rgba(150,150,150,255)'),
            row=key[0],
            col=key[1])
        fig7.add_trace(go.Scatter(
            x=myCells[1].t,
            y=traceMerge_dict[key].mean(axis=0) - traceMerge_dict[key].std(axis=0),
            name="cell {} ".format(cell),
            line_color='rgba(150,150,150,255)'),
            row=key[0],
            col=key[1])


        # We format the figure    
    fig7.update_layout(
            title_text="Fig7: Length of All B1s",
            title="Length of All B1s",
            height=nb_rows * fig_pixel_definition, # height of the plot
            width=nb_cols * fig_pixel_definition, # The number here is the width of the plot (sum of the widths of all subplots), we thus multiply by the number of columns by the width of a single subplot (3*600)
            font=dict(
                    family="Courier New, monospace",
                    size=18,
                    color="#7f7f7f")
            )
    fig7.update_xaxes(title="time (seconds)", range=[0, myCells[1].t[myCells[1].nbSteps] ]) 
    fig7.update_yaxes(title="Length ALL B1s in µm", range=[0, limxy_fig7])
    fig7.show()


if True:
    """ Mean length of All B1s """
    fig7_B = make_subplots(rows=nb_rows, cols=nb_cols, subplot_titles=my_subplot_titles)
    k0  = 0

    traceMerge_dict = dict()

    limxy_fig7 = 0

    for cell in myCells:
        k0 += 1
        nbSteps = myCells[cell].nbSteps
        row_nb, col_nb = get_row_col_nb(myCells[cell].labelRefNb, nb_cols)
        limxy_fig7 = max(limxy_fig7, myCells[cell].calculateTheLengthOfAllB1s[0:nbSteps+1].max())
        fig7_B.add_trace(go.Scatter(
                x=myCells[cell].t, #x=np.arange(0,nbSteps+1),
                y=myCells[cell].calculateTheLengthOfAllB1s / myCells[cell].countN1s,
                name="cell {} ".format(cell),
                line_color=matchCellColor(parameter_ranges, myCells[cell])),
            row=row_nb,
            col=col_nb)

        if (row_nb, col_nb) not in traceMerge_dict:
            traceMerge_dict[row_nb, col_nb] = [myCells[cell].calculateTheLengthOfAllB1s / myCells[cell].countN1s]
        else:
            traceMerge_dict[row_nb, col_nb] = np.append(traceMerge_dict[row_nb, col_nb], [myCells[cell].calculateTheLengthOfAllB1s / myCells[cell].countN1s], axis=0)
    
    for key in traceMerge_dict:
        fig7_B.add_trace(go.Scatter(
            x=myCells[1].t,
            y=traceMerge_dict[key].min(axis=0),
            name="cell {} ".format(cell),
            line_color='rgba(150,150,150,255)'),
            row=key[0],
            col=key[1])
        fig7_B.add_trace(go.Scatter(
            x=myCells[1].t,
            y=traceMerge_dict[key].mean(axis=0),
            name="cell {} ".format(cell),
            line_color='rgba(0,0,0,255)'),
            row=key[0],
            col=key[1])
        fig7_B.add_trace(go.Scatter(
            x=myCells[1].t,
            y=traceMerge_dict[key].max(axis=0),
            name="cell {} ".format(cell),
            line_color='rgba(150,150,150,255)'),
            row=key[0],
            col=key[1])
        fig7_B.add_trace(go.Scatter(
            x=myCells[1].t,
            y=traceMerge_dict[key].mean(axis=0) + traceMerge_dict[key].std(axis=0),
            name="cell {} ".format(cell),
            line_color='rgba(150,150,150,255)'),
            row=key[0],
            col=key[1])
        fig7_B.add_trace(go.Scatter(
            x=myCells[1].t,
            y=traceMerge_dict[key].mean(axis=0) - traceMerge_dict[key].std(axis=0),
            name="cell {} ".format(cell),
            line_color='rgba(150,150,150,255)'),
            row=key[0],
            col=key[1])


        # We format the figure    
    fig7_B.update_layout(
            title_text="Fig7: Length of All B1s",
            title="Mean length of All B1s",
            height=nb_rows * fig_pixel_definition, # height of the plot
            width=nb_cols * fig_pixel_definition, # The number here is the width of the plot (sum of the widths of all subplots), we thus multiply by the number of columns by the width of a single subplot (3*600)
            font=dict(
                    family="Courier New, monospace",
                    size=18,
                    color="#7f7f7f")
            )
    fig7_B.update_xaxes(title="time (seconds)", range=[0, myCells[1].t[myCells[1].nbSteps] ]) 
    fig7_B.update_yaxes(title="Mean length ALL B1s in µm", range=[0, limxy_fig7])
    fig7_B.show()


if showGraph:
    """ Length of All B2s """
    fig8 = make_subplots(rows=nb_rows, cols=nb_cols, subplot_titles=my_subplot_titles)
    k0  = 0

    traceMerge_dict = dict()

    limxy_fig8 = 0

    for cell in myCells:
        k0 += 1
        nbSteps = myCells[cell].nbSteps
        row_nb, col_nb = get_row_col_nb(myCells[cell].labelRefNb, nb_cols)
        limxy_fig8 = max(limxy_fig8, myCells[cell].calculateTheLengthOfAllB2s[0:nbSteps+1].max())
        fig8.add_trace(go.Scatter(
                x=myCells[cell].t,
                y=myCells[cell].calculateTheLengthOfAllB2s,
                name="cell {} ".format(cell),
                line_color='rgba(120,30,30,.2)'),
            row=row_nb,
            col=col_nb)
                
        if (row_nb, col_nb) not in traceMerge_dict:
            traceMerge_dict[row_nb, col_nb] = [myCells[cell].calculateTheLengthOfAllB2s]
        else:
            traceMerge_dict[row_nb, col_nb] = np.append(traceMerge_dict[row_nb, col_nb], [myCells[cell].calculateTheLengthOfAllB2s], axis=0)
    
    for key in traceMerge_dict:
        fig8.add_trace(go.Scatter(
            x=myCells[1].t,
            y=traceMerge_dict[key].min(axis=0),
            name="cell {} ".format(cell),
            line_color='rgba(150,150,150,255)'),
            row=key[0],
            col=key[1])
        fig8.add_trace(go.Scatter(
            x=myCells[1].t,
            y=traceMerge_dict[key].mean(axis=0),
            name="cell {} ".format(cell),
            line_color='rgba(0,0,0,255)'),
            row=key[0],
            col=key[1])
        fig8.add_trace(go.Scatter(
            x=myCells[1].t,
            y=traceMerge_dict[key].max(axis=0),
            name="cell {} ".format(cell),
            line_color='rgba(150,150,150,255)'),
            row=key[0],
            col=key[1])
        fig8.add_trace(go.Scatter(
            x=myCells[1].t,
            y=traceMerge_dict[key].mean(axis=0) + traceMerge_dict[key].std(axis=0),
            name="cell {} ".format(cell),
            line_color='rgba(150,150,150,255)'),
            row=key[0],
            col=key[1])
        fig8.add_trace(go.Scatter(
            x=myCells[1].t,
            y=traceMerge_dict[key].mean(axis=0) - traceMerge_dict[key].std(axis=0),
            name="cell {} ".format(cell),
            line_color='rgba(150,150,150,255)'),
            row=key[0],
            col=key[1])

    # We format the figure    
    fig8.update_layout(
            title_text="Fig8: Length of All B2s",
            title="Length of All B2s",
            height=nb_rows * fig_pixel_definition, # height of the plot
            width=nb_cols * fig_pixel_definition, # The number here is the width of the plot (sum of the widths of all subplots), we thus multiply by the number of columns by the width of a single subplot (3*600)
            font=dict(
                    family="Courier New, monospace",
                    size=18,
                    color="#7f7f7f")
            )
    fig8.update_xaxes(title="time (seconds)", range=[0, myCells[1].t[myCells[cell].nbSteps] ])
    fig8.update_yaxes(title="Length of All B2s in µm", range=[0, limxy_fig8])
    fig8.show()


if showGraph:
    """ Node Ages"""
    fig9 = make_subplots(rows=nb_rows, cols=nb_cols, subplot_titles=my_subplot_titles)
    k0  = 0

    v_nodeAge_mean_dict = dict()
    v_nodeAgeAtDeath_mean_dict = dict()
    v_nodeAgeAtDeath_maximum_dict = dict()

    limxy_bar_max = 0

    #fig2 = go.Figure()
    for cell in myCells:
            k0 += 1
            nbSteps = myCells[cell].nbSteps
            # Minimum
            if myCells[cell].labelRefNb in v_nodeAge_mean_dict:
                v_nodeAge_mean_dict[ myCells[cell].labelRefNb ]= np.append(v_nodeAge_mean_dict[ myCells[cell].labelRefNb ], myCells[cell].nodeAge_mean)
            else:
                v_nodeAge_mean_dict[ myCells[cell].labelRefNb ] = np.empty([0, ], dtype=float)
            # Mean
            if myCells[cell].labelRefNb in v_nodeAgeAtDeath_mean_dict:
                v_nodeAgeAtDeath_mean_dict[ myCells[cell].labelRefNb ]= np.append(v_nodeAgeAtDeath_mean_dict[ myCells[cell].labelRefNb ], myCells[cell].nodeAgeAtDeath_mean)
            else:
                v_nodeAgeAtDeath_mean_dict[ myCells[cell].labelRefNb ] = np.empty([0, ], dtype=float)
            # Maximum
            if myCells[cell].labelRefNb in v_nodeAgeAtDeath_maximum_dict:
                v_nodeAgeAtDeath_maximum_dict[ myCells[cell].labelRefNb ]= np.append( v_nodeAgeAtDeath_maximum_dict[ myCells[cell].labelRefNb ], myCells[cell].immatureN1TransitionPeriod_maximum)
            else:
                v_nodeAgeAtDeath_maximum_dict[ myCells[cell].labelRefNb ] = np.empty([0, ], dtype=float)



    #print( v_nodeAge_mean_dict.keys() )
    for labelRefNb in range(1,len(my_subplot_titles)+1):
        limxy_bar_max = max( limxy_bar_max, v_nodeAgeAtDeath_maximum_dict[labelRefNb].max())
        row_nb, col_nb = get_row_col_nb(labelRefNb, nb_cols)
        fig9.add_trace( go.Bar(
            x=['node age (mean) (except N0)','node age at Death (mean)','node age at Death (maximum)'],
            y=[ v_nodeAge_mean_dict[labelRefNb].mean(), v_nodeAgeAtDeath_mean_dict[labelRefNb].mean(), v_nodeAgeAtDeath_maximum_dict[labelRefNb].mean()],
            error_y=dict(
                type='data',
                symmetric=False,
                array=[v_nodeAge_mean_dict[labelRefNb].max()- v_nodeAge_mean_dict[labelRefNb].mean(), v_nodeAgeAtDeath_mean_dict[labelRefNb].max() - v_nodeAgeAtDeath_mean_dict[labelRefNb].mean(), v_nodeAgeAtDeath_maximum_dict[labelRefNb].max() - v_nodeAgeAtDeath_maximum_dict[labelRefNb].mean()],
                arrayminus=[ v_nodeAge_mean_dict[labelRefNb].mean() - v_nodeAge_mean_dict[labelRefNb].min(), v_nodeAgeAtDeath_mean_dict[labelRefNb].mean() - v_nodeAgeAtDeath_mean_dict[labelRefNb].min(), v_nodeAgeAtDeath_maximum_dict[labelRefNb].mean() - v_nodeAgeAtDeath_maximum_dict[labelRefNb].min()])),
            row=row_nb,
            col=col_nb)

    # We format the figure
    fig9.update_layout(
        title_text="Node age",
        title="Node age",
        height=nb_rows * fig_pixel_definition, # height of the plot
        width=nb_cols * fig_pixel_definition, # The number here is the width of the plot (sum of the widths of all subplots), we thus multiply by the number of columns by the width of a single subplot (3*600)
        font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f")
        )

    fig9.update_xaxes(title="age in seconds")
    fig9.update_yaxes(title="time in seconds", range=[0, 1.1*limxy_bar_max])
    fig9.show()


if showGraph:
    """ Graph bars transition time representing distances """
    fig10 = make_subplots(rows=nb_rows, cols=nb_cols, subplot_titles=my_subplot_titles)
    k0  = 0

    v_Immature_dict = dict()
    v_Intermediate_dict = dict()
    v_Mature_dict = dict()

    limxy_bar_max = 0

    for cell in myCells:
        k0 += 1
        nbSteps = myCells[cell].nbSteps
        # Minimum
        if myCells[cell].labelRefNb in v_Immature_dict:
            v_Immature_dict[ myCells[cell].labelRefNb ]= np.append(v_Immature_dict[ myCells[cell].labelRefNb ], myCells[cell].nbImmatureN1Transitions)
        else:
            v_Immature_dict[ myCells[cell].labelRefNb ] = np.empty([0, ], dtype=float)
        # Mean
        if myCells[cell].labelRefNb in v_Intermediate_dict:
            v_Intermediate_dict[ myCells[cell].labelRefNb ]= np.append(v_Intermediate_dict[ myCells[cell].labelRefNb ], myCells[cell].nbIntermediateN1Transitions)
        else:
            v_Intermediate_dict[ myCells[cell].labelRefNb ] = np.empty([0, ], dtype=float)
        # Maximum
        if myCells[cell].labelRefNb in v_Mature_dict:
            v_Mature_dict[ myCells[cell].labelRefNb ]= np.append( v_Mature_dict[ myCells[cell].labelRefNb ], myCells[cell].nbMatureN1Transitions)
        else:
            v_Mature_dict[ myCells[cell].labelRefNb ] = np.empty([0, ], dtype=float)
    
    
    for labelRefNb in range(1,len(my_subplot_titles)+1):
            limxy_bar_max = max( limxy_bar_max, v_Immature_dict[labelRefNb].max(), v_Intermediate_dict[labelRefNb].max(), v_Mature_dict[labelRefNb].max())
            row_nb, col_nb = get_row_col_nb(labelRefNb, nb_cols)
            fig10.add_trace( go.Bar(x=['Immature','Intermediate','Mature'], y=[ v_Immature_dict[labelRefNb].mean(), v_Intermediate_dict[labelRefNb].mean(), v_Mature_dict[labelRefNb].mean()], error_y=dict(
                    type='data',
                    symmetric=False,
                    array=[v_Immature_dict[labelRefNb].max()- v_Immature_dict[labelRefNb].mean(), v_Intermediate_dict[labelRefNb].max() - v_Intermediate_dict[labelRefNb].mean(), v_Mature_dict[labelRefNb].max() - v_Mature_dict[labelRefNb].mean()],
                    arrayminus=[ v_Immature_dict[labelRefNb].mean() - v_Immature_dict[labelRefNb].min(), v_Intermediate_dict[labelRefNb].mean() - v_Intermediate_dict[labelRefNb].min(), v_Mature_dict[labelRefNb].mean() - v_Mature_dict[labelRefNb].min()])),
                row=row_nb,
                col=col_nb)
    

    # We format the figure
    fig10.update_layout(
        title_text="Number of N1 Transions",
        title="Number of N1 Transions",
        height=nb_rows * fig_pixel_definition, # height of the plot
        width=nb_cols * fig_pixel_definition, # The number here is the width of the plot (sum of the widths of all subplots), we thus multiply by the number of columns by the width of a single subplot (3*600)
        font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f")
        )

    fig10.update_xaxes(title="Number of N1 Transions")
    fig10.update_yaxes(title="nb of transitions", range=[0, 1.1*limxy_bar_max])
    fig10.show()


if False:
    """ FIG 11: """
    fig11 = make_subplots(rows=1, cols=1, subplot_titles=my_subplot_titles)
    k0  = 0
    limx_fig11 = 0
    limy_fig11 = 0


    R0_1i_array = np.empty([0,],dtype=float)
    for cell in myCells:
        R0_1i_array = np.append( R0_1i_array, myCells[cell].P_N1 * myCells[cell].DN1i )


    for cell in myCells:
        if myCells[cell].label_x == "R0_1i" and myCells[cell].label_y == "CountN1s":
            k0 += 1
            nbSteps = myCells[cell].nbSteps
            row_nb, col_nb = get_row_col_nb(myCells[cell].labelRefNb, nb_cols)
            #print("k0 = ", k0, " row_nb = ", row_nb," col_nb = ", col_nb)
            #print("P_N1 = ", myCells[cell].P_N1, type(myCells[cell].P_N1), " DN1i = ", myCells[cell].DN1i, type(myCells[cell].DN1i))
            limx_fig11 = max(limx_fig11, myCells[cell].P_N1 * myCells[cell].DN1i )
            limy_fig11 = max(limy_fig11, myCells[cell].countN1s.mean() )
            fig11.add_trace(go.Scatter(
                    x= (myCells[cell].P_N1 * myCells[cell].DN1i, myCells[cell].P_N1 * myCells[cell].DN1i),
                    y= (myCells[cell].countN1s.mean(), myCells[cell].countN1s.mean()),
                        name="cell {} ".format(cell),
                        line_color='rgba(120,30,30,.2)'),
                    row=1,
                    col=1)
            #print("cell = ", cell, "; R0_1i = ", R0_1i, "; nb N1s (mean) = ", myCells[cell].countN1s.mean())


    # We format the figure    
    fig11.update_layout(
        title_text="Fig 11: Cells random migration (end points)",
        title="Cells random migration (end points)",
        height=nb_rows * fig_pixel_definition, # height of the plot
        width=nb_cols * fig_pixel_definition, # The number here is the width of the plot (sum of the widths of all subplots), we thus multiply by the number of columns by the width of a single subplot (3*600)
        font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f")
        )
    fig11.update_xaxes(title="R0_1i = P_N1 * DN1i", range=[0, 1.1*limx_fig11])
    fig11.update_yaxes(title="nb N1s (mean)", range=[0, 1.1*limy_fig11])
    fig11.show()


if showGraph:
    """ FIG 12: """
    fig12 = make_subplots(rows=nb_rows, cols=1, subplot_titles=my_subplot_titles)
    k0  = 0
    limx_fig11 = 0
    limy_fig11 = 0

    for cell in myCells:
        k0 += 1
        nbSteps = myCells[cell].nbSteps
        row_nb, col_nb = get_row_col_nb(myCells[cell].labelRefNb, nb_cols)
        #print("k0 = ", k0, " row_nb = ", row_nb," col_nb = ", col_nb)
        #print("P_N1 = ", myCells[cell].P_N1, type(myCells[cell].P_N1), " DN1m = ", myCells[cell].DN1m, type(myCells[cell].DN1m))
        limx_fig12 = max(limx_fig11, myCells[cell].P_N1 * myCells[cell].DN1m )
        limy_fig12 = max(limy_fig11, myCells[cell].countN1s.max() )
        fig12.add_trace(go.Scatter(
            x=( myCells[cell].P_N1 * myCells[cell].DN1m, myCells[cell].P_N1 * myCells[cell].DN1m),
            y=( myCells[cell].countN1s.mean(), myCells[cell].countN1s.mean()),
            name="cell {} ".format(cell),
            line_color='rgba(120,30,30,.2)'),
            row=row_nb,
            col=1)

    # We format the figure
    fig12.update_layout(
        title_text="Cells random migration (end points)",
        title="Cells random migration (end points)",
        height=nb_rows * fig_pixel_definition, # height of the plot
        width=nb_cols * fig_pixel_definition, # The number here is the width of the plot (sum of the widths of all subplots), we thus multiply by the number of columns by the width of a single subplot (3*600)
        font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f")
        )
    fig12.update_xaxes(title="R0_1i = P_N1 * DN1i", range=[0, 1.1*limx_fig12])
    fig12.update_yaxes(title="nb N1s", range=[0, limy_fig12])
    fig12.show()


