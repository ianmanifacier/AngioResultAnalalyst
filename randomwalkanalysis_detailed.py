# coding: utf-8

# random walk analysis
# external library imports
import plotly.tools as tls
import plotly.graph_objects as go
from plotly.figure_factory import create_distplot
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import pandas as pd
import os
import scipy.stats
from math import isnan

from result_reader.result_loader import historgram as myHistogram
from result_reader.result_loader import historgramFromList as myHistogramFromList

# local library imports
#from result_reader import plotmigrationpatterns as myplt
from plot_tools_and_loader import MultiCellTrajectoryAnalysis
from plot_tools_and_loader import font_dict
from plot_tools_and_loader import get_row_col_nb
from plot_tools_and_loader import getNb_rows_cols
from plot_tools_and_loader import loadData
from plot_tools_and_loader import fig_pixel_definition
from plot_tools_and_loader import mycolorscale
from plot_tools_and_loader import mycolorscaleSimplified
from plot_tools_and_loader import mycolorscale_P_values
from result_reader.tools_statistic import isThisDistributionNormal

from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

from matplotlib import pyplot as plt

# Lets clear the command line
os.system("cls")
pd.options.plotting.backend = "plotly"

"""
There are several types of random walk, here is a non exaustive liste:
- Lattice random walk (on lattice single step move in a random up, down, left or right direction)
- Pearson walkk (random direction + fixed step length)
- Lévy fligth random walk (random direction + random step lengths)
- Shrinking (shorter and shorter displacement and steps over time.)
"""

# Root Mean Square displacement Xrms (Pearson random walk)

# fixed step size
# no bias
# no correlation


# Role of the spatial dimention d
# the random walker will be moving a n dimention sphere
# we can thus look at the density of visited cites


# Lets find the distribution representing the distance per step
# Lets therefore plot the histogram of the distances


#multiCellTA = MultiCellTrajectoryAnalysis(myCells, use_detailed_label=True)


limxy = 50

nb_cols = 10
parameter_ranges, parameter_list, my_subplot_titles, myCells, multiCellTA = loadData(use_detailed_label=True)
nb_rows, nb_cols = getNb_rows_cols(multiCellTA.label_detailed_list, nb_cols)
print("nb_rows : ", nb_rows, "  nb_cols : ", nb_cols) 


def leastSQ(X,y, use_sklearn=False):
    X = X.reshape((len(X), 1))
    #
    if use_sklearn:
        lr = LinearRegression()
        lr.fit(X, y)
        w = lr.coef_[0]
    else:
        n = X.shape[1]
        r = np.linalg.matrix_rank(X)

        U, sigma, VT = np.linalg.svd(X, full_matrices=False)
        D_plus = np.diag(np.hstack([1/sigma[:r], np.zeros(n-r)]))
        V = VT.T

        X_plus = V.dot(D_plus).dot(U.T)
        w = X_plus.dot(y)
        np.linalg.lstsq(X, y, rcond=None)
    error = np.linalg.norm(X.dot(w) - y, ord=2) ** 2
    x_slope = X.reshape((len(X), ))
    y_slope = (w*X).reshape((len(X), ))
    return x_slope, y_slope, error #

def plotLeastSQ(X,y, nb_rows, nb_cols, fig_index):
    x_slope, y_slope, error = leastSQ(X,y)
    plt.subplot(nb_rows, nb_cols, fig_index)
    plt.scatter(X, y, s=0.2, marker='.', c='b')
    if error < 1:
        plt.plot(x_slope, y_slope, c='green')
    return error

""" Trajectories (for each detailed label) """
if False:
    fig1 = make_subplots(rows=nb_rows, cols=nb_cols, subplot_titles=multiCellTA.label_detailed_list)
    print(len(multiCellTA.label_detailed_list))
    for i in range(0,len(multiCellTA.label_detailed_list)):
        label = multiCellTA.label_detailed_list[i]
        row_nb, col_nb = get_row_col_nb(i+1, nb_cols)
        x = multiCellTA.x[label]
        y = multiCellTA.y[label]
        for j in range(0,len(multiCellTA.y[label])):
            fig1.add_trace(go.Scatter(
                x=x[j][-2:-1]-x[j][1],
                y=y[j][-2:-1]-y[j][1]),
                row=row_nb,
                col=col_nb)
        print("row: ", row_nb, "   col: ", col_nb, "  index Ref nb: ", i)

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


""" Distribution of distances """
if False:
    # tuto: https://plot.ly/python/histograms/
    fig_hist_1 = make_subplots(rows=nb_rows, cols=nb_cols, subplot_titles=multiCellTA.label_detailed_list)
    for i in range(0,len(multiCellTA.label_detailed_list)):
        label = multiCellTA.label_detailed_list[i]
        row_nb, col_nb = get_row_col_nb(i+1, nb_cols)
        distances = multiCellTA.distance_from_origin[label]
        fig_hist_1.add_trace(
                        go.Histogram(x=distances),
                        row=row_nb,
                        col=col_nb)
    #fig_hist_1.add_trace(go.Histogram(x=distances_k))
    #fig_hist_1.add_trace(go.Histogram(x=distances))
    fig_hist_1.update_layout(barmode='overlay')
    fig_hist_1.update_layout(
                        title_text="Distribution distances (end points)",
                        title="Distribution distances (end points)",
                        height=nb_rows * fig_pixel_definition, # height of the plot
                        width=nb_cols * fig_pixel_definition) # The number here is the width of the plot (sum of the widths of all subplots)
    fig_hist_1.update_xaxes(range=[0, 25])
    fig_hist_1.update_yaxes(range=[0, 25])
    fig_hist_1.show()


""" Distribution X and Y values """
if True:
    fig_hist_2 = make_subplots(rows=nb_rows, cols=nb_cols, subplot_titles=multiCellTA.label_detailed_list)
    for i in range(0,len(multiCellTA.label_detailed_list)):
        label = multiCellTA.label_detailed_list[i]
        row_nb, col_nb = get_row_col_nb(i+1, nb_cols)
        # we do for x
        color_x = 'rgb(100, 0, 0, 0.5)'
        final_x = multiCellTA.x_final[label]
        try:
            create_distplot([final_x], ["x"])
        except:
            newplot = go.Histogram(x=final_x, marker=dict(color=color_x) )
        else:
            newplot = create_distplot([final_x], ["x"], colors=[color_x] )
            newplot = newplot.data[0]
        fig_hist_2.add_trace(newplot,
                        row=row_nb,
                        col=col_nb)
        # we do for y
        color_y = 'rgb(0, 100, 0, 0.5)'
        final_y = multiCellTA.y_final[label]
        try:
            create_distplot([final_y], ["y"])
        except:
            newplot = go.Histogram(x=final_y, marker=dict(color=color_y) )
        else:
            newplot = create_distplot([final_y], ["y"], colors=[color_y] )
            newplot = newplot.data[0]
        fig_hist_2.add_trace(newplot,
                        row=row_nb,
                        col=col_nb)
    
    fig_hist_2.update_layout(barmode='overlay')
    fig_hist_2.update_layout(
            title_text="Cells random migration (end points)",
            title="Cells random migration (end points)",
            height=nb_rows * fig_pixel_definition, # height of the plot
            width=nb_cols * fig_pixel_definition) # The number here is the width of the plot (sum of the widths of all subplots)
    fig_hist_2.update_xaxes(range=[-10, 10])
    fig_hist_2.update_yaxes(range=[0, 1])
    fig_hist_2.show()


pearson_correlation = True
rank_correlatation = False

if False:
    correlation_matrix = "empty" #np.empty([0,len(multiCellTA.parameters_path_analysis_keys_list)], dtype=float)
    p_value_matrix = "empty"
    row_i = 0
    for label in multiCellTA.get_label_list():
        row_i = row_i+1
        if row_i == 1:
            z_row = np.empty([1,0], dtype=float)
            p_value_row = np.empty([1,0], dtype=float)
        pearson_xy = scipy.stats.pearsonr(x=multiCellTA.x_final[label],y=multiCellTA.y_final[label])
        if isnan(pearson_xy[0]) or isnan(pearson_xy[1]):
            print("label: ", label, "  pearson_xy: ", pearson_xy)
            pearson_xy= (0.1, 0.1)
            print("label: ", label, "  pearson_xy: ", pearson_xy)
        z_xy = pearson_xy[0] # correlation value
        p_value_xy = pearson_xy[1] # p-value
        z_row = np.append(z_row, [z_xy])
        p_value_row = np.append(p_value_row, [p_value_xy])
        #
        if row_i>=nb_cols:
            if correlation_matrix == "empty":
                correlation_matrix = [z_row]
                p_value_matrix = [p_value_row]
            else:
                correlation_matrix = np.append(correlation_matrix,[z_row], axis=0)
                p_value_matrix = np.append(p_value_matrix, [p_value_row], axis=0)
            row_i = 0

    x_label_list = list(range(1,nb_cols+1))
    #correlation_matrix = correlation_matrix[0:23][0:4]
    print("correlation matrix:", type(correlation_matrix))
    print("shape row = ", z_row.shape)
    print("shape correlation_matrix = ", correlation_matrix.shape)
    print("lenght x labels : ", len(x_label_list))
    print("length y labels : ", len(multiCellTA.label_list))
    #print(multiCellTA.label_list)
    for i in x_label_list:
        x_label_list[i-1] = 'col '+str(i)
    print(x_label_list)
    #x_label_list = ['a','b','c','d','e','f','g','h','i','j']
    """ Drawing CORRELATION figure"""
    fig_correlation = go.Figure(
            data=go.Heatmap(
                z=correlation_matrix,
                x=x_label_list,
                y=multiCellTA.label_list,
                xgap=0.1,
                ygap=0.1,
                zmin=-1,
                zmax=1,
                colorscale=mycolorscale()))
    fig_correlation.update_layout(
        title='Correlation heat map between x and y',
        font=font_dict, # font properties
        height=3 * fig_pixel_definition, # height of the plot
        width=4 * fig_pixel_definition,) # width of the plot
    fig_correlation.show()

    """ Drawing P-VALUE figure"""
    fig_p_value = go.Figure(
            data=go.Heatmap(
            z=p_value_matrix,
            x=x_label_list,
            y=multiCellTA.label_list,
            xgap=0.2,
            ygap=0.2,
            zmin=0,
            zmax=1,
            colorscale=mycolorscale_P_values()))

    fig_p_value.update_layout(
        title='P-value heat map',
        font=font_dict, # font properties
        height=3 * fig_pixel_definition, # height of the plot
        width=4 * fig_pixel_definition,) # width of the plot

    fig_p_value.show()

    p_value_threshold = 0.05 # 5% chance that the result may be due to chance (the higher this value the more we accept error due to noise)
    correlation_threshold = 0.5 # R²>=0.5

    z_PearsonConclusion_matrix = correlation_matrix * ( ((correlation_matrix > correlation_threshold) + (correlation_matrix < -correlation_threshold)) * (p_value_matrix < p_value_threshold) )


    """ Drawing Pearson Conclusion figure"""
    title_str = 'CONCLUSION HEAT MAP of Pearson Correlation between x and y'
    fig_PearsonConclusion = go.Figure(
            data=go.Heatmap(
            z= z_PearsonConclusion_matrix,
            x=x_label_list,
            y=multiCellTA.label_list,
            xgap=0.1,
            ygap=0.1,
            zmin=-1,
            zmax=1,
            colorscale=mycolorscaleSimplified()))

    fig_PearsonConclusion.update_layout(
        title=title_str,
        font=font_dict, # font properties
        height=3 * fig_pixel_definition, # height of the plot
        width=4 * fig_pixel_definition,) # width of the plot

    fig_PearsonConclusion.show()



if True:
    """ X AXIS:  Distribution analysis """
    normal_matrix_x = "empty"
    row_i = 0
    for dtl_label in multiCellTA.get_label_list():
        row_i = row_i+1
        if row_i == 1:
            z_row = np.empty([1,0], dtype=float)
        #
        data_x = multiCellTA.x_final[dtl_label]
        normal_bool = isThisDistributionNormal(data_x, alpha=0.05, quiet=True)
        if normal_bool == True:
            z_xy = 1
        else:
            z_xy = 0
        z_row = np.append(z_row, [z_xy])
        #
        if row_i>=nb_cols:
            if normal_matrix_x == "empty":
                normal_matrix_x = [z_row]
            else:
                normal_matrix_x = np.append(normal_matrix_x, [z_row], axis=0)
            row_i = 0

    x_label_list = list(range(1,nb_cols+1))
    for i in x_label_list:
        x_label_list[i-1] = 'col '+str(i)


    """ X AXIS: Drawing Normality HEAT MAP """
    fig_correlation = go.Figure(
            data=go.Heatmap(
                z=normal_matrix_x,
                x=x_label_list,
                y=multiCellTA.label_list,
                xgap=0.1,
                ygap=0.1,
                zmin=0,
                zmax=1,
                colorscale=mycolorscale()))
    fig_correlation.update_layout(
        title='X AXIS:NORMALITY HEAT MAP of final position on x axis (green: normaly distributed, Red: not normaly distributed)',
        font=font_dict, # font properties
        height=3 * fig_pixel_definition, # height of the plot
        width=4 * fig_pixel_definition,) # width of the plot
    fig_correlation.show()


    """ Y AXIS:  Distribution analysis """
    normal_matrix_y = "empty"
    row_i = 0
    for dtl_label in multiCellTA.get_label_list():
        row_i = row_i+1
        if row_i == 1:
            z_row = np.empty([1,0], dtype=float)
        #
        data_y = multiCellTA.y_final[dtl_label]
        normal_bool = isThisDistributionNormal(data_y, alpha=0.05, quiet=True)
        if normal_bool == True:
            z_xy = 1
        else:
            z_xy = 0
        z_row = np.append(z_row, [z_xy])
        #
        if row_i>=nb_cols:
            if normal_matrix_y == "empty":
                normal_matrix_y = [z_row]
            else:
                normal_matrix_y = np.append(normal_matrix_y, [z_row], axis=0)
            row_i = 0
    x_label_list = list(range(1,nb_cols+1))
    for i in x_label_list:
        x_label_list[i-1] = 'col '+str(i)

    """ Y AXIS:  Drawing Normality HEAT MAP """
    fig_correlation = go.Figure(
            data=go.Heatmap(
                z=normal_matrix_y,
                x=x_label_list,
                y=multiCellTA.label_list,
                xgap=0.1,
                ygap=0.1,
                zmin=0,
                zmax=1,
                colorscale=mycolorscale()))
    fig_correlation.update_layout(
        title='Y AXIS: NORMALITY HEAT MAP of final position on Y axis (green: normaly distributed, Red: not normaly distributed)',
        font=font_dict, # font properties
        height=3 * fig_pixel_definition, # height of the plot
        width=4 * fig_pixel_definition,) # width of the plot
    fig_correlation.show()

    """ BOTH AXIS:  Drawing Normality HEAT MAP """
    
    normal_matrix_xy = normal_matrix_x * normal_matrix_y

    fig_correlation = go.Figure(
            data=go.Heatmap(
                z=normal_matrix_xy,
                x=x_label_list,
                y=multiCellTA.label_list,
                xgap=0.1,
                ygap=0.1,
                zmin=0,
                zmax=1,
                colorscale=mycolorscale()))
    fig_correlation.update_layout(
        title='BOTH AXIS: NORMALITY HEAT MAP of final position on both axis (green: normaly distributed, Red: not normaly distributed)',
        font=font_dict, # font properties
        height=3 * fig_pixel_definition, # height of the plot
        width=4 * fig_pixel_definition,) # width of the plot
    fig_correlation.show()





if False:
    """ X AXIS:  Biased analysis """
    bias_minimum = 0
    bias_maximum = 0
    bias_matrix_x = "empty"
    row_i = 0
    for dtl_label in multiCellTA.get_label_list():
        row_i = row_i+1
        if row_i == 1:
            z_row = np.empty([1,0], dtype=float)
        #
        z_xy = np.mean(multiCellTA.x_final[dtl_label])
        bias_maximum = max(bias_maximum, z_xy)
        bias_minimum = min(bias_minimum, z_xy)
        z_row = np.append(z_row, [z_xy])
        #
        if row_i>=nb_cols:
            if bias_matrix_x == "empty":
                bias_matrix_x = [z_row]
            else:
                bias_matrix_x = np.append(bias_matrix_x, [z_row], axis=0)
            row_i = 0
    print("X AXIS:")
    print("bias_minimum = ", bias_minimum)
    print("bias_maximum = ", bias_maximum)

    x_label_list = list(range(1,nb_cols+1))
    for i in x_label_list:
        x_label_list[i-1] = 'col '+str(i)

    """ X AXIS: Drawing Normality HEAT MAP """
    fig_correlation = go.Figure(
            data=go.Heatmap(
                z=bias_matrix_x,
                x=x_label_list,
                y=multiCellTA.label_list,
                xgap=0.1,
                ygap=0.1,
                zmin=-1,
                zmax=1,
                colorscale=mycolorscale()))
    fig_correlation.update_layout(
        title='X AXIS:NORMALITY HEAT MAP of final position on x axis (green: normaly distributed, Red: not normaly distributed)',
        font=font_dict, # font properties
        height=3 * fig_pixel_definition, # height of the plot
        width=4 * fig_pixel_definition,) # width of the plot
    fig_correlation.show()


if False:
    """ Y AXIS:  Biased analysis """
    bias_minimum = 0
    bias_maximum = 0
    bias_matrix_y = "empty"
    row_i = 0
    for dtl_label in multiCellTA.get_label_list():
        row_i = row_i+1
        if row_i == 1:
            z_row = np.empty([1,0], dtype=float)
        #
        z_xy = np.mean(multiCellTA.y_final[dtl_label])
        bias_maximum = max(bias_maximum, z_xy)
        bias_minimum = min(bias_minimum, z_xy)
        z_row = np.append(z_row, [z_xy])
        #
        if row_i>=nb_cols:
            if bias_matrix_y == "empty":
                bias_matrix_y = [z_row]
            else:
                bias_matrix_y = np.append(bias_matrix_y, [z_row], axis=0)
            row_i = 0
    print("Y AXIS:")
    print("bias_minimum = ", bias_minimum)
    print("bias_maximum = ", bias_maximum)

    x_label_list = list(range(1,nb_cols+1))
    for i in x_label_list:
        x_label_list[i-1] = 'col '+str(i)

    """ Y AXIS: Drawing Normality HEAT MAP """
    fig_correlation = go.Figure(
            data=go.Heatmap(
                z=bias_matrix_y,
                x=x_label_list,
                y=multiCellTA.label_list,
                xgap=0.1,
                ygap=0.1,
                zmin=-1,
                zmax=1,
                colorscale=mycolorscale()))
    fig_correlation.update_layout(
        title='Y AXIS:NORMALITY HEAT MAP of final position on x axis (green: normaly distributed, Red: not normaly distributed)',
        font=font_dict, # font properties
        height=3 * fig_pixel_definition, # height of the plot
        width=4 * fig_pixel_definition,) # width of the plot
    fig_correlation.show()



    """ DIFFUSION RATE """
if False:
    DiffuRate_maximum = 0
    DiffuRate_matrix_y = "empty"
    row_i = 0
    for dtl_label in multiCellTA.get_label_list():
        row_i = row_i+1
        if row_i == 1:
            z_row = np.empty([1,0], dtype=float)
        #
        z_xy = multiCellTA.DiffusionRate(dtl_label)
        DiffuRate_maximum = max(DiffuRate_maximum, z_xy)
        z_row = np.append(z_row, [z_xy])
        #
        if row_i>=nb_cols:
            if DiffuRate_matrix_y == "empty":
                DiffuRate_matrix_y = [z_row]
            else:
                DiffuRate_matrix_y = np.append(DiffuRate_matrix_y, [z_row], axis=0)
            row_i = 0
    print("Y AXIS:")
    print("DiffuRate_maximum = ", DiffuRate_maximum)

    x_label_list = list(range(1,nb_cols+1))
    for i in x_label_list:
        x_label_list[i-1] = 'col '+str(i)

    """ Y AXIS: Drawing Normality HEAT MAP """
    fig_correlation = go.Figure(
            data=go.Heatmap(
                z=DiffuRate_matrix_y,
                x=x_label_list,
                y=multiCellTA.label_list,
                xgap=0.1,
                ygap=0.1,
                zmin=0,
                zmax=DiffuRate_maximum,
                colorscale=mycolorscale()))
    fig_correlation.update_layout(
        title='Y AXIS:NORMALITY HEAT MAP of final position on x axis (green: normaly distributed, Red: not normaly distributed)',
        font=font_dict, # font properties
        height=3 * fig_pixel_definition, # height of the plot
        width=4 * fig_pixel_definition,) # width of the plot
    fig_correlation.show()





""" Mean Square displacement over time (for each detailed label) """
if False:
    fig1 = make_subplots(rows=nb_rows, cols=nb_cols, subplot_titles=multiCellTA.label_detailed_list)
    for i in range(0,len(multiCellTA.label_detailed_list)):
        label = multiCellTA.label_detailed_list[i]
        row_nb, col_nb = get_row_col_nb(i+1, nb_cols)
        t = np.empty([1,0], dtype=float)
        msd = np.empty([1,0], dtype=float)
        for step in range(0,len(multiCellTA.t)):
            t_i, msd_i = multiCellTA.meanSquareDisplacement(label, step=step)
            t = np.append(t,t_i)
            msd = np.append(msd,msd_i)
        fig1.add_trace(go.Scatter(
            x=t,
            y=msd),
            row=row_nb,
            col=col_nb)
        x_slope, y_slope, error = leastSQ(t,msd, use_sklearn=False)
        fig1.add_trace(go.Scatter(
            x=x_slope,
            y=y_slope),
            row=row_nb,
            col=col_nb)
        print("row: ", row_nb, "   col: ", col_nb, "  index Ref nb: ", i, "  error: ", error)

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
    fig1.update_xaxes(title="time in seconds", range=[0, 1.1*t[-1]])
    fig1.update_yaxes(title="Mean Square Displacement", range=[0, 130])
    fig1.show()


""" MATPLOTLIB: Mean Square displacement over time (for each detailed label) """
if False:
    lbl_nb = 5
    for i in range(0,len(multiCellTA.label_detailed_list)):
        label = multiCellTA.label_detailed_list[i]
        row_nb, col_nb = get_row_col_nb(i+1, nb_cols)
        t = np.empty([1,0], dtype=float)
        msd = np.empty([1,0], dtype=float)
        for step in range(0,len(multiCellTA.t)):
            t_i, msd_i = multiCellTA.meanSquareDisplacement(label, step=step)
            t = np.append(t,t_i)
            msd = np.append(msd,msd_i)
        fig_index = i + 1
        error = plotLeastSQ(t,msd, nb_rows, nb_cols, fig_index)
        print("row: ", row_nb, "   col: ", col_nb, "  index Ref nb: ", fig_index, "  error: ", error)
    # We format the figure
    plt.show()

""" MATPLOTLIB: Mean Square displacement over time (for each detailed label) """
if False:
    lbl_nb = 5
    fig_index = 1
    for i in range(lbl_nb*10,lbl_nb*10+10):
        label = multiCellTA.label_detailed_list[i]
        row_nb, col_nb = get_row_col_nb(i+1, nb_cols)
        t = np.empty([1,0], dtype=float)
        msd = np.empty([1,0], dtype=float)
        for step in range(0,len(multiCellTA.t)):
            t_i, msd_i = multiCellTA.meanSquareDisplacement(label, step=step)
            t = np.append(t,t_i)
            msd = np.append(msd,msd_i)
        error = plotLeastSQ(t,msd, 1, nb_cols, fig_index)
        print("row: ", row_nb, "   col: ", col_nb, "  index Ref nb: ", fig_index, "  error: ", error)
        fig_index = fig_index + 1
    # We format the figure
    plt.show()


