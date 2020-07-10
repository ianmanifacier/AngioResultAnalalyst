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
nb_cols = 4


parameter_ranges, parameter_list, my_subplot_titles, myCells, multiCellTA = loadData()
nb_rows, nb_cols = getNb_rows_cols(multiCellTA.get_label_list(), nb_cols)

print(my_subplot_titles)

#https://realpython.com/numpy-scipy-pandas-correlation-python/


mesurement_list = ("distance of travel", "distance from origin", "tortuosity", "average curved speed","average speed (straight line)")





#print("curved_length mean : ", multiCellTA["curved_length"]["P_N1"].mean())

#z = np.random.uniform(-1,1,size=(len(parameter_list), len(mesurement_list)))

#print("x= ", multiCellTA.append("R0_1i",1,10))
#print("y= ",multiCellTA.append("tortuosity",1,10))


if True:
    correlation_matrix = "empty" #np.empty([0,len(multiCellTA.parameters_path_analysis_keys_list)], dtype=float)
    p_value_matrix = "empty"
    for p in multiCellTA.get_label_list():
        z_row = np.empty([1,0], dtype=float)
        p_value_row = np.empty([1,0], dtype=float)
        for q in multiCellTA.parameters_path_analysis_keys_list:
            pearson_xy = scipy.stats.pearsonr(x=multiCellTA[p],y=multiCellTA[q][p])
            #z_xy = np.corrcoef(x=multiCellTA[p],y=multiCellTA[q][p])[0,1]
            #print(z_xy)
            #print("pearson_xy =", pearson_xy)
            if pearson_correlation:
                z_xy = pearson_xy[0] # correlation value
                p_value_xy = pearson_xy[1] # p-value
                title_str = 'Pearson CONCLUSION heat map'
            elif rank_correlatation:
                z_xy = pearson_xy[0] # correlation value
                p_value_xy = pearson_xy[1] # p-value
                title_str = 'Pearson CONCLUSION heat map'
            z_row = np.append(z_row, [z_xy])
            p_value_row = np.append(p_value_row, [p_value_xy])
        if correlation_matrix == "empty":
            correlation_matrix = [z_row]
            p_value_matrix = [p_value_row]
        else:
            correlation_matrix = np.append(correlation_matrix,[z_row], axis=0)
            p_value_matrix = np.append(p_value_matrix, [p_value_row], axis=0)

    """
    z_xy = np.corrcoef(x=multiCellTA.append("R0_1i",1,10),y=multiCellTA.append("tortuosity",1,10))[0,1]
    z_row = np.append(z_row, [z_xy])
    """


    print("shape row = ", z_row.shape)
    print("shape correlation_matrix = ", correlation_matrix.shape)
    print("correlation matrix:", type(correlation_matrix))

    """ Drawing CORRELATION figure"""
    fig_correlation = go.Figure(
            data=go.Heatmap(
            z=correlation_matrix,
            x=mesurement_list,
            y=multiCellTA.get_label_list(),
            xgap=0.1,
            ygap=0.1,
            zmin=-1,
            zmax=1,
            colorscale=mycolorscale()))
    fig_correlation.update_layout(
        title='Correlation heat map',
        font=font_dict, # font properties
        height=3 * fig_pixel_definition, # height of the plot
        width=4 * fig_pixel_definition,) # width of the plot
    fig_correlation.show()


    """ Drawing P-VALUE figure"""
    fig_p_value = go.Figure(
            data=go.Heatmap(
            z=p_value_matrix,
            x=mesurement_list,
            y=multiCellTA.get_label_list(),
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
    fig_PearsonConclusion = go.Figure(
            data=go.Heatmap(
            z= z_PearsonConclusion_matrix,
            x=mesurement_list,
            y=multiCellTA.get_label_list(),
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


