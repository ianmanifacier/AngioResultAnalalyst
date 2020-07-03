#!/usr/bin/python
# coding: utf-8

# external library imports
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
import scipy.stats

# local library imports
#from result_reader import plotmigrationpatterns as myplt
from plot_tools_and_loader import *


pearson_correlation = True
rank_correlatation = True

#https://realpython.com/numpy-scipy-pandas-correlation-python/

mesurement_list = ("distance of travel", "distance from origin", "tortuosity", "average curved speed","average speed (straight line)")

class MultiCellTrajectoryAnalysis(dict):
    def __init__(self, myCells, ipls):
        self.labelRefNb_list = list()
        self.parameters_path_analysis_keys_list =list()
        # path results (initialization dictionnary)
        self.curved_length = dict() 
        self.distance_from_origin = dict() 
        self.tortuosity = dict()
        self.curved_speed = dict()
        self.speed = dict()
        """
        i_count = 0
        index_ipls = 0
        """
        for cell in myCells:
            """
            if cell == 1:
                myCells[cell].label = ipls[index_ipls]
            elif i_count >= 100:
                index_ipls = index_ipls + 1
                i_count = 0
                myCells[cell].label = ipls[index_ipls]
            else:
                myCells[cell].label = ipls[index_ipls]
            
            i_count = i_count + 1
            """

            labelRefNb = myCells[cell].label_x #myCells[cell].labelRefNb
            if labelRefNb not in self.curved_length.keys():
                # parameters related (initialization dictionnary key)
                if labelRefNb == "l1":
                    self[labelRefNb] =  np.array([getattr(myCells[cell],"l1min")], dtype=float)
                    #self["l1max"] =  np.array([getattr(myCells[cell],"l1max")], dtype=float)
                else:
                    self[labelRefNb] =  np.array([getattr(myCells[cell],labelRefNb)], dtype=float)
                # path results (initialization dictionnary key)
                self.curved_length[labelRefNb] = np.array( [myCells[cell].curvedDistanceFromOrigin()], dtype=float)
                self.distance_from_origin[labelRefNb] = np.array( [myCells[cell].distanceFromOrigin()], dtype=float)
                self.tortuosity[labelRefNb] = np.array( [myCells[cell].tortuosity()], dtype=float)
                self.curved_speed[labelRefNb]  = np.array( [myCells[cell].curvedDistanceFromOrigin()/(myCells[cell].nbSteps*myCells[cell].dt)], dtype=float) 
                self.speed[labelRefNb] = np.array( [myCells[cell].distanceFromOrigin()/(myCells[cell].nbSteps*myCells[cell].dt)], dtype=float)
            else:
                # parameters related (append)
                if labelRefNb == "l1":
                    self[labelRefNb] = np.append( self[labelRefNb], [getattr(myCells[cell],"l1min")]) # making the values accessible using dictionnary properties
                    #self["l1max"] = np.append( self["l1max"], [getattr(myCells[cell],"l1max")])
                else:
                    self[labelRefNb] = np.append( self[labelRefNb], [getattr(myCells[cell],labelRefNb)]) # making the values accessible using dictionnary properties
                # path results (append)
                self.curved_length[labelRefNb] = np.append(self.curved_length[labelRefNb], [myCells[cell].curvedDistanceFromOrigin()])
                self.distance_from_origin[labelRefNb] = np.append(self.distance_from_origin[labelRefNb], [myCells[cell].distanceFromOrigin()])
                self.tortuosity[labelRefNb] = np.append(self.tortuosity[labelRefNb], [myCells[cell].tortuosity()])
                self.curved_speed[labelRefNb] = np.append(self.curved_speed[labelRefNb], [myCells[cell].curvedDistanceFromOrigin() / (myCells[cell].nbSteps*myCells[cell].dt)])
                self.speed[labelRefNb] = np.append(self.speed[labelRefNb], [myCells[cell].distanceFromOrigin()/(myCells[cell].nbSteps*myCells[cell].dt)])
        # path analysis parameters results (making the values accessible using dictionnary properties) 
        self["curved_length"] = self.curved_length
        self["distance_from_origin"] = self.distance_from_origin
        self["tortuosity"] = self.tortuosity
        self["curved_speed"] = self.curved_speed
        self["speed"] = self.speed
        self.parameters_path_analysis_keys_list = ["curved_length","distance_from_origin","tortuosity","curved_speed","speed"]
        self.labelRefNb_list = list(self.curved_length.keys())
        if self.labelRefNb_list == ipls:
            print("OK")
        else:
            print("input parameter lists do not match")
            print(ipls)
            print(self.labelRefNb_list)
    def appendByLabelRefnb(self, key, first_labelRfnb, last_labelRfnb):
        my_array = self[key][first_labelRfnb]
        for i in range(first_labelRfnb+1, last_labelRfnb+1):
            my_array = np.append(my_array, self[key][i])
        return my_array


def mycolorscale():
    cs = list()
    for i in np.linspace(0, 1, num=21):
        if float(i) < 0.5:
            cs.append([float(i), "rgb(" + str(255*(0.5-i)/0.5) + ",0,0)"])
        elif float(i) >0.5:
            cs.append([float(i), "rgb(0," + str(255*(i-0.5)/0.5)+ ",0)"])
            #print("cs =" + str(cs) )
    return cs

def mycolorscaleSimplified(correlation_threshold = 0.5):
    cs = list()
    for i in np.linspace(0, 1, num=21):
        if float(i) < 0.5-0.5*correlation_threshold:
            cs.append([float(i), "rgb(" + str(255*(0.5-i)/0.5) + ",0,0)"])
        elif float(i) > (0.5+0.5*correlation_threshold):
            cs.append([float(i), "rgb(0," + str(255*(i-0.5)/0.5)+ ",0)"])
            #print("cs =" + str(cs) )
        else:
            cs.append([float(i), "rgb(0,0,0)"])
    return cs

def mycolorscale_P_values():
    cs = list()
    for i in np.linspace(0, 1, num=201):
        if float(i) < (1-0.997):
            cs.append([float(i), "rgb(5,177,77)"])
        elif float(i) <(1-0.95):
            cs.append([float(i), "rgb(246,142,43)"])
        elif float(i) <(1-0.68):
            cs.append([float(i), "rgb(165,36,38)"])
        else:
            cs.append([float(i), "rgb(0,0,0)"])
        #print("cs =" + str(cs) )
    return cs



input_parameters_list =    ["DN1i","DN1m","DN2",
                            "l1","l2","l2presstress",
                            "P_N1","P_N20","P_N2i","P_N2m",
                            "k10","k1","k2",
                            "gamma1","delta",
                            "alpha0","alpha10",
                            "alpha1i_slope_coefficient","alphasat",
                            "alpha1m","alpha2",
                            "ruptureForce_N1","ruptureForce_N2"]


multiCellTA = MultiCellTrajectoryAnalysis(myCells, input_parameters_list)
#print("curved_length mean : ", multiCellTA["curved_length"]["P_N1"].mean())

#z = np.random.uniform(-1,1,size=(len(parameter_list), len(mesurement_list)))

#print("x= ", multiCellTA.append("R0_1i",1,10))
#print("y= ",multiCellTA.append("tortuosity",1,10))

correlation_matrix = "empty" #np.empty([0,len(multiCellTA.parameters_path_analysis_keys_list)], dtype=float)
p_value_matrix = "empty"
for p in multiCellTA.labelRefNb_list:
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
        elif rank_correlatation:
            z_xy = pearson_xy[0] # correlation value
            p_value_xy = pearson_xy[1] # p-value
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


""" Drawing CORRELATION figure"""
fig_correlation = go.Figure(
        data=go.Heatmap(
        z=correlation_matrix,
        x=mesurement_list,
        y=multiCellTA.labelRefNb_list,
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
        y=multiCellTA.labelRefNb_list,
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
        y=multiCellTA.labelRefNb_list,
        xgap=0.1,
        ygap=0.1,
        zmin=-1,
        zmax=1,
        colorscale=mycolorscaleSimplified()))

fig_PearsonConclusion.update_layout(
    title='Pearson CONCLUSION heat map',
    font=font_dict, # font properties
    height=3 * fig_pixel_definition, # height of the plot
    width=4 * fig_pixel_definition,) # width of the plot

fig_PearsonConclusion.show()


