#!/usr/bin/python
# coding: utf-8

# external library imports
import sys
import os
import pickle

# Math libraries
import numpy as np
import scipy.stats

# graphic libraries
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# local library imports
from result_reader import result_loader as rl

limxy = 50

fig_pixel_definition = 600
nb_cols = 4

font_dict=dict(
    family="Franklin Gothic",
    size=20,
    color="#000000")           

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

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

def getNb_rows_cols(my_subplot_titles, nb_cols):
    nb_figures = len(my_subplot_titles)
    nb_rows = nb_figures // nb_cols
    if( (nb_cols % nb_figures)>0 ):
        nb_rows = nb_rows + 1
    #print("nb_rows = {}; nb_cols = {}".format(nb_rows, nb_cols))
    return nb_rows, nb_cols


def get_row_col_nb(labelRefNb, nb_cols):
    row_nb = (labelRefNb-1) // nb_cols + 1 # OK
    col_nb = (labelRefNb-1) % nb_cols +1
    #print("label : {}, labelRefNb = {}; row ={}; col = {} ".format(myCells[cell].label, myCells[cell].labelRefNb, row_nb, col_nb))
    return row_nb, col_nb


def read_CSV_storeData(use_detailed_label=False):
    input_parameters_list =    ["DN1i","DN1m","DN2",
                                "l1","l2","l2_prestress",
                                "P_N1","P_N20","P_N2i","P_N2m",
                                "k10","k1","k2",
                                "gamma1", 'B1_stretch_maturation_threshold', "delta",
                                "alpha0","alpha10",
                                "alpha1i_slope_coefficient","alphasat",
                                "alpha1m","alpha2",
                                "ruptureForce_N1","ruptureForce_N2"]
    # define subplot titles
    print(os.getcwd())
    filename = os.path.join(os.getcwd(),'results.csv')
    print("uploading file: {}".format(filename))
    parameter_ranges, parameter_list, my_subplot_titles, myCells = rl.load_results(filename)
    multiCellTA = MultiCellTrajectoryAnalysis(  myCells,
                                                ipls =input_parameters_list,
                                                use_detailed_label=use_detailed_label)
    var_dict = dict()
    var_dict['parameter_ranges'] = parameter_ranges
    var_dict['parameter_list'] = parameter_list
    var_dict['my_subplot_titles'] = my_subplot_titles
    var_dict['myCells'] = myCells
    var_dict['multiCellTA'] = multiCellTA
    if use_detailed_label:
        mcfile = open('MultiCellTA_detailed_pickle', 'wb')
    else:
        mcfile = open('MultiCellTA_undetailed_pickle', 'wb')
    pickle.dump(var_dict,mcfile)
    mcfile.close()

def calculate_MultiCellTA_storeData(use_detailed_label=False):
    input_parameters_list =    ["DN1i","DN1m","DN2",
                                "l1","l2","l2_prestress",
                                "P_N1","P_N20","P_N2i","P_N2m",
                                "k10","k1","k2",
                                "gamma1", 'B1_stretch_maturation_threshold', "delta",
                                "alpha0","alpha10",
                                "alpha1i_slope_coefficient","alphasat",
                                "alpha1m","alpha2",
                                "ruptureForce_N1","ruptureForce_N2"]
    # define subplot titles
    print("curent working directory : ", os.getcwd())
    if use_detailed_label:
        mcfile = open('MultiCellTA_detailed_pickle', 'rb')
    else:
        mcfile = open('MultiCellTA_undetailed_pickle', 'rb')
    var_dict = pickle.load(mcfile)
    mcfile.close()
    myCells = var_dict['myCells']
    var_dict['multiCellTA'] = MultiCellTrajectoryAnalysis(  myCells,
                                                ipls =input_parameters_list,
                                                use_detailed_label=use_detailed_label)
    if use_detailed_label:
        mcfile = open('MultiCellTA_detailed_pickle', 'wb')
    else:
        mcfile = open('MultiCellTA_undetailed_pickle', 'wb')
    pickle.dump(var_dict,mcfile)
    mcfile.close()


def loadData(use_detailed_label=False):
    print("curent working directory : ", os.getcwd())
    if use_detailed_label:
        mcfile = open('MultiCellTA_detailed_pickle', 'rb')
    else:
        mcfile = open('MultiCellTA_undetailed_pickle', 'rb')
    var_dict = pickle.load(mcfile)
    mcfile.close()
    parameter_ranges = var_dict['parameter_ranges']
    parameter_list = var_dict['parameter_list']
    my_subplot_titles = var_dict['my_subplot_titles']
    myCells = var_dict['myCells']
    multiCellTA = var_dict['multiCellTA']
    return parameter_ranges, parameter_list, my_subplot_titles, myCells, multiCellTA


class MultiCellTrajectoryAnalysis(dict):
    def __init__(self, myCells, ipls = [], use_detailed_label=False):
        super().__init__()
        self.use_detailed_label = use_detailed_label
        self.labelRefNb_list = list()
        self.label_list = list()
        self.label_detailed_list = list()
        self.parameters_path_analysis_keys_list =list()
        # path results (initialization dictionnary)
        self.curved_length = dict() 
        self.distance_from_origin = dict()
        self.distance_from_origin_x = dict()
        self.distance_from_origin_y = dict()
        self.x_final = dict()
        self.y_final = dict()
        self.tortuosity = dict()
        self.curved_speed = dict()
        self.speed = dict()
        self.distancePerStep = dict()
        self.distancePerStep_x = dict()
        self.distancePerStep_y = dict()
        self.t = myCells[1].t
        self.x = dict()
        self.y = dict()
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
            #labelRefNb = myCells[cell].label_x
            #labelRefNb = myCells[cell].labelRefNb
            label = myCells[cell].label
            label_detailed = label
            if use_detailed_label:
                    label_detailed = label + " = " + str(getattr(myCells[cell], label))
            #
            labelRefNb = myCells[cell].labelRefNb
            if labelRefNb not in self.labelRefNb_list:
                self.labelRefNb_list.append(labelRefNb)

            label_not_defined = True
            if use_detailed_label:
                if label_detailed in self.label_detailed_list:
                    label_not_defined = False
            else:
                if label_detailed in self.label_list:
                    label_not_defined = False
            #print("label_not_defined = ", label_not_defined)
            #
            if label_not_defined:
                #
                if label not in self.label_list:
                    self.label_list.append(label)
                #
                if label_detailed not in self.label_detailed_list:
                    self.label_detailed_list.append(label_detailed)
                #
                # parameters related (initialization dictionnary key)
                if label == "l1":
                    self[label_detailed] = np.array([getattr(myCells[cell],"l1_mean")], dtype=float)
                    self["l1_min"] = np.array([getattr(myCells[cell],"l1min")], dtype=float)
                    self["l1_max"] = np.array([getattr(myCells[cell],"l1max")], dtype=float)
                    #self["l1max"] = np.array([getattr(myCells[cell],"l1max")], dtype=float)
                else:
                    #self[labelRefNb] =  np.array([getattr(myCells[cell],labelRefNb)], dtype=float)
                    self[label_detailed] =  np.array([getattr(myCells[cell], label)], dtype=float)
                #print("label :", label)
                print("detailed label :", label_detailed)
                # path results (initialization dictionnary key)
                self.curved_length[label_detailed] = np.array( [myCells[cell].curvedDistanceFromOrigin()], dtype=float)
                self.distance_from_origin[label_detailed] = np.array( [myCells[cell].distanceFromOrigin()], dtype=float)
                self.distance_from_origin_x[label_detailed] = np.array( [myCells[cell].distanceFromOrigin_x()], dtype=float)
                self.distance_from_origin_y[label_detailed] = np.array( [myCells[cell].distanceFromOrigin_y()], dtype=float)
                self.x_final[label_detailed] = np.array( [myCells[cell].x[-1]-myCells[cell].x[0]], dtype=float)
                self.y_final[label_detailed] = np.array( [myCells[cell].y[-1]-myCells[cell].y[0]], dtype=float)
                self.x[label_detailed] = [myCells[cell].x]
                self.y[label_detailed] = [myCells[cell].y]
                self.tortuosity[label_detailed] = np.array( [myCells[cell].tortuosity()], dtype=float)
                self.curved_speed[label_detailed]  = np.array( [myCells[cell].curvedDistanceFromOrigin()/(myCells[cell].nbSteps*myCells[cell].dt)], dtype=float) 
                self.speed[label_detailed] = np.array( [myCells[cell].distanceFromOrigin()/(myCells[cell].nbSteps*myCells[cell].dt)], dtype=float)
                self.distancePerStep[label_detailed] = myCells[cell].calulateDistancePerStep()
                self.distancePerStep_x[label_detailed] = myCells[cell].calulateDistancePerStep_x()
                self.distancePerStep_y[label_detailed] = myCells[cell].calulateDistancePerStep_y()
            else:
                # parameters related (append)
                if label == "l1":
                    self[label_detailed] = np.append( self[label_detailed], [getattr(myCells[cell],"l1_mean")]) # making the values accessible using dictionnary properties
                    self["l1_min"] = np.append( self[label_detailed], [getattr(myCells[cell],"l1min")]) # making the values accessible using dictionnary properties
                    self["l1_max"] = np.append( self[label_detailed], [getattr(myCells[cell],"l1max")])
                else:
                    #self[labelRefNb] = np.append( self[labelRefNb], [getattr(myCells[cell],labelRefNb)]) # making the values accessible using dictionnary properties
                    self[label_detailed] = np.append( self[label_detailed], [getattr(myCells[cell], label)]) # making the values accessible using dictionnary properties
                # path results (append)
                self.curved_length[label_detailed] = np.append(self.curved_length[label_detailed], [myCells[cell].curvedDistanceFromOrigin()])
                self.distance_from_origin[label_detailed] = np.append(self.distance_from_origin[label_detailed], [myCells[cell].distanceFromOrigin()])
                self.distance_from_origin_x[label_detailed] = np.append(self.distance_from_origin[label_detailed], [myCells[cell].distanceFromOrigin_x()])
                self.distance_from_origin_y[label_detailed] = np.append(self.distance_from_origin[label_detailed], [myCells[cell].distanceFromOrigin_y()])
                self.x_final[label_detailed] = np.append(self.x_final[label_detailed], [myCells[cell].x[-1]-myCells[cell].x[0]])
                self.y_final[label_detailed] = np.append(self.y_final[label_detailed], [myCells[cell].y[-1]-myCells[cell].y[0]])
                self.x[label_detailed].append(myCells[cell].x)
                self.y[label_detailed].append(myCells[cell].y)
                self.tortuosity[label_detailed] = np.append(self.tortuosity[label_detailed], [myCells[cell].tortuosity()])
                self.curved_speed[label_detailed] = np.append(self.curved_speed[label_detailed], [myCells[cell].curvedDistanceFromOrigin() / (myCells[cell].nbSteps*myCells[cell].dt)])
                self.speed[label_detailed] = np.append(self.speed[label_detailed], [myCells[cell].distanceFromOrigin()/(myCells[cell].nbSteps*myCells[cell].dt)])
                #self.distancePerStep[label_detailed] = np.append(self.distancePerStep[label_detailed], myCells[cell].calulateDistancePerStep())
                #self.distancePerStep_x[label_detailed] = np.append(self.distancePerStep_x[label_detailed], myCells[cell].calulateDistancePerStep_x())
                #self.distancePerStep_y[label_detailed] = np.append(self.distancePerStep_x[label_detailed], myCells[cell].calulateDistancePerStep_y())
        # path analysis parameters results (making the values accessible using dictionnary properties) 
        self["curved_length"] = self.curved_length
        self["distance_from_origin"] = self.distance_from_origin
        self["tortuosity"] = self.tortuosity
        self["curved_speed"] = self.curved_speed
        self["speed"] = self.speed
        self["DiffusionRate"] = self.DiffusionRate
        self["label_list"] = self.label_list
        self.parameters_path_analysis_keys_list = ["curved_length","distance_from_origin","tortuosity","curved_speed","speed"]
        
        #self.label_list = list(self.curved_length.keys())
        if ipls != []:
            print("\n\nComparaison input parameter & label lists: ", end = '')
            if self.get_label_list() == ipls:
                print( bcolors.OKGREEN + "OK" + bcolors.ENDC + "\n")
            else:
                print(bcolors.WARNING + "Input parameter list does not match with label list"  + bcolors.ENDC + "\n")
                print("self.label_list : ")
                for lbl in self.label_list:
                    print("     " + str(lbl))
                print("\nself.labelRefNb_list :", self.labelRefNb_list)
                if self.label_list == ipls:
                    print("\n" + bcolors.OKGREEN + "Ok label_list (undetailed) matched input parameter list" + bcolors.ENDC + "\n")
                else:
                    print("\n" + bcolors.WARNING + "Warning: label_list (undetailed) DID NOT match input parameter list" + bcolors.ENDC +"\n")
    def appendByLabelRefnb(self, key, first_labelRfnb, last_labelRfnb):
        my_array = self[key][first_labelRfnb]
        for i in range(first_labelRfnb+1, last_labelRfnb+1):
            my_array = np.append(my_array, self[key][i])
        return my_array
    def meanSquareDisplacement(self, label, step=-1):
        t = self.t[step]
        x = np.empty([0, ], dtype=float)
        y = np.empty([0, ], dtype=float)
        for cell in range( len(self.x[label]) ):
            xi = self.x[label][cell][step] - self.x[label][cell][0]
            yi = self.y[label][cell][step] - self.y[label][cell][0]
            x = np.append(x,xi)
            y = np.append(y,yi)
        # MSD = Mean Square Displacement
        MSD = np.mean(x*x) + np.mean(y*y)
        return t, MSD
    # def meanSquareDisplacement(self, label):
    #     x = self.distance_from_origin_x[label]
    #     y = self.distance_from_origin_y[label]
    #     # MSD = Mean Square Displacement
    #     MSD = np.mean(x*x) + np.mean(y*y)
    #     return MSD
    def DiffusionRate(self, label):
        n = 2 # number of dimentions
        return self.meanSquareDisplacement(label)/(2*n*np.max(self.t))
    def Bias_x(self, label):
        x = self.distance_from_origin_x[label]
        return np.mean(x)
    def Bias_y(self, label):
        y = self.distance_from_origin_y[label]
        return np.mean(y)
    def get_label_list(self):
        if self.use_detailed_label:
            return self.label_detailed_list
        else:
            return self.label_list