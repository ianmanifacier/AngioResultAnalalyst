#!/usr/bin/python
# coding: utf-8

import plotly.figure_factory as ff
import plotly.graph_objects as go
import numpy as np
import csv
from math import sqrt
from math import floor
from math import acos

class CellSimulationResult:
    """ CellSimulationResult is an object that contains all the results of a simulation
        - time (seconds): stored as t (self.t which is a numpy vertical vector storing ) 
        - position (µm): stored as x, y numpy vertical vectors all x, y coordonates at time t
        - last step of the simulation
    """
    def __init__(self, simulationNumber):
        self.simulationNumber = simulationNumber #0
        self.t = np.empty([0, ], dtype=float) #1
        self.x = np.empty([0, ], dtype=float) #2
        self.y = np.empty([0, ], dtype=float) #3
        self.labelRefNb = int() #4
        self.label = "not defined" #5
        self.getNbNodes = np.empty([0, ], dtype=int) #6
        self.countN1s = np.empty([0, ], dtype=int)  #7
        self.countN2s = np.empty([0, ], dtype=int)  #8
        self.getNbInteractions = np.empty([0, ], dtype=int)  #9
        self.calculateTheLengthOfAllInteractions = np.empty([0, ], dtype=float) #10
        self.calculateTheLengthOfAllB1s = np.empty([0, ], dtype=float) #11
        self.calculateTheLengthOfAllB2s = np.empty([0, ], dtype=float) #12
        self.totalNumberOfNodesDuringCellLifeTime = float() #13
        self.nodeAge_mean = float() #14
        self.nodeAgeAtDeath_mean = float() #15
        self.nodeAgeAtDeath_maximum = float() #16
        self.nbImmatureN1Transitions = int() #17
        self.nbIntermediateN1Transitions = int() #18
        self.nbMatureN1Transitions = int() #19
        self.immatureN1TransitionPeriod_minimum = float() # 20
        self.immatureN1TransitionPeriod_maximum  = float() # 21
        self.immatureN1TransitionPeriod_mean  = float() # 22
        self.intermediateN1TransitionPeriod_minimum = float() #23
        self.intermediateN1TransitionPeriod_maximum = float() #24
        self.intermediateN1TransitionPeriod_mean = float() #25
        self.matureN1TransitionPeriod_minimum = float() #26
        self.matureN1TransitionPeriod_maximum = float() #27
        self.matureN1TransitionPeriod_mean = float() #28
        self.nbSteps = 0
        self.nbExtremities = "not defined"
    def distanceFromOrigin(self):
        x = self.x
        y = self.y
        return sqrt( (x[0]-x[self.nbSteps])**2 + (y[0]-y[self.nbSteps])**2 )
    def curvedDistanceFromOrigin(self):
        x = self.x
        y = self.y
        curved_distance = 0.0
        for i in range(0,self.nbSteps):
            curved_distance = curved_distance + sqrt( (x[i]-x[i+1])**2 + (y[i]-y[i+1])**2 )
        return curved_distance
    def tortuosity(self):
        """Tortuosity
        source : https://en.wikipedia.org/wiki/Tortuosity
        The simplest mathematical method to estimate tortuosity is the arc-chord ratio: the ratio of the length of the curve (C) to the distance between its ends (L):
        # arc-chord ratio: tau =C/L
        Arc-chord ratio equals 1 for a straight line and is infinite for a circle."""
        if self.distanceFromOrigin() == 0.0:
            return 0
        return self.curvedDistanceFromOrigin() / self.distanceFromOrigin()
    def nextStep(self):
        self.nbSteps += 1
    def calculatesMigrationAngles(self):
        theta = np.empty([self.nbSteps-2, ], dtype=float)
        for i in range(1,self.nbSteps):
            ux = self.x[i]-self.x[i-1]
            uy = self.y[i]-self.y[i-1]
            vx = self.x[i+1]-self.x[i]
            vy = self.y[i+1]-self.y[i]
            theta[i-1] = acos( (ux*vx +uy*uy)/( sqrt( (ux*ux+uy*uy) + (vx*vx+vy*vy) ) ))
        return theta


def historgram(x):
    """ histogram(x) makes a histogram
    x is a numpy array width 1 and length n """
    hist_data = [x] #np.transpose(x)]
    group_labels = ['distance'] # name of the dataset
    fig = ff.create_distplot(hist_data, group_labels, bin_size=1)
    fig.show()

def historgramFromFist(data_list, data_labels):
    """ histogram(x) makes a histogram
    x is a numpy array width 1 and length n """
    hist_data = data_list
    group_labels = data_labels # name of the dataset
    fig = ff.create_distplot(hist_data, group_labels, bin_size=1) # 
    fig.show()


############################################################
################# The Script Starts Here ###################
############################################################

def load_results(filename1 = 'results.csv', filename2 = 'parameters.csv'):
    my_subplot_titles = []
    with open(filename1) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        myCells =  dict() # this is the dictonnary that contains all the cell results
        for row in csv_reader:
            simKey = int(row[0]) # simulationNumber
            if simKey not in myCells:
                myCells[simKey] = CellSimulationResult(simKey)
                myCells[simKey].labelRefNb = int(row[4])  # label ref NB
                myCells[simKey].label = str(row[5]) # label (string)
                #my_subplot_titles[myCells[simKey].labelRefNb-1] = myCells[simKey].label
            else:
                myCells[simKey].nextStep()
            myCells[simKey].t = np.append( myCells[simKey].t, float(row[1]) ) # t (time)
            myCells[simKey].x = np.append( myCells[simKey].x, float(row[2]) ) # x
            myCells[simKey].y = np.append( myCells[simKey].y, float(row[3]) ) # y
            myCells[simKey].getNbNodes = np.append( myCells[simKey].getNbNodes, int(row[6])) #6
            myCells[simKey].countN1s = np.append( myCells[simKey].countN1s, int(row[7]))  #7
            myCells[simKey].countN2s = np.append( myCells[simKey].countN2s, int(row[8]))  #8
            myCells[simKey].getNbInteractions = np.append( myCells[simKey].getNbInteractions, int(row[9]))  #9
            myCells[simKey].calculateTheLengthOfAllInteractions = np.append( myCells[simKey].calculateTheLengthOfAllInteractions, float(row[10])) #10
            myCells[simKey].calculateTheLengthOfAllB1s = np.append( myCells[simKey].calculateTheLengthOfAllB1s, float(row[11])) #11
            myCells[simKey].calculateTheLengthOfAllB2s = np.append( myCells[simKey].calculateTheLengthOfAllB2s, float(row[12])) #12
            myCells[simKey].totalNumberOfNodesDuringCellLifeTime = float(row[13]) #13
            myCells[simKey].nodeAge_mean = float(row[14]) #14
            myCells[simKey].nodeAgeAtDeath_mean = float(row[15]) #15
            myCells[simKey].nodeAgeAtDeath_maximum = float(row[16]) #16
            myCells[simKey].nbImmatureN1Transitions = int(row[17]) #17
            myCells[simKey].nbIntermediateN1Transitions = int(row[18]) #18
            myCells[simKey].nbMatureN1Transitions = int(row[19]) #19
            myCells[simKey].immatureN1TransitionPeriod_minimum = float(row[20]) # 20
            myCells[simKey].immatureN1TransitionPeriod_maximum = float(row[21]) # 21
            myCells[simKey].immatureN1TransitionPeriod_mean = float(row[22]) # 22
            myCells[simKey].intermediateN1TransitionPeriod_minimum = float(row[23])#23
            myCells[simKey].intermediateN1TransitionPeriod_maximum = float(row[24])#24
            myCells[simKey].intermediateN1TransitionPeriod_mean = float(row[25])#25
            myCells[simKey].matureN1TransitionPeriod_minimum = float(row[26])#26
            myCells[simKey].matureN1TransitionPeriod_maximum = float(row[27])#27
            myCells[simKey].matureN1TransitionPeriod_mean = float(row[28])#28

            if myCells[simKey].label not in my_subplot_titles:
                my_subplot_titles.append(myCells[simKey].label)
            line_count += 1
        my_subplot_titles = tuple(my_subplot_titles)
    print('Up load Parameters')
    with open(filename2) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count ==0:
                print("list parameters = ", row)
                parameter_list = row[3:33]
            else:
                simKey = int(row[2])
                myCells[simKey].duration = float(row[0])
                myCells[simKey].dt = float(row[1])
                myCells[simKey].simulation_NB = int(row[2])
                myCells[simKey].DN1i = float(row[3])
                myCells[simKey].DN1m = float(row[4])
                myCells[simKey].DN2 = float(row[5])
                myCells[simKey].l1min = float(row[6])
                myCells[simKey].l1max = float(row[7])
                myCells[simKey].l2 = float(row[8])
                myCells[simKey].l2eq = float(row[9])
                myCells[simKey].k10 = float(row[10])
                myCells[simKey].k1 = float(row[11])
                myCells[simKey].k2 = float(row[12])
                myCells[simKey].gamma1 = float(row[13])
                myCells[simKey].B1_tension_only = bool(row[14])
                myCells[simKey].B2_tension_only = bool(row[15])
                myCells[simKey].B1_stretch_maturation_theshold = float(row[16])
                myCells[simKey].alphasat = float(row[17])
                myCells[simKey].alpha0 = float(row[18])
                myCells[simKey].alpha10 = float(row[19])
                myCells[simKey].alpha1i_constant_coefficient = float(row[20])
                myCells[simKey].alpha1i_slope_coefficient = float(row[21])
                myCells[simKey].alpha1m = float(row[22])
                myCells[simKey].alpha2 = float(row[23])
                myCells[simKey].delta = float(row[24])
                myCells[simKey].ruptureForce_N0 = float(row[25])
                myCells[simKey].ruptureForce_N1 = float(row[26])
                myCells[simKey].ruptureForce_N2 = float(row[27])
                myCells[simKey].P_N1 = float(row[28])
                myCells[simKey].P_N20 = float(row[29])
                myCells[simKey].P_N2i = float(row[30])
                myCells[simKey].P_N2m = float(row[31])
                #myCells[simKey].labelRefNb = row[32]
                #myCells[simKey].label = row[33]
                myCells[simKey].label_x = row[34]
                myCells[simKey].label_y = row[35]
            
                if myCells[simKey].simulation_NB != simKey:
                    print("Warning: Error loading file")
            line_count = line_count + 1
        return parameter_list, my_subplot_titles, myCells

##############################################################
################# Ploting tools ##############################
##############################################################

def assign_subplot(rows, cols, nvalue):
    """Return assigns the row and column of the element to be ploted based on it's nvalue.
        
    Inputs:
        - rows (int) : number of rows of subplots
        - columns (int) : number of columns of suplot
        - nvalue (int) : number assigned to the category
    Output:
        - row (int)
        - col (int)
    """

    if nvalue > (rows*cols):
        print('value out of bounds')
        exit()
        # raise Error('value out of bounds')
    else:
        col = nvalue % cols + 1
        row = floor(nvalue/cols) + 1
        return row, col

def assign_subplot_row(cols, nvalue):
    return floor(nvalue/cols) + 1

def assign_subplot_col(cols, nvalue):
    return nvalue % cols + 1

###########################################################
#### Mygration patern Analysis ####
###########################################################

# Analysis of angle distribution

# is the distribution flat?

# is the distribution gaussian (wrap arround gaussian)

# is there a bias ?

# is there persistance ?