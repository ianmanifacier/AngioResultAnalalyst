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
        self.simulationNumber = simulationNumber
        self.K_stiffness_central = "not defined"
        self.K_stiffness_edge = "not defined"
        self.t = np.empty([1000, ], dtype=float)
        self.x = np.empty([1000, ], dtype=float)
        self.y = np.empty([1000, ], dtype=float)
        self.area = np.empty([1000, ], dtype=float)
        self.nbSteps = 0
        self.nbExtremities = "not defined"
    def distance(self):
        x = self.x
        y = self.y
        return sqrt( (x[0]-x[self.nbSteps])**2 + (y[0]-y[self.nbSteps])**2 )
    def append_x(self, x):
        self.x[self.nbSteps] = x
    def append_y(self, y):
        self.y[self.nbSteps] = y
    def append_t(self, t):
        self.t[self.nbSteps] = t
    def append_area(self, area):
        self.area[self.nbSteps] = area
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

def load_results(filename = 'results.csv'):
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        myCells =  dict() # this is the dictonnary that contains all the cell results
        for row in csv_reader:
            simKey = int(row[0]) # simulationNumber
            if simKey not in myCells:
                myCells[simKey] = CellSimulationResult(simKey)
                myCells[simKey].K_stiffness_central = float(row[1]) # K_stiffness_central
                myCells[simKey].K_stiffness_edge = float(row[2]) # K_stiffness_edge
                myCells[simKey].nbExtremities = int(row[7])
            else:
                myCells[simKey].nextStep()
            myCells[simKey].append_t( float(row[3]) ) # t (time)
            myCells[simKey].append_x( float(row[4]) ) # x
            myCells[simKey].append_y( float(row[5]) ) # y
            myCells[simKey].append_area( float(row[6]) ) # area

            line_count += 1
        return myCells

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