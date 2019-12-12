import plotly.figure_factory as ff
import plotly.graph_objects as go
import numpy as np
import csv
from math import sqrt

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
with open('results.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    myCells =  dict() # this is the dictonnary that contains all the cell results
    for row in csv_reader:
        simKey = int(row[0]) # simulationNumber
        if simKey not in myCells:
            myCells[simKey] = CellSimulationResult(simKey)
            myCells[simKey].K_stiffness_central = float(row[1]) # K_stiffness_central
            myCells[simKey].K_stiffness_edge = float(row[2]) # K_stiffness_edge
        else:
            myCells[simKey].nextStep()
        myCells[simKey].append_t( float(row[3]) ) # t (time)
        myCells[simKey].append_x( float(row[4]) ) # x
        myCells[simKey].append_y( float(row[5]) ) # y
        myCells[simKey].append_area( float(row[6]) ) # area
        line_count += 1


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
historgram(distances)

""" Histogram of x, y and distance """
x0 = np.ones(x.shape, dtype=float)
x_centered = x - x[0]*x0

y0 = np.ones(y.shape, dtype=float)
y_centered = y - y[0]*y0

data_list = [x_centered, y_centered]
data_labels = ["x" , "y"]
historgramFromFist(data_list, data_labels)



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
data_labels = ["K_edge is null " , "K edge is non null" , " ALL "]
historgramFromFist(data_list, data_labels)



# tuto: https://plot.ly/python/histograms/
fig1 = go.Figure()
fig1.add_trace(go.Histogram(x=distances_k0))
fig1.add_trace(go.Histogram(x=distances_k))
fig1.add_trace(go.Histogram(x=distances))
fig1.update_layout(barmode='overlay')
fig1.update_layout(title=go.layout.Title(text="Over lay of distributions"))
fig1.show()


fig2 = go.Figure()
for cell in range(100,200):
    fig2.add_trace(go.Scatter(
        x=myCells[cell].x[1:20]-myCells[cell].x[0],
        y=myCells[cell].y[1:20]-myCells[cell].y[0],
        name="cell {} ".format(cell)
        ))

fig2.update_layout(
    title="Cells random migration",
    xaxis_title="migration along the x in µm",
    yaxis_title="migration along the y in µm",
    font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
            )
)

fig2.show()



