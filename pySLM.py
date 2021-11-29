#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.dockarea import *
from pyqtgraph.ptime import time as pgtime
from pyqtgraph.graphicsItems.GradientEditorItem import Gradients

import io_utilities

import matplotlib.pyplot as plt
import sys,os
import numpy as np
import numexpr as ne
from functools import partial
import cmath 
import math

pg.mkQApp()
pg.setConfigOption('background', (30, 30, 30))

# Define main window class from template
path = os.path.dirname(os.path.abspath(__file__))
uiFile = os.path.join(path, 'pySLM.ui')
WindowTemplate, TemplateBaseClass = pg.Qt.loadUiType(uiFile)


### BEGIN Modele class ###
class Patterns():
    
    pass

class Network():
    def __init__(self,window_res_x=1920,window_res_y=1080, unit_type='gaussian',shape='square',x0=0,y0=0,theta=0,nb_x=2,nb_y=1,pitch=10,scale=1.0):
        
        self.params = {}
        self.params.update({key:val for (key,val) in locals().items() if key!='self'})
        
        # Create dict of unit_types
        self.unit_types = {}
        for unit_type in ['gaussian','circle','ellipse','rectangle','phi']:
            self.unit_types[unit_type] = {}
        
        #test
        # Create dict of shapes 
        self.shapes = {}
        for shape in ['square','phi']:
            self.shapes[shape] = {}
            
        # Get list of all the centers according to shape and nb_x, nb_y
        self.create_list_centers()
        
        # Modify center list according to x0, y0 and theta
        self.modify_list_centers()
        
        # Instantiate unit_type at each center and store instances in self.dict_units class attribute
        self.load_units()
        
        # Update the data array
        self.update_data()
        
    
    def update_param(self,param,val,unit_number=None,reset=False):
        
        ''' Modifies a parameter and update self.data '''
        
        if not(unit_number):
            assert param in self.params.keys(), f"Param {param} not known for Network instance"
            if not(reset):
                dict_units_BACKUP = self.dict_units
            
            self.params[param] = val
            self.create_list_centers()
            self.modify_list_centers()
            self.load_units()
            # To set again all the units parameters to previous values
                        
            if not(reset):
                for key in self.dict_units.keys():
                    for param in [temp for temp in self.dict_units[key].params.keys() if temp not in ['x0','y0']]:
                        self.dict_units[key].params[param] = dict_units_BACKUP[key].params[param]
                for key in self.dict_units.keys():
                    self.dict_units[key].update_data()            
        else:
            assert param in self.dict_units[f"{self.params['unit_type']}1"].params.keys(), f"Param {param} not known for Unit instance"
            if unit_number=='all':
                for key in self.dict_units.keys():
                    self.dict_units[f"{key}"].params[param] = val
                    self.dict_units[f"{key}"].update_data()
            else:
                self.dict_units[f"{self.params['unit_type']}{unit_number}"].params[param] = val
                self.dict_units[f"{self.params['unit_type']}{unit_number}"].update_data()
        

        
        self.update_data()

    
    def update_data(self):
        
        ''' Update the array of data '''
        
        self.data = np.zeros((self.params['window_res_y'],self.params['window_res_x']))
        
        for key,val in self.dict_units.items():
            self.data += val.data
        
    def load_units(self):
        
        ''' Instantiate unit_type at each center and store instances in self.dict_units class attribute '''
        
        flag = 1
        self.dict_units = {}
        
        x = np.linspace(0, self.params['window_res_x']-1, self.params['window_res_x'])
        y = np.linspace(0, self.params['window_res_y']-1, self.params['window_res_y'])                                 
        x,y = np.meshgrid(x, y)
        
        for unit_index in range(len(self.list_centers[0])):
            
            if self.params['unit_type'] == 'gaussian':
                self.dict_units[f'gaussian{flag}'] = Gaussian(x,y,x0=self.list_centers[0][unit_index],y0=self.list_centers[1][unit_index],parent=self)
            elif self.params['unit_type'] == 'circle':
                self.dict_units[f'circle{flag}'] = Circle(x,y,x0=self.list_centers[0][unit_index],y0=self.list_centers[1][unit_index],parent=self)
            elif self.params['unit_type'] == 'ellipse':
                self.dict_units[f'ellipse{flag}'] = Ellipse(x,y,x0=self.list_centers[0][unit_index],y0=self.list_centers[1][unit_index],parent=self)
            elif self.params['unit_type'] == 'rectangle':
                self.dict_units[f'rectangle{flag}'] = Rectangle(x,y,x0=self.list_centers[0][unit_index],y0=self.list_centers[1][unit_index],parent=self)
            elif self.params['unit_type'] == 'phi':
                self.dict_units[f'phi{flag}'] = Phi(x,y,x0=self.list_centers[0][unit_index],y0=self.list_centers[1][unit_index],parent=self)
            else:
                raise(AttributeError(f"Unit type {self.params['unit_type']} not found"))
            
            flag += 1
        
    def create_list_centers(self):
        
        ''' Calculate all the centers and store them in list_centers a class attribute '''
        
        if self.params['shape'] == 'square':
            self.list_centers = self.get_centers_square()
        elif self.params['shape'] == 'phi':
            self.list_centers = self.get_centers_phi()
        else:
            raise(AttributeError(f"Shape {self.params['shape']} not found"))
        
        self.list_centers_init = self.list_centers
        
    def get_centers_square(self):
        
        ''' Returns a 2D list of all the centers for square shape '''
        
        list_centers_square = [[],[]]
        for indx in range(1,self.params['nb_x']+1):
            for indy in range(1,self.params['nb_y']+1):
                list_centers_square[0].append( indx * self.params['window_res_x']/(self.params['nb_x']+1) )  # x0
                list_centers_square[1].append( indy * self.params['window_res_y']/(self.params['nb_y']+1) )  # y0
            
        return np.array(list_centers_square)
    
    def get_centers_phi(self):
        
        ''' Returns a 2D list of all the centers for phi shape '''
        x0= self.params['window_res_x']/2
        y0= self.params['window_res_y']/2
        II = cmath.sqrt(-1)

        list_centers_phi = [[],[]]

        pitch =(self.params['pitch'])*20
        nb=self.params['nb_x']-4-(self.params['nb_x']%2)
        h =pitch/math.sqrt(2*(1 - math.cos(2*math.pi/nb))) # radius of the circle of the phi
        for i in range(1,nb+1): 
            x = cmath.exp(II*(i)*2*(math.pi)/nb) #2pi/nb = angle at each center is placed on the circle
            list_centers_phi[0].append( x0+h*x.real )  # x0
            list_centers_phi[1].append( y0+h*x.imag )  # y0
        list_centers_phi[0].append(x0+h+(1*pitch) )  # x0 output arm
        list_centers_phi[1].append( y0)  # y0 output arm
        list_centers_phi[0].append(x0+h+(2*pitch) )  # x0 output arm
        list_centers_phi[1].append( y0)  # y0 output arm
        list_centers_phi[0].append(x0-h-(1*pitch) )  # x0 input arm
        list_centers_phi[1].append( y0)  # y0 input arm
        list_centers_phi[0].append(x0-h-(2*pitch) )  # x0 input arm
        list_centers_phi[1].append( y0)  # y0 input arm
        if self.params['nb_x']%2 ==1: # put another center on the oytput arm if nb is odd
                list_centers_phi[0].append(x0+h+(3*pitch) )  # x0 output arm
                list_centers_phi[1].append( y0)  # y0 output arm

        return np.array(list_centers_phi)
        
    def modify_list_centers(self):
        
        ''' Translate and rotate the list of centers '''
        
        # Re-scale
        center = [self.params['window_res_x']/2,self.params['window_res_y']/2]  # Center of rescale is the window center
        list_centers_x = (self.list_centers_init[0]-center[0]) * self.params['scale'] + center[0]
        list_centers_y = (self.list_centers_init[1]-center[1]) * self.params['scale'] + center[1]
        
        # Translation
        list_centers_x = list_centers_x + self.params['x0']
        list_centers_y = list_centers_y + self.params['y0']
        list_centers = [list_centers_x,list_centers_y]
        
        # Rotation
        theta  = self.params['theta'] * np.pi/180  # theta in degree
        center = np.array([[self.params['window_res_x']/2],[self.params['window_res_y']/2]])  # Center of rotation is the window center
        
        self.list_centers = np.add(np.matmul(((np.cos(theta),-np.sin(theta)),(np.sin(theta),np.cos(theta))), np.subtract(list_centers,center)),center)
        

############### BEGIN Unit Classes ###############
class Gaussian():
    
    def __init__(self,x,y, x0=0., y0=0., offset=0, amplitude=140, sigma_x=100, sigma_y=None, theta=0 ,parent=None):
                
        if not sigma_y:
            sigma_y = sigma_x        # for locals()
            self.sigma_y = sigma_x
        else: self.sigma_y = sigma_y
        
        self.params = {}
        self.params.update({key:val for (key,val) in locals().items() if key!='self' and key!='x' and key!='y' and key!='parent'})
        
        self.x = x
        self.y = y
        
        self.utilities_params = {}
        self.utilities_params['x0'] = {'min':0,'max':parent.params['window_res_x'],'step':1}
        self.utilities_params['y0'] = {'min':0,'max':parent.params['window_res_y'],'step':1}
        self.utilities_params['offset']    = {'min':0,'max':255,'step':1}
        self.utilities_params['amplitude'] = {'min':0,'max':255,'step':1}
        self.utilities_params['theta'] = {'min':-180,'max':180,'step':1}
        self.utilities_params['sigma_x']   = {'min':1,'max':int(parent.params['window_res_x']/2),'step':1}
        self.utilities_params['sigma_y']   = {'min':1,'max':int(parent.params['window_res_y']/2),'step':1}
        
        self.update_data()
    
    def update_data(self):
        
        ''' Update the 2D gaussian data '''
        
        self.params['theta'] = self.params['theta'] * np.pi/180  # theta in degree
        
        # Nice speedups using numexpr
        a = ne.evaluate('(cos(theta)**2)/(2*sigma_x**2) + (sin(theta)**2)/(2*sigma_y**2)',local_dict=self.params)
        b = ne.evaluate('-(sin(2*theta))/(4*sigma_x**2) + (sin(2*theta))/(4*sigma_y**2)',local_dict=self.params)
        c = ne.evaluate('(sin(theta)**2)/(2*sigma_x**2) + (cos(theta)**2)/(2*sigma_y**2)',local_dict=self.params)
        self.data = ne.evaluate('offset + amplitude * exp( - (a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) + c*((y-y0)**2)))',local_dict=dict(self.params,**{'a':a,'b':b,'c':c,'x':self.x,'y':self.y}))    


class Ellipse():
    
    def __init__(self,x,y, x0=0., y0=0., offset=0, amplitude=140, radius_x=100, radius_y=100, theta=0,parent=None):
        
        self.params = {}
        self.params.update({key:val for (key,val) in locals().items() if key!='self' and key!='x' and key!='y' and key!='parent'})
        
        self.x = x
        self.y = y
        
        self.utilities_params = {}
        self.utilities_params['x0'] = {'min':0,'max':parent.params['window_res_x'],'step':1}
        self.utilities_params['y0'] = {'min':0,'max':parent.params['window_res_y'],'step':1}
        self.utilities_params['offset']    = {'min':0,'max':255,'step':1}
        self.utilities_params['amplitude'] = {'min':0,'max':255,'step':1}
        self.utilities_params['theta']     = {'min':-180,'max':180,'step':1}
        self.utilities_params['radius_x']  = {'min':1,'max':int(parent.params['window_res_x']/2),'step':1}
        self.utilities_params['radius_y']  = {'min':1,'max':int(parent.params['window_res_y']/2),'step':1}
        
        self.update_data()
        
    def update_data(self):
        
        ''' Update the 2D circle data '''
        
        # Rotation
        theta = self.params['theta'] * np.pi/180  # theta in degree
        X = ne.evaluate("(x-x0)*cos(theta) - (y-y0)*sin(theta)", local_dict=dict(self.params,**{'theta':theta,'x':self.x,'y':self.y}))
        Y = ne.evaluate("(x-x0)*sin(theta) + (y-y0)*cos(theta)", local_dict=dict(self.params,**{'theta':theta,'x':self.x,'y':self.y}))
                
        self.data = ne.evaluate("X**2 / radius_x**2 + Y**2 / radius_y**2",local_dict=dict(self.params,**{'X':X,'Y':Y})) <= 1
        self.data.astype(int)
        
        self.data = self.data * self.params['amplitude'] + self.params['offset']


class Circle():
    
    def __init__(self,x,y, x0=0., y0=0., offset=0, amplitude=140, radius=100,parent=None):
        
        self.params = {}
        self.params.update({key:val for (key,val) in locals().items() if key!='self' and key!='x' and key!='y' and key!='parent'})
        
        self.x = x
        self.y = y
        
        self.utilities_params = {}
        self.utilities_params['x0'] = {'min':0,'max':parent.params['window_res_x'],'step':1}
        self.utilities_params['y0'] = {'min':0,'max':parent.params['window_res_y'],'step':1}
        self.utilities_params['offset']    = {'min':0,'max':255,'step':1}
        self.utilities_params['amplitude'] = {'min':0,'max':255,'step':1}
        self.utilities_params['radius'] = {'min':1,'max':int(parent.params['window_res_x']/2),'step':1}
        
        self.update_data()
        
    def update_data(self):
        
        ''' Update the 2D circle data '''

        self.data = (self.x - self.params['x0'])**2 + (self.y - self.params['y0'])**2 <= self.params['radius']**2
        self.data.astype(int)

        self.data = self.data * self.params['amplitude'] + self.params['offset']
        
class Rectangle():
    
    def __init__(self,x,y, x0=0., y0=0., offset=0, amplitude=140, size_x=100,theta=0,size_y=None,parent=None):
        
        if not size_y: 
            size_y = size_x        # for locals()
            self.size_y = size_x
        else: self.size_y = size_y
        
        self.params = {}
        self.params.update({key:val for (key,val) in locals().items() if key!='self' and key!='x' and key!='y' and key!='parent'})
        
        self.x = x
        self.y = y
        
        self.utilities_params = {}
        self.utilities_params['x0'] = {'min':0,'max':parent.params['window_res_x'],'step':1}
        self.utilities_params['y0'] = {'min':0,'max':parent.params['window_res_y'],'step':1}
        self.utilities_params['offset']    = {'min':0,'max':255,'step':1}
        self.utilities_params['amplitude'] = {'min':0,'max':255,'step':1}
        self.utilities_params['size_x'] = {'min':1,'max':int(parent.params['window_res_x']),'step':1}
        self.utilities_params['size_y'] = {'min':1,'max':int(parent.params['window_res_y']),'step':1}
        self.utilities_params['theta']     = {'min':-180,'max':180,'step':1}
        
        self.update_data()
        
    def update_data(self):
        
        ''' Update the 2D circle data '''
        
        self.data = np.zeros((len(self.x), len(self.x[0])))
        theta = self.params['theta'] * np.pi/180  # theta in degree
        
        X = ne.evaluate("(x-x0)*cos(theta) - (y-y0)*sin(theta)", local_dict=dict(self.params,**{'theta':theta,'x':self.x,'y':self.y}))
        Y = ne.evaluate("(x-x0)*sin(theta) + (y-y0)*cos(theta)", local_dict=dict(self.params,**{'theta':theta,'x':self.x,'y':self.y}))
                
        self.data = (Y + (self.params['size_y']/2.) >=0) & (Y - (self.params['size_y']/2.) <=0) & (X + (self.params['size_x']/2.) >=0) & (X - (self.params['size_x']/2.) <=0)

        self.data = self.data * self.params['amplitude'] + self.params['offset']
        
class Phi():
    
    def __init__(self,x,y, x0=0., y0=0., offset=0, amplitude=140, radius=170,arm_y=50,arm_in=140,arm_out=140,theta=0,hole=70,parent=None):
        
        
        self.params = {}
        self.params.update({key:val for (key,val) in locals().items() if key!='self' and key!='x' and key!='y' and key!='parent'})
        
        self.x = x
        self.y = y
        
        self.utilities_params = {}
        self.utilities_params['x0'] = {'min':0,'max':parent.params['window_res_x'],'step':1}
        self.utilities_params['y0'] = {'min':0,'max':parent.params['window_res_y'],'step':1}
        self.utilities_params['offset']    = {'min':0,'max':255,'step':1}
        self.utilities_params['amplitude'] = {'min':0,'max':255,'step':1}
        self.utilities_params['radius'] = {'min':1,'max':int(parent.params['window_res_x']/2),'step':1}
        self.utilities_params['arm_y'] = {'min':1,'max':int(parent.params['window_res_y']),'step':1}
        self.utilities_params['arm_in'] = {'min':1,'max':int(parent.params['window_res_x']),'step':1}
        self.utilities_params['arm_out'] = {'min':1,'max':int(parent.params['window_res_x']),'step':1}
        self.utilities_params['hole'] ={'min':1,'max':100,'step':1}
        self.utilities_params['theta']     = {'min':-180,'max':180,'step':1}
        
        self.update_data()
        
    def update_data(self):
        
        ''' Update the 2D circle data '''
        
        self.data = np.zeros((len(self.x), len(self.x[0])))


        data1=np.zeros((len(self.x), len(self.x[0])))
        data1=((self.x - self.params['x0'])**2 + (self.y - self.params['y0'])**2 <= self.params['radius']**2) & ((self.x - self.params['x0'])**2 + (self.y - self.params['y0'])**2 >= ((self.params['hole']*self.params['radius'])/100)**2 ) # circle with a hole
        data1.astype(int)

        theta = self.params['theta'] * np.pi/180  # theta in degree     
        X = ne.evaluate("(x-x0)*cos(theta) - (y-y0)*sin(theta)", local_dict=dict(self.params,**{'theta':theta,'x':self.x,'y':self.y}))
        Y = ne.evaluate("(x-x0)*sin(theta) + (y-y0)*cos(theta)", local_dict=dict(self.params,**{'theta':theta,'x':self.x,'y':self.y}))
        
        data2=np.zeros((len(self.x), len(self.x[0])))     
        data2= (Y + (self.params['arm_y']/2.) >=0) & (Y - (self.params['arm_y']/2.) <=0) & (X + self.params['radius'] + self.params['arm_in'] >=0) & (X + self.params['radius'] <=0) # input arm
        
        data3=np.zeros((len(self.x), len(self.x[0])))     
        data3= (Y + (self.params['arm_y']/2.) >=0) & (Y - (self.params['arm_y']/2.) <=0) & (X - self.params['radius'] - self.params['arm_out'] <=0) & (X - self.params['radius'] >=0) # output arm
        
        self.data =data1+data2+data3
        
        self.data = self.data * self.params['amplitude'] + self.params['offset']
        

#def hand_drawing():
    #from matplotlib import pyplot as plt

    #class LineBuilder:
        #def __init__(self, line):
            #self.line = line
            #self.xs = list(line.get_xdata())
            #self.ys = list(line.get_ydata())
            #self.cid = line.figure.canvas.mpl_connect('button_press_event', self)

        #def __call__(self, event):
            #print('click', event)
            #if event.inaxes!=self.line.axes: return
            #self.xs.append(event.xdata)
            #self.ys.append(event.ydata)
            #self.line.set_data(self.xs, self.ys)
            #self.line.figure.canvas.draw()

    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.set_title('click to add points')
    #line, = ax.plot([], [], marker="o", color="r")
    #linebuilder = LineBuilder(line)

    #plt.show()
    
############### END Unit Classes ###############



############### BEGIN MainWindow class ###############

class MainWindow(TemplateBaseClass,Network):
    def __init__(self):

        # Create variables and parameters
        Network.__init__(self)

        # Extra useful attributes
        self.flag_colormaps  = 1
        self.colormaps_list  = ['thermal','yellowy','greyclip','grey','viridis','inferno']
        self.flag_title_bar  = True
        
        # Load UI
        TemplateBaseClass.__init__(self)
        self.setWindowTitle('Spatial Light Modulator (SLM)')

        # Create the main window
        self.ui = WindowTemplate()
        self.ui.setupUi(self)

        # Set main theme from self.window_params['theme']
        if 'theme' in self.__dict__.keys() and self.theme == 'dark':
            QtGui.QApplication.setStyle("Fusion")
            self.palette = self.palette()
            self.palette.setColor(QtGui.QPalette.Window, QtGui.QColor(53, 53, 53))
            self.palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
            self.palette.setColor(QtGui.QPalette.Base, QtGui.QColor(25, 25, 25))
            self.palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53, 53, 53))
            self.palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.black)
            self.palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
            self.palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
            self.palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53))
            self.palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
            self.palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
            self.palette.setColor(QtGui.QPalette.Link, QtGui.QColor(42, 130, 218))
            self.palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
            self.palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
            self.setPalette(self.palette)
 
        ########################## BEGIN figure layout and docks ##########################
        # Dock declaration and initial placement
        self.main_dock_area = self.ui.dock_area
        self.docks = {}
        
        # First dock on main window (colored)
        dock_name_1 = 'colored'
        self.docks[dock_name_1] = {}
        self.docks[dock_name_1]['dock'] = Dock(dock_name_1, size=(500,1000), closable=True)
        self.main_dock_area.addDock(**{'dock':self.docks[dock_name_1]['dock'],'position':'left'})  # key used: 'dock', 'position' and 'relativeTo'
        self.create_ImageView_colored(dock_name_1)

        # Second dock fullscreen out main window (black/white)
        dock_name_2 = 'black_white'
        self.docks[dock_name_2] = {}
        self.docks[dock_name_2]['dock'] = Dock(dock_name_2, size=(500,1000), closable=True)
        self.main_dock_area.addDock(**{'dock':self.docks[dock_name_2]['dock'],'position':'right','relativeTo':dock_name_1})  # key used: 'dock', 'position' and 'relativeTo'
        self.create_ImageView_blackwhite(dock_name_2)
        ########################## END figure layout and docks ##########################
 
        ##############################  BEGIN Trees declaration  ############################
        
        ### BEGIN Main Tree
        self.spinboxes = {}
        self.tree = self.ui.tree
        self.tree.setColumnCount(2)
        self.tree.keyPressEvent = self.keyPressEvent # allow keys catching for focus on trees
        self.tree.setHeaderLabels(['Main',''])
        
        param = 'window_res_x'
        tree_widget_item = pg.TreeWidgetItem([param])
        self.create_spinbox(param,0,5000,1,tree_widget_item)
        param = 'window_res_y'
        tree_widget_item = pg.TreeWidgetItem([param])
        self.create_spinbox(param,0,5000,1,tree_widget_item)

        param = 'nb_x'
        tree_widget_item = pg.TreeWidgetItem([param])
        self.create_spinbox(param,0,20,1,tree_widget_item)
        
        param = 'nb_y'
        tree_widget_item = pg.TreeWidgetItem([param])
        self.create_spinbox(param,0,10,1,tree_widget_item)
        
        param = 'pitch'
        tree_widget_item = pg.TreeWidgetItem([param])
        self.create_spinbox(param,0,20,1,tree_widget_item) 

        
        #test
        self.group_buttons_main = QtGui.QButtonGroup()
        self.group_buttons_main.setExclusive(True)
        for shape in self.shapes.keys():
            self.shapes[shape]['checkbox'] = QtGui.QCheckBox()
            self.group_buttons_main.addButton(self.shapes[shape]['checkbox'], 1)
            temp = pg.TreeWidgetItem([shape])
            temp.setWidget(1, self.shapes[shape]['checkbox'])
            self.tree.addTopLevelItem(temp)
            if shape == self.params['shape']:
                self.shapes[shape]['checkbox'].setChecked(True) # set initial state
            self.shapes[shape]['checkbox'].keyPressEvent = self.keyPressEvent

        self.group_buttons_main.buttonClicked.connect(self.update_checkbox_main)
        ### END Main Tree
        
        
        ### BEGIN Modes Tree
        self.tree_units = self.ui.tree_units
        self.tree_units.setColumnCount(2)
        self.tree_units.keyPressEvent = self.keyPressEvent
        self.tree_units.setHeaderLabels(['Unit types',''])

        # Create a group of buttons to allow "exclusive" behavior
        self.group_buttons_units = QtGui.QButtonGroup()
        self.group_buttons_units.setExclusive(True)
        for unit in self.unit_types.keys():
            self.unit_types[unit]['checkbox'] = QtGui.QCheckBox()
            self.group_buttons_units.addButton(self.unit_types[unit]['checkbox'], 1)
            temp = pg.TreeWidgetItem([unit])
            temp.setWidget(1, self.unit_types[unit]['checkbox'])
            self.tree_units.addTopLevelItem(temp)
            if unit == self.params['unit_type']:
                self.unit_types[unit]['checkbox'].setChecked(True) # set initial state
            self.unit_types[unit]['checkbox'].keyPressEvent = self.keyPressEvent

        self.group_buttons_units.buttonClicked.connect(self.update_checkbox_units)
        ### END Modes Tree
        
        
        ### BEGIN Params Tree
        self.create_param_tree()
        
        # Sub params
        self.create_sub_param_tree()
        ### END Params Tree
        
        ##############################  END Trees declaration  ############################
        
        # Load an existing configration
        if len(sys.argv) > 1:
            option_name = sys.argv[1]
            if option_name != '--pylab':
                assert option_name == '-f', f"option '{option_name}' not understood. Known option is only '-f' with filename as value"
                assert len(sys.argv) == 3, "Your command should look like: python pySLM.py -f params.txt"
                input_file = sys.argv[2]
                print('Loading config file...')
                self = io_utilities.load_config(self,input_file)
                print('Done')
                
        # Update all plots
        self.update_plots()
                
        # Start showing the window
        self.show()
    
    
    def create_param_tree(self):  
        self.sliders           = {}
        self.sliders_spinboxes = {}
        self.tree_params = self.ui.tree_params
        self.tree_params.setColumnCount(3)
        self.tree_params.keyPressEvent = self.keyPressEvent # allow keys catching for focus on trees
        self.tree_params.setHeaderLabels(['Params','',''])
        
        param = 'x0'
        tree_widget_item = pg.TreeWidgetItem([param])
        self.create_slider_tree(param,-int(self.params['window_res_x']/2),int(self.params['window_res_x']/2),1,tree_widget_item)
        param = 'y0'
        tree_widget_item = pg.TreeWidgetItem([param])
        self.create_slider_tree(param,-int(self.params['window_res_y']/2),int(self.params['window_res_y']/2),1,tree_widget_item)
        param = 'theta'
        tree_widget_item = pg.TreeWidgetItem([param])
        self.create_slider_tree(param,-180,180,1,tree_widget_item)
        param = 'scale'
        tree_widget_item = pg.TreeWidgetItem([param])
        self.create_slider_tree(param,0,500,1,tree_widget_item,conv_factor=1.0/100) # conv_factor*param = value (float value, != int value requested by slider)
    
    def create_sub_param_tree(self):
        self.sub_sliders = {}
        self.sub_sliders_spinboxes = {}
        
        if len(self.dict_units.keys()) > 1:
            # For the 'all' group
            unit = list(self.dict_units.keys())[0]
            tree_widget_item = pg.TreeWidgetItem([self.text_num_split(unit)[0]+'_all'])
            self.tree_params.addTopLevelItem(tree_widget_item)
            
            for param in self.dict_units[unit].params.keys():
                if param != 'x0' and param != 'y0':
                    sub_tree_widget_item = pg.TreeWidgetItem([param])
                    self.create_slider_sub_tree_all(unit,param,sub_tree_widget_item,parent_param=tree_widget_item)
            
        # For the independent groups
        for unit in self.dict_units.keys():
            tree_widget_item = pg.TreeWidgetItem([unit])
            self.tree_params.addTopLevelItem(tree_widget_item)
            
            for param in self.dict_units[unit].params.keys():
                sub_tree_widget_item = pg.TreeWidgetItem([param])
                self.create_slider_sub_tree(unit,param,sub_tree_widget_item,parent_param=tree_widget_item)
    
    
    def create_slider_sub_tree_all(self,unit,param,tree_widget_item,parent_param=False):
        self.sub_sliders['all_'+unit+param] = QtGui.QSlider()
        mi = self.dict_units[unit].utilities_params[param]['min']
        ma = self.dict_units[unit].utilities_params[param]['max']
        step = self.dict_units[unit].utilities_params[param]['step']
        self.sub_sliders['all_'+unit+param].setRange(mi,ma)
        self.sub_sliders['all_'+unit+param].setSingleStep(step)                               # integers only
        self.sub_sliders['all_'+unit+param].setOrientation(QtCore.Qt.Orientation.Horizontal)  # horizontale
        conv_factor = int(1 / self.dict_units[unit].utilities_params[param]['step'])
        value = self.dict_units[unit].params[param] * conv_factor
        self.sub_sliders['all_'+unit+param].setValue(int(value))
        self.sub_sliders['all_'+unit+param].valueChanged.connect(partial(self.update_slider_unit_all,unit,conv_factor,param))
        self.sub_sliders_spinboxes['all_'+unit+param] = QtGui.QDoubleSpinBox()
        tree_widget_item.setWidget(2, self.sub_sliders['all_'+unit+param])
        parent_param.addChild(tree_widget_item)
        self.tree_params.addTopLevelItem(tree_widget_item)
        self.sub_sliders_spinboxes['all_'+unit+param] = QtGui.QDoubleSpinBox()
        self.sub_sliders_spinboxes['all_'+unit+param].setRange(conv_factor * mi,conv_factor * ma)
        self.sub_sliders_spinboxes['all_'+unit+param].setSingleStep(conv_factor * step)              
        self.sub_sliders_spinboxes['all_'+unit+param].setValue(value)
        self.sub_sliders_spinboxes['all_'+unit+param].setKeyboardTracking(False) # emit signal only when enter is pressed
        tree_widget_item.setWidget(1, self.sub_sliders_spinboxes['all_'+unit+param])
        self.sub_sliders_spinboxes['all_'+unit+param].valueChanged.connect(partial(self.update_all_slider_from_spinbox,unit,conv_factor,param))
        parent_param.addChild(tree_widget_item)
    
    
    def update_slider_unit_all(self,unit,conv_factor,param):
        ivalue = self.sub_sliders['all_'+unit+param].value()
        value = ivalue*conv_factor
        self.sub_sliders_spinboxes['all_'+unit+param].setValue(ivalue)
        for a_unit in self.dict_units.keys():
            self.sub_sliders_spinboxes[a_unit+param].setValue(value)
        if(value != self.dict_units[unit].params[param]):    
            self.update_param(param,value,'all')
            self.update_plots()
            
    def update_all_slider_from_spinbox(self,unit,conv_factor,param):
        value = self.sub_sliders_spinboxes['all_'+unit+param].value()
        ivalue = int(value/conv_factor)
        self.sub_sliders['all_'+unit+param].setValue(ivalue)
        for a_unit in self.dict_units.keys():
            self.update_sub_slider_from_spinbox(a_unit,conv_factor,param)
        unit_number = self.text_num_split(unit)[-1]
        if(value != self.dict_units[unit].params[param]):
            self.update_param(param,value,'all')
            self.update_plots()
        
    def create_slider_sub_tree(self,unit,param,tree_widget_item,parent_param=False):
        self.sub_sliders[unit+param] = QtGui.QSlider()
        mi = self.dict_units[unit].utilities_params[param]['min']
        ma = self.dict_units[unit].utilities_params[param]['max']
        step = self.dict_units[unit].utilities_params[param]['step']
        self.sub_sliders[unit+param].setRange(mi,ma)
        self.sub_sliders[unit+param].setSingleStep(step)                               # integers only
        self.sub_sliders[unit+param].setOrientation(QtCore.Qt.Orientation.Horizontal)  # horizontale
        conv_factor = int(1 / step)
        value = self.dict_units[unit].params[param] * conv_factor
        self.sub_sliders[unit+param].setValue(int(value))
        self.sub_sliders[unit+param].valueChanged.connect(partial(self.update_slider_unit,unit,conv_factor,param))
        tree_widget_item.setWidget(2, self.sub_sliders[unit+param])
        self.sub_sliders_spinboxes[unit+param] = QtGui.QDoubleSpinBox()
        self.sub_sliders_spinboxes[unit+param].setRange(conv_factor * mi,conv_factor * ma)
        self.sub_sliders_spinboxes[unit+param].setSingleStep(conv_factor * step)              
        self.sub_sliders_spinboxes[unit+param].setValue(value)
        self.sub_sliders_spinboxes[unit+param].setKeyboardTracking(False) # emit signal only when enter is pressed
        tree_widget_item.setWidget(1, self.sub_sliders_spinboxes[unit+param])
        self.sub_sliders_spinboxes[unit+param].valueChanged.connect(partial(self.update_sub_slider_from_spinbox,unit,conv_factor,param))
        parent_param.addChild(tree_widget_item)
    
    def update_slider_unit(self,unit,conv_factor,param):
        ivalue = self.sub_sliders[unit+param].value()
        value = ivalue*conv_factor
        self.sub_sliders_spinboxes[unit+param].setValue(ivalue)
        unit_number = self.text_num_split(unit)[-1]
        if(value != self.dict_units[unit].params[param]):
            self.update_param(param,value,unit_number)
            self.update_plots()
        
    def update_sub_slider_from_spinbox(self,unit,conv_factor,param):
        value = self.sub_sliders_spinboxes[unit+param].value()
        ivalue = int(value/conv_factor)
        self.sub_sliders[unit+param].setValue(ivalue)
        unit_number = self.text_num_split(unit)[-1]
        if(value != self.dict_units[unit].params[param]):
            self.update_param(param,value,unit_number)
            self.update_plots()
        

    def create_slider_tree(self,param,mi,ma,step,tree_widget_item,conv_factor=1):
        self.sliders[param] = QtGui.QSlider()
        self.sliders[param].setRange(mi,ma)
        self.sliders[param].setSingleStep(step)                               # integers only
        self.sliders[param].setOrientation(QtCore.Qt.Orientation.Horizontal)  # horizontale
        value = self.params[param]
        ivalue = int(value / conv_factor)
        self.sliders[param].setValue(ivalue)
        self.sliders[param].valueChanged.connect(partial(self.update_slider,conv_factor,param))
        tree_widget_item.setWidget(2, self.sliders[param])
        #self.tree_params.addTopLevelItem(tree_widget_item)
        self.sliders_spinboxes[param] = QtGui.QDoubleSpinBox()
        self.sliders_spinboxes[param].setRange(conv_factor * mi,conv_factor * ma)
        self.sliders_spinboxes[param].setSingleStep(conv_factor * step)              
        self.sliders_spinboxes[param].setValue(value)
        self.sliders_spinboxes[param].setKeyboardTracking(False) # emit signal only when enter is pressed
        tree_widget_item.setWidget(1, self.sliders_spinboxes[param])
        self.tree_params.addTopLevelItem(tree_widget_item)
        self.sliders_spinboxes[param].valueChanged.connect(partial(self.update_slider_from_spinbox,conv_factor,param))
    
    def update_slider(self,conv_factor,param):
        ivalue = self.sliders[param].value()
        value = ivalue*conv_factor
        self.sliders_spinboxes[param].setValue(value)
        self.update_param(param,value)
        self.update_plots()
        
    def update_slider_from_spinbox(self,conv_factor,param):
        value = self.sliders_spinboxes[param].value()
        ivalue = int(value/conv_factor)
        self.sliders[param].setValue(ivalue)
        self.update_param(param,value)
        self.update_plots()
    
    def create_spinbox(self,param,mi,ma,step,tree_widget_item):
        
        ''' Create spinboxes of the "Main" params '''
        
        self.spinboxes[param] = QtGui.QSpinBox()
        self.spinboxes[param].setRange(mi,ma)
        self.spinboxes[param].setSingleStep(step)
        self.spinboxes[param].setValue(self.params[param])
        self.spinboxes[param].setKeyboardTracking(False) # emit signal only when enter is pressed
        self.spinboxes[param].valueChanged.connect(partial(self.update_spinbox,param))
        tree_widget_item.setWidget(1, self.spinboxes[param])
        self.tree.addTopLevelItem(tree_widget_item)
    
    def update_spinbox(self,param):
        value = int(self.spinboxes[param].value())
        self.update_param(param,value,reset=True)
        self.erase_unit_numbers('colored')
        self.write_unit_number('colored')
        self.update_plots()
        self.tree_params.clear()
        self.tree_params.reset()
        self.create_param_tree()
        self.create_sub_param_tree()
    
    def update_plots(self):
        for dock_name in self.docks.keys():
            self.docks[dock_name]['actual_plot'].setImage(self.data)
        self.update_unit_texts('colored')
        self.repaint_all_plots()
        
    def repaint_all_plots(self):
        for dock_name in self.docks.keys():
            if 'actual_plot' in self.docks[dock_name]:
                self.docks[dock_name]['actual_plot'].setLevels(0,255)
                self.docks[dock_name]['actual_plot'].repaint()

    def update_checkbox_units(self):
        for unit in self.unit_types.keys():
            if self.unit_types[unit]['checkbox'].isChecked():                
                self.update_param('unit_type',unit,reset=True)
        self.update_plots()
        self.tree_params.clear()
        self.tree_params.reset()
        self.create_param_tree()
        self.create_sub_param_tree()


    def update_checkbox_main(self):
        for shape in self.shapes.keys():
            if self.shapes[shape]['checkbox'].isChecked():                
                self.update_param('shape',shape,reset=True)
        self.update_plots()
        self.tree_params.clear()
        self.tree_params.reset()
        self.create_param_tree()
        self.create_sub_param_tree()
    
    def update_images_colormap(self):
        self.flag_colormaps += 1
        cmap_name = self.colormaps_list[np.mod(self.flag_colormaps,len(self.colormaps_list))]
        gradient = Gradients[cmap_name]
        cmap     = pg.ColorMap(pos=[c[0] for c in gradient['ticks']],color=[c[1] for c in gradient['ticks']], mode=gradient['mode'])
        self.docks['colored']['actual_plot'].setColorMap(cmap)
        self.repaint_all_plots()

    def create_ImageView_colored(self,dock_name):
        # Item for displaying image data
        pl = pg.PlotItem()  # to get axis
        img = pg.ImageItem(axisOrder='row-major')  # to rotate 90 degree

        # Create an ImageView Widget
        self.docks[dock_name]['actual_plot'] = pg.ImageView(view=pl,imageItem=img)
        self.docks[dock_name]['actual_plot'].setLevels(0,255)
        # Set initial states
        self.docks[dock_name]['actual_plot'].view.setRange(xRange=(0,self.params['window_res_x']),yRange=(0,self.params['window_res_y']))
        self.docks[dock_name]['actual_plot'].view.invertY(False)
        self.docks[dock_name]['actual_plot'].view.setAspectLocked(True)
        self.docks[dock_name]['actual_plot'].view.setMouseEnabled(x=False,y=False)
        self.docks[dock_name]['actual_plot'].view.disableAutoRange(True)
        self.docks[dock_name]['actual_plot'].ui.menuBtn.hide()
        #self.docks[dock_name]['actual_plot'].ui.menuBtn.show()
        #self.docks[dock_name]['actual_plot'].ui.histogram.hide()
        #self.docks[dock_name]['actual_plot'].ui.roiBtn.hide()
        
        # Set text box
        self.write_unit_number(dock_name)
        self.docks[dock_name]['actual_plot'].view.setXRange(0,self.params['window_res_x'])
        
        # Set colormap to be used
        gradient = Gradients[self.colormaps_list[self.flag_colormaps]]
        cmap = pg.ColorMap(pos=[c[0] for c in gradient['ticks']],color=[c[1] for c in gradient['ticks']], mode=gradient['mode'])
        self.docks[dock_name]['actual_plot'].setColorMap(cmap)
        
        self.docks[dock_name]['dock'].addWidget(self.docks[dock_name]['actual_plot'])
        self.docks[dock_name]['actual_plot'].keyPressEvent = self.keyPressEvent
    
    def write_unit_number(self,dock_name):
        self.dict_unit_texts = {}
        for unit in self.dict_units.keys():
            unit_number = self.text_num_split(unit)[-1]
            self.dict_unit_texts[f'{dock_name}_{unit_number}'] = pg.TextItem(html=f'<span style="color: #B3B3B3">{unit_number}</span>', anchor=(0.5,0.5), angle=0, border='#B3B3B3', fill=(0, 0, 255, 0))
            self.docks[dock_name]['actual_plot'].addItem(self.dict_unit_texts[f'{dock_name}_{unit_number}'])
            self.dict_unit_texts[f'{dock_name}_{unit_number}'].setPos(self.dict_units[unit].params['x0'], self.dict_units[unit].params['y0'])
    def erase_unit_numbers(self,dock_name):
        for key in self.dict_unit_texts.keys():
            self.docks[dock_name]['actual_plot'].removeItem(self.dict_unit_texts[key])
    def update_unit_texts(self,dock_name):
        for unit in self.dict_units.keys():
            unit_number = self.text_num_split(unit)[-1]
            self.dict_unit_texts[f'{dock_name}_{unit_number}'].setPos(self.dict_units[unit].params['x0'], self.dict_units[unit].params['y0'])
            
    def create_ImageView_blackwhite(self,dock_name):
        # Item for displaying image data
        pl = pg.PlotItem()  # to get axis
        pl.hideAxis('bottom')
        pl.hideAxis('left')
        img = pg.ImageItem(axisOrder='row-major')  # to rotate 90 degree
        
        # Create an ImageView Widget
        self.docks[dock_name]['actual_plot'] = pg.ImageView(view=pl,imageItem=img)
        self.docks[dock_name]['actual_plot'].setLevels(0,255)
        # Set initial states
        self.docks[dock_name]['actual_plot'].view.invertY(False)
        self.docks[dock_name]['actual_plot'].view.setAspectLocked(True)
        self.docks[dock_name]['actual_plot'].view.setMouseEnabled(x=False,y=False)
        self.docks[dock_name]['actual_plot'].view.disableAutoRange(True)
        self.docks[dock_name]['actual_plot'].ui.menuBtn.hide()
        self.docks[dock_name]['actual_plot'].ui.histogram.hide()
        self.docks[dock_name]['actual_plot'].ui.roiBtn.hide()

        # Set colormap to be used
        gradient = Gradients['grey']
        cmap = pg.ColorMap(pos=[c[0] for c in gradient['ticks']],color=[c[1] for c in gradient['ticks']], mode=gradient['mode'])
        self.docks[dock_name]['actual_plot'].setColorMap(cmap)
        
        self.docks[dock_name]['dock'].addWidget(self.docks[dock_name]['actual_plot'])
        self.docks[dock_name]['actual_plot'].keyPressEvent = self.keyPressEvent
    

    def keyPressEvent(self, event):
        """ Set keyboard interactions """

        try: key = event.text()
        except: key = event  # allow calling keys programatically
        
        if key == 'q':
            sys.exit()
        elif key == 'l':
            print('Loading config file...')
            io_utilities.load_config(self)
            print('Done')
        elif key == 's':
            print('Saving config file...')
            io_utilities.save_config(self)
            print('Done')
        elif key == 'f':
            self.flag_title_bar = not(self.flag_title_bar)
            if self.flag_title_bar: self.docks['black_white']['dock'].showTitleBar()
            else: self.docks['black_white']['dock'].hideTitleBar()
        elif key == 'c':
            self.update_images_colormap()
        else:
            if key != "" and event.key() != QtCore.Qt.Key_Return:
                print(f'Keyboard event "{key}" not None')



    def text_num_split(self,item):
        for index, letter in enumerate(item, 0):
            if letter.isdigit():
                return [item[:index],item[index:]]



## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    ### BEGIN Start the window ###
    win = MainWindow()

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
